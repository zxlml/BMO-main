"""
BMO Example: 
1. Upper-level: Meta-network (MLP) learns to assign loss weights per sample
2. Lower-level: Generator and Discriminator trained with weighted losses
"""
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from model import ResNet32  # Imported from meta_train.py

class MetaNet(nn.Module):
    """Meta-network: Inputs loss value, outputs sample weight"""
    def __init__(self, hidden_size=100, num_layers=1):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), nn.ReLU()]
        for _ in range(num_layers-1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MetaGAN:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create datasets
        self.train_dataset = create_dataset(opt)
        self.meta_dataset = self._create_meta_dataset()  # Create clean meta-dataset
        
        # Initialize models
        self.netG, self.netD = create_model(opt).netG, create_model(opt).netD
        self.meta_net = MetaNet(opt.meta_hidden_size, opt.meta_num_layers).to(self.device)
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_meta = torch.optim.Adam(self.meta_net.parameters(), lr=opt.meta_lr)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    def _create_meta_dataset(self):
        """Create clean meta-dataset subset for meta-learning"""
        # Implement based on actual requirements
        return self.train_dataset  # Simplified example

    def _compute_meta_loss(self, netD, data):
        """Compute meta-loss: Discriminator performance on clean data"""
        real_images = data['data'].to(self.device)
        valid = torch.ones(real_images.size(0), device=self.device)
        
        # Forward pass
        predictions = netD(real_images).view(-1)
        loss = F.binary_cross_entropy_with_logits(predictions, valid)
        return loss

    def update_meta_network(self, real_images):
        """Two-level optimization for meta-network update"""
        # Clone current models
        pseudo_netG = type(self.netG)().to(self.device).load_state_dict(self.netG.state_dict())
        pseudo_netD = type(self.netD)().to(self.device).load_state_dict(self.netD.state_dict())
        
        # --- Inner optimization (weighted by current meta-net) ---
        # Train discriminator
        valid = torch.ones(real_images.size(0), device=self.device)
        fake = torch.zeros(real_images.size(0), device=self.device)
        
        # Real images
        real_pred = pseudo_netD(real_images).view(-1)
        loss_real = self.criterion(real_pred, valid)
        with torch.no_grad():
            weights_real = self.meta_net(loss_real.detach().unsqueeze(1))
        loss_real = (weights_real.squeeze() * loss_real).mean()
        
        # Generated images
        z = torch.randn(real_images.size(0), self.opt.latent_dim, device=self.device)
        fake_images = pseudo_netG(z).detach()
        fake_pred = pseudo_netD(fake_images).view(-1)
        loss_fake = self.criterion(fake_pred, fake)
        with torch.no_grad():
            weights_fake = self.meta_net(loss_fake.detach().unsqueeze(1))
        loss_fake = (weights_fake.squeeze() * loss_fake).mean()
        
        # Update pseudo discriminator
        loss_D = (loss_real + loss_fake) / 2
        pseudo_netD.zero_grad()
        loss_D.backward()
        for param, pseudo_param in zip(self.netD.parameters(), pseudo_netD.parameters()):
            pseudo_param.grad = param.grad.clone()
        self.optimizer_D.step()  # Simulated update
        
        # --- Meta-loss computation ---
        try:
            meta_data = next(self.meta_loader_iter)
        except:
            self.meta_loader_iter = iter(DataLoader(self.meta_dataset, batch_size=self.opt.batch_size, shuffle=True))
            meta_data = next(self.meta_loader_iter)
        
        meta_loss = self._compute_meta_loss(pseudo_netD, meta_data)
        
        # --- Update meta-network ---
        self.optimizer_meta.zero_grad()
        meta_loss.backward()
        self.optimizer_meta.step()
        
        return meta_loss.item()

    def train_epoch(self, epoch, visualizer):
        """Train single epoch"""
        epoch_start_time = time.time()
        for i, data in enumerate(self.train_dataset):
            # Prepare data
            real_images = data['data'].to(self.device)
            batch_size = real_images.size(0)
            
            # --- Train Discriminator ---
            self.optimizer_D.zero_grad()
            
            # Real images
            valid = torch.ones(batch_size, device=self.device)
            real_pred = self.netD(real_images).view(-1)
            loss_real = self.criterion(real_pred, valid)
            
            # Apply meta-network weighting
            with torch.no_grad():
                weights_real = self.meta_net(loss_real.detach().unsqueeze(1))
            loss_real = (weights_real.squeeze() * loss_real).mean()
            
            # Generated images
            z = torch.randn(batch_size, self.opt.latent_dim, device=self.device)
            fake_images = self.netG(z).detach()
            fake = torch.zeros(batch_size, device=self.device)
            fake_pred = self.netD(fake_images).view(-1)
            loss_fake = self.criterion(fake_pred, fake)
            
            # Apply meta-network weighting
            with torch.no_grad():
                weights_fake = self.meta_net(loss_fake.detach().unsqueeze(1))
            loss_fake = (weights_fake.squeeze() * loss_fake).mean()
            
            # Combined loss and backpropagation
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            self.optimizer_D.step()
            
            # --- Train Generator ---
            self.optimizer_G.zero_grad()
            valid = torch.ones(batch_size, device=self.device)
            gen_images = self.netG(z)
            gen_pred = self.netD(gen_images).view(-1)
            loss_G = self.criterion(gen_pred, valid)
            
            # Apply meta-network weighting
            with torch.no_grad():
                weights_G = self.meta_net(loss_G.detach().unsqueeze(1))
            loss_G = (weights_G.squeeze() * loss_G).mean()
            
            loss_G.backward()
            self.optimizer_G.step()
            
            # --- Update meta-network (periodic) ---
            if i % self.opt.meta_interval == 0:
                meta_loss = self.update_meta_network(real_images)
                print(f'Meta Loss: {meta_loss:.4f}')
            
            # Visualization and logging
            if i % self.opt.print_freq == 0:
                errors = {'G': loss_G.item(), 'D_real': loss_real.item(), 'D_fake': loss_fake.item()}
                visualizer.print_current_losses(epoch, i, errors, 0, 0)
                
                if self.opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, i/len(self.train_dataset), errors)

    def train(self):
        """Full training loop"""
        visualizer = Visualizer(self.opt)
        self.meta_loader_iter = iter(DataLoader(self.meta_dataset, batch_size=self.opt.batch_size, shuffle=True))
        
        for epoch in range(self.opt.epoch_count, self.opt.n_epochs + self.opt.n_epochs_decay + 1):
            epoch_start_time = time.time()
            self.train_epoch(epoch, visualizer)
            
            # Model saving
            if epoch % self.opt.save_epoch_freq == 0:
                torch.save(self.netG.state_dict(), f'netG_epoch_{epoch}.pth')
                torch.save(self.netD.state_dict(), f'netD_epoch_{epoch}.pth')
                torch.save(self.meta_net.state_dict(), f'meta_net_epoch_{epoch}.pth')
            
            print(f'End of epoch {epoch}/{self.opt.n_epochs+self.opt.n_epochs_decay} '
                  f'Time: {time.time()-epoch_start_time:.2f}s')


if __name__ == '__main__':
    # Extend training options
    opt = TrainOptions().parse()
    opt.meta_hidden_size = 100      # Meta-network hidden size
    opt.meta_num_layers = 1         # Meta-network layers
    opt.meta_lr = 1e-5              # Meta-learning rate
    opt.meta_interval = 50          # Meta-network update interval
    
    # Initialize and train model
    model = MetaGAN(opt)
    model.train()