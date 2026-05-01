# Fine-grained Analysis on the Stability and Generalization for First-order Bilevel Minimax Optimization
The implementation codes for validating the theoretical generalization bounds of BMO algorithms and investigating the connections to practical settings, e.g., max iterations, step sizes, sample sizes.

## Code & Data Acknowledgement
### Meta Weighting Network
The bilevel framework for automatic instance weighting is adopted from [1] with several modifications (e.g., modify the lower problem as the minimax-form objective, changing the step sizes, max iterations and meta data sizes).

## Generative Adversarial Network
The lower problem with minimax object is built on the sample denoising and generation task [2], where the key modifications include the new iterative update rules for upper MLP and lower generators and discriminators.

## Data Source
### Charlie Chaplin's Frames
The corresponding data can be downloaded from [Chaplin's Frames](https://github.com/zhigang-yao/MFCGAN)

Reference
>  [1] Meta-weight-net: Learning an explicit mapping for sample weighting


```
@article{shu2019meta,
  title={Meta-weight-net: Learning an explicit mapping for sample weighting},
  author={Shu, Jun and Xie, Qi and Yi, Lixuan and Zhao, Qian and Zhou, Sanping and Xu, Zongben and Meng, Deyu},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```

> [2] Manifold fitting with CycleGAN




```
@article{yao2024manifold,
  title={Manifold fitting with CycleGAN},
  author={Yao, Zhigang and Su, Jiaji and Yau, Shing-Tung},
  journal={Proceedings of the National Academy of Sciences},
  volume={121},
  number={5},
  pages={e2311436121},
  year={2024},
  publisher={National Academy of Sciences}
}
```
