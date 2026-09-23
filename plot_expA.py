# -*- coding: utf-8 -*-
"""实验A绘图：合并 expA_detail_m*_s*.csv，按 (m1, k) 聚合画论文图1/4风格曲线。"""
import os
import sys
import glob

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_experiment import RESULTS_DIR


def main():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'expA_detail_m*_s*.csv')))
    if not files:
        print('未找到 expA_detail_m*_s*.csv')
        return
    detail = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    detail.to_csv(os.path.join(RESULTS_DIR, 'expA_detail.csv'), index=False)
    summary = detail.groupby(['m1', 'step'])[
        ['val_error', 'test_error', 'gap', 'gap_raw']].agg(['mean', 'std']).reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, 'expA_summary.csv'), index=False)

    m1s = sorted(detail['m1'].unique())
    metrics = [('val_error', 'Errors in Meta Set'),
               ('test_error', 'Errors in Test Set'),
               ('gap', 'Generalization Gap'),
               ('gap_raw', 'Generalization Gap (raw)')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    for ax, (metric, label) in zip(np.atleast_1d(axes), metrics):
        for m1 in m1s:
            sub = summary[summary['m1'] == m1].sort_values('step')
            ax.errorbar(sub['step'], sub[(metric, 'mean')],
                        yerr=sub[(metric, 'std')].fillna(0), marker='o',
                        capsize=3, label=f'm1={m1}')
        ax.set_xlabel('Inner Iteration K')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend()
        ax.grid(alpha=0.3)
    T = int(detail['step'].count() and len(detail) // max(1, len(m1s)))
    fig.suptitle(f'Exp A: TSGDA-1 (repeats={len(files)})')
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, 'expA.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(summary.to_string())
    print('saved', out)


if __name__ == '__main__':
    main()
