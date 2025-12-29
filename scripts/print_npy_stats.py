import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'

def show(name):
    p = DATA / name
    try:
        a = np.load(p)
        print(name, 'shape', a.shape, 'mean', float(np.nanmean(a)), 'min', float(np.nanmin(a)), 'max', float(np.nanmax(a)))
    except Exception as e:
        print('Cannot load', name, e)

if __name__ == '__main__':
    for n in ['mae_kovae_h1.npy','mae_aedmd_h1.npy','mae_diff_h1.npy','bias_kovae_h1.npy','bias_aedmd_h1.npy']:
        show(n)
