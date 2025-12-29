import pickle
from pathlib import Path
import numpy as np
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'
REPORTS.mkdir(parents=True, exist_ok=True)


def load(p):
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f'No se pudo cargar {p}: {e}')
        return None


def find_array(obj):
    # Try to find arrays or dicts with predicted/true values
    if isinstance(obj, dict):
        for k in obj.keys():
            if 'pred' in k.lower() or 'yhat' in k.lower() or 'forecast' in k.lower():
                return obj[k]
        # fallback: return first numpy-like value
        for k,v in obj.items():
            if hasattr(v, 'shape'):
                return v
    return None


def main():
    era5 = load(DATA / 'era5_2020_daily_for_kovae.pkl')
    kev = load(DATA / 'kovae_evaluation_metrics.pkl')
    fr = load(DATA / 'forecast_results_2020.pkl')

    # try get y_true from era5
    y_true = None
    if isinstance(era5, dict):
        for k in ['y_test','Y_test','y','Y','y_true']:
            if k in era5:
                y_true = era5[k]
                break
        if y_true is None:
            # try common arrays inside
            for v in era5.values():
                if hasattr(v, 'shape') and v.ndim >= 3:
                    y_true = v
                    break

    # try get preds for kovae
    preds_kovae = None
    if kev:
        preds_kovae = find_array(kev)

    # try get preds for aedmd from forecast_results
    preds_aedmd = None
    if fr and isinstance(fr, dict):
        # forecast_results likely contains per-horizon arrays
        # try keys 'kovae' or 'aedmd'
        for k in fr.keys():
            if 'aedmd' in k.lower() or 'dmd' in k.lower():
                preds_aedmd = fr[k]
            if 'kovae' in k.lower():
                preds_kovae = fr[k]

    # Validation: need y_true and preds_kovae; if preds_aedmd not available, try mae numpy
    if y_true is None:
        print('No se pudo localizar y_true en era5 pickle. Abortando.')
        return

    # Ensure arrays are numpy
    y_true = np.array(y_true)

    # preds_kovae may be dict with 'horizons' or array shape (n,h,w1,w2)
    if preds_kovae is None and kev and isinstance(kev, dict):
        preds_kovae = kev.get('predictions') or kev.get('y_pred') or find_array(kev)

    if preds_kovae is None:
        print('No se encontraron predicciones KoVAE.')
        return

    preds_kovae = np.array(preds_kovae)

    # Align shapes: assume preds shape is (n_samples, horizons, H, W) or (n_samples, H, W)
    if preds_kovae.ndim == 3:
        preds_kovae = preds_kovae[:, None, ...]

    if y_true.ndim == 4:
        # (n_samples, horizons, H, W) or (n_samples, H, W, 1)
        if y_true.shape[1] != preds_kovae.shape[1] and y_true.shape[-1] == 1:
            y_true = y_true[..., 0]
        if y_true.ndim == 4 and y_true.shape[1] == preds_kovae.shape[1]:
            y = y_true
        else:
            # try reduce to horizons last
            y = y_true
    else:
        y = y_true

    # compute threshold for extremes for horizon 1
    # extract ground truth at horizon index 0
    try:
        y_h1 = y[:, 0, ...] if y.ndim >= 3 else y
    except Exception:
        y_h1 = y

    vals = y_h1.flatten()
    thresh = np.nanpercentile(vals, 95)
    mask_ext = y_h1 >= thresh

    # Compute MAE overall and on extremes for horizon 1
    from sklearn.metrics import mean_absolute_error

    def mae_over(preds, truths, mask=None):
        preds = np.array(preds)
        truths = np.array(truths)
        if preds.ndim == truths.ndim-1:
            preds = preds[:, 0, ...]
        pred_flat = preds.flatten()
        true_flat = truths.flatten()
        if mask is not None:
            mask_flat = mask.flatten()
            if mask_flat.sum() == 0:
                return np.nan
            return mean_absolute_error(true_flat[mask_flat], pred_flat[mask_flat])
        return mean_absolute_error(true_flat, pred_flat)

    mae_kovae_all = mae_over(preds_kovae, y)
    mae_kovae_ext = mae_over(preds_kovae, y, mask_ext)

    # Try AE+DMD: if available in data processed mae_aedmd_h1.npy
    mae_aedmd_all = None
    mae_aedmd_ext = None
    try:
        mae_aedmd_h1 = np.load(DATA / 'mae_aedmd_h1.npy')
        mae_aedmd_all = float(np.mean(mae_aedmd_h1))
        # cannot compute extremes from that array; leave NaN
    except Exception:
        pass

    # Save CSV
    out = REPORTS / 'extremes_performance.csv'
    with open(out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model','horizon','mae_all','mae_extremes','notes'])
        writer.writerow(['kovae',1,float(mae_kovae_all),float(mae_kovae_ext),'extremes threshold='+str(thresh)])
        if mae_aedmd_all is not None:
            writer.writerow(['aedmd',1,mae_aedmd_all,'', 'mae from mae_aedmd_h1.npy'])

    print(f'Saved extremes performance CSV: {out}')

    # Top 20 worst cells by absolute error (h=1)
    pred_h1 = preds_kovae[:,0,...] if preds_kovae.ndim>=3 else preds_kovae
    err = np.abs(pred_h1 - y_h1)
    # average error per cell across samples
    cell_err = np.nanmean(err, axis=0)
    # get top20 indices
    flat = cell_err.flatten()
    idx = np.argsort(flat)[-20:][::-1]
    H,W = cell_err.shape
    rows = []
    for i,ind in enumerate(idx):
        r = ind // W
        c = ind % W
        rows.append((i+1, r, c, float(flat[ind])))

    out_top = REPORTS / 'extremes_top20_cells.csv'
    with open(out_top, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank','row','col','mean_abs_error'])
        for r in rows:
            writer.writerow(r)

    print(f'Saved top20 worst cells CSV: {out_top}')


if __name__ == '__main__':
    main()
