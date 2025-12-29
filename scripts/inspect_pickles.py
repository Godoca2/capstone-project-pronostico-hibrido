import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'

def inspect(p):
    print('---', p)
    try:
        with open(p,'rb') as f:
            obj = pickle.load(f)
    except Exception as e:
        print('Cannot load:', e)
        return
    print('Type:', type(obj))
    if isinstance(obj, dict):
        print('Keys:', list(obj.keys())[:50])
        for k in list(obj.keys())[:50]:
            v = obj[k]
            try:
                shp = v.shape
            except Exception:
                shp = None
            print(' ',k, type(v), shp)
    else:
        try:
            print('Shape:', obj.shape)
        except Exception:
            pass

if __name__ == '__main__':
    for name in ['kovae_evaluation_metrics.pkl','forecast_results_2020.pkl','era5_2020_daily_for_kovae.pkl']:
        inspect(DATA / name)
