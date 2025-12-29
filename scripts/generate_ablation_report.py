import os
import pickle
import numpy as np
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'
FIGS = REPORTS / 'figures'
REPORTS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

def load_pickle(p):
    try:
        with open(p, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"No se pudo cargar pickle {p}: {e}")
        return None


def main():
    out_csv = REPORTS / 'ablation_mae_by_horizon.csv'
    out_macro = REPORTS / 'ablation_mae_by_macrozona.csv'
    out_md = REPORTS / 'ablation_report.md'

    # Try to load kovae evaluation metrics
    kovae_metrics = load_pickle(DATA / 'kovae_evaluation_metrics.pkl')
    experiments = load_pickle(DATA / 'experiments/experiments_results.pkl')

    # Attempt to extract MAE by horizon for both variants
    # Flexible parsing depending on saved structures
    mae_table = {
        'horizons': [],
        'gamma0': [],
        'gamma0.1': []
    }

    # First, check experiments results (preferred)
    if experiments and isinstance(experiments, dict):
        # expect experiments to contain keys like 'ablation_long' with results
        for k,v in experiments.items():
            if 'ablation' in k.lower():
                # try to find mae arrays
                g0 = v.get('gamma_0.0') or v.get('gamma0') or v.get('0.0')
                g1 = v.get('gamma_0.1') or v.get('gamma0.1') or v.get('0.1')
                if g0 and g1:
                    # assume they have 'mae_by_horizon'
                    h = g0.get('horizons') or list(range(1, len(g0.get('mae_by_horizon',[]))+1))
                    mae_table['horizons'] = h
                    mae_table['gamma0'] = g0.get('mae_by_horizon', [])
                    mae_table['gamma0.1'] = g1.get('mae_by_horizon', [])
                    break

    # fallback: try kovae_metrics
    if (not mae_table['horizons']) and kovae_metrics:
        # kovae_metrics might be dict with keys 'mae_by_horizon' or similar
        if 'ablation' in kovae_metrics:
            a = kovae_metrics['ablation']
            mae_table['horizons'] = a.get('horizons', [])
            mae_table['gamma0'] = a.get('gamma0', [])
            mae_table['gamma0.1'] = a.get('gamma0.1', [])
        else:
            # try common keys
            if 'mae_by_horizon' in kovae_metrics:
                mae_table['horizons'] = list(range(1, len(kovae_metrics['mae_by_horizon'])+1))
                mae_table['gamma0'] = kovae_metrics.get('mae_by_horizon', [])
                # no gamma0.1 available

    # Final fallback: look for saved numpy files for h1 only
    if (not mae_table['horizons']) or (len(mae_table['gamma0'])==0):
        # try to load mae_kovae_h1.npy and mae_aedmd_h1.npy
        try:
            mae_k_h1 = np.load(DATA / 'mae_kovae_h1.npy')
            mae_a_h1 = np.load(DATA / 'mae_aedmd_h1.npy')
            mae_table['horizons'] = [1]
            mae_table['gamma0'] = [float(np.mean(mae_k_h1))]
            mae_table['gamma0.1'] = [float(np.mean(mae_a_h1))]
        except Exception:
            pass

    # Save CSV
    if mae_table['horizons']:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['horizon','mae_gamma0','mae_gamma0.1'])
            for i,h in enumerate(mae_table['horizons']):
                g0 = mae_table['gamma0'][i] if i < len(mae_table['gamma0']) else ''
                g1 = mae_table['gamma0.1'][i] if i < len(mae_table['gamma0.1']) else ''
                writer.writerow([h,g0,g1])
        print(f'Saved CSV: {out_csv}')
    else:
        print('No se encontraron MAE por horizonte para guardar CSV.')

    # Macrozone MAE/bias: try to load kovae_bias_macrozone.pkl
    macro = load_pickle(DATA / 'kovae_bias_macrozone.pkl')
    if macro and isinstance(macro, dict):
        # expect structure: { 'gamma0': {h: array}, 'gamma0.1': ... }
        # flatten to CSV
        with open(out_macro, 'w', newline='') as f:
            writer = csv.writer(f)
            # header
            headers = ['macrozona','horizon','mae_gamma0','mae_gamma0.1','bias_gamma0','bias_gamma0.1']
            writer.writerow(headers)
            zones = macro.get('zones') or list(macro.keys())
            # best-effort parsing
            for zone, vals in macro.items():
                if zone=='zones':
                    continue
                for h,stats in vals.items():
                    mae0 = stats.get('mae_gamma0') if isinstance(stats, dict) else ''
                    mae1 = stats.get('mae_gamma0.1') if isinstance(stats, dict) else ''
                    bias0 = stats.get('bias_gamma0') if isinstance(stats, dict) else ''
                    bias1 = stats.get('bias_gamma0.1') if isinstance(stats, dict) else ''
                    writer.writerow([zone,h,mae0,mae1,bias0,bias1])
        print(f'Saved macrozone CSV: {out_macro}')
    else:
        print('No se encontró información macrozona estructurada en kovae_bias_macrozone.pkl')

    # Compose markdown report
    with open(out_md, 'w', encoding='utf8') as f:
        f.write('# Ablation report: KoVAE gamma=0 vs gamma=0.1\n\n')
        f.write('## Resumen rápido\n')
        f.write('- Figura comparativa: reports/figures/ablation_long_mae_by_horizon.png\n')
        if mae_table['horizons']:
            f.write('\n## MAE por horizonte\n')
            f.write('|Horizon|MAE gamma=0.0|MAE gamma=0.1|\n')
            f.write('|---:|---:|---:|\n')
            for i,h in enumerate(mae_table['horizons']):
                g0 = mae_table['gamma0'][i] if i < len(mae_table['gamma0']) else ''
                g1 = mae_table['gamma0.1'][i] if i < len(mae_table['gamma0.1']) else ''
                f.write(f'|{h}|{g0}|{g1}|\n')
        else:
            f.write('\nNo se encontró MAE por horizonte numerizado.\n')

        f.write('\n## Observaciones y recomendaciones\n')
        f.write('- Los valores de pérdida durante el entrenamiento difieren en escala; usar MAE/RMSE en unidades reales para decisiones.\n')
        f.write('- Enmascarar celdas de varianza casi nula para interpretar R².\n')
        f.write('- Revisar mapas de error y las 20 peores celdas (ya generadas en `data/processed/kovae_worst_cells_examples.csv` si existe).\n')
        f.write('- Evaluar desempeño en eventos extremos (top 5% precipitación).\n')

    print(f'Saved report: {out_md}')


if __name__ == '__main__':
    main()
