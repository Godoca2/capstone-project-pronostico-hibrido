"""metrics.py
=================
Conjunto de métricas para evaluación de pronósticos hidrológico/climáticos.
Incluye métricas genéricas (MAE, RMSE, R²) y de dominio (NSE, Skill Score,
PBIAS). Todas las funciones aceptan arrays o listas y aplastan la entrada
para evaluación global sobre múltiples estaciones/tiempos.

Notas de uso
------------
* NSE y Skill Score permiten contextualizar el desempeño frente a baseline
 (media y persistencia respectivamente).
* PBIAS reporta sesgo porcentual; se recomienda complementarlo con MAE/RMSE.
* Las métricas añaden `1e-10` en denominadores para estabilidad numérica.

Autor: César Godoy Delaigue (Magíster Data Science UDD - 2025)
"""
import numpy as np
from typing import Union


def mse(y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list]) -> float:
    """Mean Squared Error (MSE).

    Minimiza grandes errores cuadráticamente; sensible a outliers.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean((y_true - y_pred)**2))


def rmse(y_true: Union[np.ndarray, list],
         y_pred: Union[np.ndarray, list]) -> float:
    """Root Mean Squared Error (RMSE).

    Raíz del MSE para mantener unidades originales.
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list]) -> float:
    """Mean Absolute Error (MAE).

    Penaliza todos los errores linealmente; más robusto a outliers que RMSE.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: Union[np.ndarray, list],
             y_pred: Union[np.ndarray, list]) -> float:
    """Coeficiente de determinación R².

    Proporción de varianza explicada por el modelo (1 = perfecto).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - (ss_res / (ss_tot + 1e-10)))


def nse(y_true: Union[np.ndarray, list],
        y_pred: Union[np.ndarray, list]) -> float:
    """Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 (perfecto), 0 (igual a usar la media), <0 (peor que la media).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    numerator = np.sum((y_true - y_pred)**2)
    denominator = np.sum((y_true - np.mean(y_true))**2)
    return float(1 - (numerator / (denominator + 1e-10)))


def pbias(y_true: Union[np.ndarray, list],
          y_pred: Union[np.ndarray, list]) -> float:
    """Percent Bias (PBIAS).

    PBIAS > 0 indica subestimación; < 0 sobreestimación.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return float(100 * np.sum(y_true - y_pred) / (np.sum(y_true) + 1e-10))


def skill_score_persistence(y_true: Union[np.ndarray, list],
                            y_pred: Union[np.ndarray, list]) -> float:
    """Skill Score contra baseline de persistencia.

    SS = 1 - MSE_modelo / MSE_persistencia.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Modelo persistente: y_pred_naive[t] = y_true[t-1]
    y_persist = np.roll(y_true, 1)
    y_persist[0] = y_true[0]  # primer valor no tiene histórico

    mse_model = mse(y_true, y_pred)
    mse_persist = mse(y_true, y_persist)

    return float(1 - (mse_model / (mse_persist + 1e-10)))


def evaluate_all(y_true: Union[np.ndarray, list],
                 y_pred: Union[np.ndarray, list]) -> dict:
    """Calcula métricas agregadas y retorna dict.

    Returns
    -------
    dict
    Keys: MAE, RMSE, R2, NSE, PBIAS, SkillScore.
    """
    return {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'NSE': nse(y_true, y_pred),
        'PBIAS': pbias(y_true, y_pred),
        'SkillScore': skill_score_persistence(y_true, y_pred)
    }


def evaluate_by_event_type(y_true: Union[np.ndarray, list],
                           y_pred: Union[np.ndarray, list],
                           thresholds: dict = None) -> dict:
    """Evalúa métricas por tipo de evento meteorológico.

    Parameters
    ----------
    y_true : array-like
    Valores observados (mm/día)
    y_pred : array-like
    Valores predichos (mm/día)
    thresholds : dict, optional
    Umbrales para clasificar eventos. Default:
    {'dry': 0.1, 'normal': 10.0}
    - Días secos: < dry
    - Días normales: dry <= x < normal
    - Días extremos: >= normal

    Returns
    -------
    dict
    Métricas por categoría: 'dry', 'normal', 'extreme'
    """
    if thresholds is None:
        thresholds = {'dry': 0.1, 'normal': 10.0}

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Clasificar eventos
    dry_mask = y_true < thresholds['dry']
    extreme_mask = y_true >= thresholds['normal']
    normal_mask = ~(dry_mask | extreme_mask)

    results = {}

    for event_type, mask in [('dry', dry_mask),
                             ('normal', normal_mask),
                             ('extreme', extreme_mask)]:
        if mask.sum() > 0:
            results[event_type] = {
                'count': int(mask.sum()),
                'MAE': mae(y_true[mask], y_pred[mask]),
                'RMSE': rmse(y_true[mask], y_pred[mask]),
                'PBIAS': pbias(y_true[mask], y_pred[mask])
            }
        else:
            results[event_type] = {
                'count': 0,
                'MAE': np.nan,
                'RMSE': np.nan,
                'PBIAS': np.nan
            }

    return results


def residual_analysis(y_true: Union[np.ndarray, list],
                      y_pred: Union[np.ndarray, list]) -> dict:
    """Análisis estadístico de residuos.

    Returns
    -------
    dict
    Estadísticas de residuos: mean, std, percentiles (5, 25, 50, 75, 95)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    residuals = y_true - y_pred

    return {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'p5': float(np.percentile(residuals, 5)),
        'p25': float(np.percentile(residuals, 25)),
        'p50': float(np.percentile(residuals, 50)),  # mediana
        'p75': float(np.percentile(residuals, 75)),
        'p95': float(np.percentile(residuals, 95)),
        'skewness': float(_skewness(residuals)),
        'kurtosis': float(_kurtosis(residuals))
    }


def _skewness(x: np.ndarray) -> float:
    """Coeficiente de asimetría (skewness)."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    return float(np.sum((x - mean)**3) / (n * std**3 + 1e-10))


def _kurtosis(x: np.ndarray) -> float:
    """Coeficiente de curtosis (kurtosis)."""
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    return float(np.sum((x - mean)**4) / (n * std**4 + 1e-10) - 3)


def skill_score_climatology(y_true: Union[np.ndarray, list],
                            y_pred: Union[np.ndarray, list],
                            climatology: Union[np.ndarray, list]) -> float:
    """Skill Score contra baseline de climatología.

    Parameters
    ----------
    y_true : array-like
    Valores observados
    y_pred : array-like
    Valores predichos
    climatology : array-like
    Valores climatológicos (media histórica por día del año)

    Returns
    -------
    float
    SS = 1 - MSE_modelo / MSE_climatología
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    climatology = np.asarray(climatology).flatten()

    mse_model = mse(y_true, y_pred)
    mse_clim = mse(y_true, climatology)

    return float(1 - (mse_model / (mse_clim + 1e-10)))
