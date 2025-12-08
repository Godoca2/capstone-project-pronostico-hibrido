"""
Data Loader Unificado para Pipeline de Pronóstico de Precipitaciones
Carga ÚNICAMENTE datos reales ERA5 y derivados.
NO genera ni usa datos sintéticos.
"""
import xarray as xr
import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
from sklearn.preprocessing import StandardScaler
import warnings

# Rutas globales
PROJECT_ROOT = Path(__file__).parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
RAW_DIR = DATA_DIR / 'raw'

# Archivos de datos reales
ERA5_FULL_PATH = PROCESSED_DIR / 'era5_precipitation_chile_full.nc'
ERA5_KOVAE_PATH = PROCESSED_DIR / 'era5_2020_daily_for_kovae.pkl'
FORECAST_RESULTS_PATH = PROCESSED_DIR / 'forecast_results_2020.pkl'


def verify_real_data(file_path: Path) -> bool:
    """
    Verifica que el archivo sea de datos reales (no sintéticos).
    
    Returns:
        True si es archivo real, False si es sintético o no existe
    """
    if not file_path.exists():
        warnings.warn(f"⚠️ Archivo no encontrado: {file_path}")
        return False
    
    # Verificar que NO sea archivo sintético conocido
    synthetic_files = ['precipitation_data.npy', 'precipitation_test.csv']
    if file_path.name in synthetic_files:
        warnings.warn(f"⚠️ Archivo sintético detectado: {file_path.name}")
        return False
    
    return True


def load_era5_full(
    year_filter: Optional[str] = None,
    normalize: bool = True
) -> Tuple[np.ndarray, xr.Dataset, Optional[StandardScaler]]:
    """
    Carga dataset ERA5 completo desde NetCDF.
    
    Args:
        year_filter: '2019', '2020', '2019-2020', None (todos)
        normalize: Si True, aplica StandardScaler
        
    Returns:
        precip_array: (T, lat, lon, 1) normalizado si aplica
        ds: Dataset xarray original
        scaler: StandardScaler usado (None si normalize=False)
    """
    if not verify_real_data(ERA5_FULL_PATH):
        raise FileNotFoundError(f"Datos reales ERA5 no encontrados: {ERA5_FULL_PATH}")
    
    print(f"[LOAD] Cargando ERA5 desde: {ERA5_FULL_PATH.relative_to(PROJECT_ROOT)}")
    ds = xr.open_dataset(ERA5_FULL_PATH)
    
    # Filtrar años si se especifica
    if year_filter:
        if year_filter == '2019':
            ds = ds.sel(valid_time=slice('2019-01-01', '2019-12-31'))
        elif year_filter == '2020':
            ds = ds.sel(valid_time=slice('2020-01-01', '2020-12-31'))
        elif year_filter == '2019-2020':
            ds = ds.sel(valid_time=slice('2019-01-01', '2020-12-31'))
        print(f"  Filtrado a: {year_filter}")
    
    # Extraer precipitación
    precip = ds['tp'].values  # (T, lat, lon)
    n_samples, n_lat, n_lon = precip.shape
    
    print(f"  Shape: {precip.shape}")
    print(f"  Rango: [{precip.min():.4f}, {precip.max():.4f}] mm/día")
    
    # Normalizar si se solicita
    scaler = None
    if normalize:
        precip_flat = precip.reshape(n_samples, -1)
        scaler = StandardScaler()
        precip_norm = scaler.fit_transform(precip_flat)
        precip_array = precip_norm.reshape(n_samples, n_lat, n_lon, 1)
        print(f"  Normalizado: media≈{precip_norm.mean():.2e}, std≈{precip_norm.std():.2f}")
    else:
        precip_array = precip.reshape(n_samples, n_lat, n_lon, 1)
    
    print(f"[OK] ERA5 cargado: {precip_array.shape}")
    return precip_array, ds, scaler


def load_era5_kovae() -> Dict:
    """
    Carga dataset ERA5 2020 preparado para KoVAE.
    
    Returns:
        dict con 'precip_2020', metadatos
    """
    if not verify_real_data(ERA5_KOVAE_PATH):
        raise FileNotFoundError(f"Datos KoVAE no encontrados: {ERA5_KOVAE_PATH}")
    
    print(f"[LOAD] Cargando ERA5 KoVAE desde: {ERA5_KOVAE_PATH.relative_to(PROJECT_ROOT)}")
    with open(ERA5_KOVAE_PATH, 'rb') as f:
        data = pickle.load(f)
    
    precip_2020 = data['precip_2020']
    print(f"[OK] ERA5 KoVAE cargado: {precip_2020.shape}")
    print(f"  Rango: [{precip_2020.min():.4f}, {precip_2020.max():.4f}]")
    
    return data


def load_forecast_results() -> Dict:
    """
    Carga resultados de pronóstico AE+DMD.
    
    Returns:
        dict con 'y_test_real', 'forecast_results', métricas
    """
    if not verify_real_data(FORECAST_RESULTS_PATH):
        raise FileNotFoundError(f"Resultados no encontrados: {FORECAST_RESULTS_PATH}")
    
    print(f"[LOAD] Cargando forecast results desde: {FORECAST_RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    with open(FORECAST_RESULTS_PATH, 'rb') as f:
        results = pickle.load(f)
    
    print(f"[OK] Forecast results cargado")
    print(f"  Keys disponibles: {list(results.keys())}")
    
    if 'y_test_real' in results:
        print(f"  y_test_real shape: {results['y_test_real'].shape}")
    if 'forecast_results' in results:
        horizons = list(results['forecast_results'].keys())
        print(f"  Horizontes disponibles: {horizons}")
    
    return results


def split_temporal(
    data: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    División temporal de datos (NO aleatoria, preserva orden temporal).
    
    Args:
        data: (T, lat, lon, channels)
        train_ratio, val_ratio, test_ratio: Proporciones (deben sumar 1.0)
        
    Returns:
        X_train, X_val, X_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios deben sumar 1.0"
    
    n_samples = len(data)
    n_train = int(train_ratio * n_samples)
    n_val = int(val_ratio * n_samples)
    
    X_train = data[:n_train]
    X_val = data[n_train:n_train + n_val]
    X_test = data[n_train + n_val:]
    
    print(f"[SPLIT] División temporal:")
    print(f"  Train: {X_train.shape} ({len(X_train)} muestras)")
    print(f"  Val:   {X_val.shape} ({len(X_val)} muestras)")
    print(f"  Test:  {X_test.shape} ({len(X_test)} muestras)")
    
    return X_train, X_val, X_test


def get_data_info() -> Dict:
    """
    Retorna información de todos los archivos de datos disponibles.
    """
    info = {
        'era5_full': {
            'path': ERA5_FULL_PATH,
            'exists': ERA5_FULL_PATH.exists(),
            'size_mb': ERA5_FULL_PATH.stat().st_size / (1024**2) if ERA5_FULL_PATH.exists() else 0
        },
        'era5_kovae': {
            'path': ERA5_KOVAE_PATH,
            'exists': ERA5_KOVAE_PATH.exists(),
            'size_mb': ERA5_KOVAE_PATH.stat().st_size / (1024**2) if ERA5_KOVAE_PATH.exists() else 0
        },
        'forecast_results': {
            'path': FORECAST_RESULTS_PATH,
            'exists': FORECAST_RESULTS_PATH.exists(),
            'size_mb': FORECAST_RESULTS_PATH.stat().st_size / (1024**2) if FORECAST_RESULTS_PATH.exists() else 0
        }
    }
    
    print("[INFO] Archivos de datos reales:")
    for name, details in info.items():
        status = "✅" if details['exists'] else "❌"
        size_str = f"{details['size_mb']:.2f} MB" if details['exists'] else "N/A"
        print(f"  {status} {name}: {size_str}")
    
    return info


def validate_pipeline_data():
    """
    Valida que todos los archivos de datos reales estén presentes.
    Lanza error si detecta archivos sintéticos o faltantes críticos.
    """
    print("\n" + "="*70)
    print("VALIDACIÓN DE DATOS DEL PIPELINE")
    print("="*70)
    
    critical_files = [ERA5_FULL_PATH, ERA5_KOVAE_PATH]
    all_ok = True
    
    for file_path in critical_files:
        if verify_real_data(file_path):
            print(f"✅ {file_path.name}")
        else:
            print(f"❌ {file_path.name} - FALTANTE O SINTÉTICO")
            all_ok = False
    
    # Verificar que NO existan archivos sintéticos en uso
    synthetic_check = [
        RAW_DIR / 'precipitation_data.npy'
    ]
    
    for syn_file in synthetic_check:
        if syn_file.exists():
            print(f"⚠️  ADVERTENCIA: Archivo sintético detectado: {syn_file.name}")
            print(f"    Este archivo NO debe usarse en el pipeline principal")
    
    print("="*70)
    
    if not all_ok:
        raise RuntimeError("❌ Validación fallida: archivos de datos críticos faltantes")
    
    print("✅ Validación exitosa: todos los datos reales están presentes\n")
    return True


if __name__ == "__main__":
    # Test del módulo
    print("Probando data_loader...")
    try:
        validate_pipeline_data()
        get_data_info()
        print("\n✅ data_loader.py funcional")
    except Exception as e:
        print(f"\n❌ Error: {e}")
