import mlflow
import os

def set_tracking(local_dir: str = "mlruns"):
 # Usaremos almacenamiento local por simplicidad
 uri = f"file:{os.path.abspath(local_dir)}"
 mlflow.set_tracking_uri(uri)
 return uri
