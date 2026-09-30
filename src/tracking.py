import time
from pathlib import Path

import mlflow

# SQLite statt des veralteten dateibasierten Trackings ("./mlruns") -
# neuere MLflow-Versionen verlangen einen Datenbank-Backend. SQLite ist
# eine einfache, dateibasierte Datenbank ohne zusaetzlichen Server-Betrieb
# noetig - ideal fuer ein einzelnes Portfolio-Projekt.
MLFLOW_DB_PATH = Path("mlflow.db")
EXPERIMENT_NAME = "privacyshield-inference"

_tracking_initialized = False


def init_tracking():
    """Richtet MLflow einmalig ein und legt (falls noch nicht vorhanden)
    unser Experiment an. Ein 'Experiment' in MLflow ist ein Container fuer
    zusammengehoerige Laeufe - bei uns: alle Inferenz-Anfragen der API."""
    global _tracking_initialized
    if _tracking_initialized:
        return

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH.resolve()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    _tracking_initialized = True


def log_inference(num_faces: int, num_plates: int, processing_time_seconds: float, model_version: str = "yolov8n-v1"):
    """Protokolliert eine einzelne Anfrage als MLflow-'Run'. In der Praxis
    wuerde man nicht jede einzelne Anfrage einzeln loggen (zu viel
    Overhead), sondern z.B. stuendlich aggregierte Werte - fuer unser
    Portfolio-Beispiel macht die Pro-Anfrage-Protokollierung die Funktions-
    weise aber besser nachvollziehbar und sichtbar."""
    init_tracking()

    with mlflow.start_run():
        mlflow.log_param("model_version", model_version)
        mlflow.log_metric("num_faces_detected", num_faces)
        mlflow.log_metric("num_plates_detected", num_plates)
        mlflow.log_metric("processing_time_seconds", processing_time_seconds)
        mlflow.log_metric("total_detections", num_faces + num_plates)


class InferenceTimer:
    """Kleiner Hilfs-Kontextmanager, um die Verarbeitungszeit einer
    Anfrage sauber zu messen, ohne den Aufrufer-Code mit manueller
    Zeitmessung zu ueberladen."""

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_seconds = time.perf_counter() - self.start_time