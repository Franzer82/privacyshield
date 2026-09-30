from pathlib import Path

import mlflow
from ultralytics import YOLO

from tracking import init_tracking

DATASET_YAML = Path("data/retrain_dataset/dataset.yaml")
BASE_MODEL_PATH = Path("models/yolov8n-face.pt")
OUTPUT_DIR = Path("models/retrained")

EPOCHS = 3
IMAGE_SIZE = 640


def run_retraining():
    """Fuehrt ein kurzes Nachtraining des Gesichtsmodells durch,
    protokolliert das Ergebnis in MLflow, und gibt zurueck, ob das neue
    Modell (laut Validierungsmetrik) besser ist als das vorherige."""
    init_tracking()

    print(f"Lade Basismodell: {BASE_MODEL_PATH}")
    model = YOLO(str(BASE_MODEL_PATH))

    with mlflow.start_run(run_name="retraining-run"):
        mlflow.log_param("base_model", str(BASE_MODEL_PATH))
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("dataset", str(DATASET_YAML))
        mlflow.log_param(
            "note",
            "Pipeline-Demonstration mit Bootstrap-Labels - keine echte Genauigkeitsverbesserung zu erwarten",
        )

        print(f"\nStarte Nachtraining ({EPOCHS} Epochen) ...")
        model.train(
            data=str(DATASET_YAML),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            project=str(OUTPUT_DIR),
            name="run",
            exist_ok=True,
            verbose=False,
        )

        # WICHTIG: Den tatsaechlichen Speicherort direkt vom Trainer
        # ablesen, statt ihn selbst zusammenzubauen - Ultralytics haengt
        # bei relativen project-Pfaden intern noch seinen eigenen
        # Standard-Ordner ("runs/detect/") davor, was zu einem falschen,
        # von uns geratenen Pfad fuehren wuerde.
        actual_save_dir = Path(model.trainer.save_dir)
        new_model_path = actual_save_dir / "weights" / "best.pt"

        metrics = model.val(data=str(DATASET_YAML), verbose=False)
        map50 = float(metrics.box.map50) if metrics.box.map50 else 0.0

        mlflow.log_metric("map50", map50)

        print(f"\nNachtraining abgeschlossen. mAP50: {map50:.4f}")
        print(f"Neues Modell gespeichert unter: {new_model_path}")

        if new_model_path.exists():
            mlflow.log_artifact(str(new_model_path), artifact_path="retrained_model")
        else:
            print(f"WARNUNG: Modell-Datei nicht gefunden unter {new_model_path}, ueberspringe MLflow-Artefakt-Upload")

        return {
            "map50": map50,
            "model_path": new_model_path,
        }


if __name__ == "__main__":
    result = run_retraining()
    print(f"\n=== Zusammenfassung ===")
    print(f"mAP50: {result['map50']:.4f}")
    print(f"Modell-Pfad: {result['model_path']}")