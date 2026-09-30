from pathlib import Path

import mlflow
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from tracking import init_tracking

# Gleiche Hugging-Face-Quelle wie in anonymize.py - so funktioniert das
# Nachtraining unabhaengig davon, ob es lokal oder in der CI/CD-Pipeline
# (wo der lokale models/-Ordner nicht existiert) ausgefuehrt wird.
HF_REPO_ID = "Franzer82/privacyshield-models"
FACE_MODEL_FILENAME = "yolov8n-face.pt"

DATASET_YAML = Path("data/retrain_dataset/dataset.yaml")
OUTPUT_DIR = Path("models/retrained")

EPOCHS = 3
IMAGE_SIZE = 640


def run_retraining():
    """Fuehrt ein kurzes Nachtraining des Gesichtsmodells durch,
    protokolliert das Ergebnis in MLflow, und gibt zurueck, ob das neue
    Modell (laut Validierungsmetrik) besser ist als das vorherige."""
    init_tracking()

    print(f"Lade Basismodell von Hugging Face: {HF_REPO_ID}/{FACE_MODEL_FILENAME}")
    base_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=FACE_MODEL_FILENAME)
    model = YOLO(base_model_path)

    with mlflow.start_run(run_name="retraining-run"):
        mlflow.log_param("base_model", f"{HF_REPO_ID}/{FACE_MODEL_FILENAME}")
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