import cv2
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Modelle liegen auf dem Hugging Face Hub statt lokal - wichtig, damit
# JEDER (Tester, CI/CD-Pipeline, spaeteres Deployment) die Modelle
# automatisch bekommt, ohne dass sie manuell heruntergeladen und lokal
# abgelegt werden muessen. Der models/-Ordner ist bewusst NICHT Teil des
# Git-Repositories (zu gross), die Datei wird beim ersten Start
# automatisch heruntergeladen und danach lokal zwischengespeichert.
HF_REPO_ID = "Franzer82/privacyshield-models"
FACE_MODEL_FILENAME = "yolov8n-face.pt"
PLATE_MODEL_FILENAME = "license-plate-best.pt"

CONFIDENCE_THRESHOLD = 0.4
BLUR_STRENGTH = 25


def load_models():
    """Lädt beide vortrainierten YOLOv8-Modelle - bei Bedarf automatisch
    von Hugging Face heruntergeladen (nur beim allerersten Start; danach
    wird die lokal zwischengespeicherte Version genutzt)."""
    face_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=FACE_MODEL_FILENAME)
    plate_model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=PLATE_MODEL_FILENAME)

    face_model = YOLO(face_model_path)
    plate_model = YOLO(plate_model_path)
    return {"face": face_model, "plate": plate_model}


def detect_regions(image: np.ndarray, models: dict) -> list:
    """Führt beide Modelle auf einem BELIEBIGEN Bild aus (egal ob eigenes
    Testbild oder von einem Tester/einer Testerin hochgeladenes Foto) und
    sammelt alle gefundenen Bounding Boxes (Gesichter + Kennzeichen)."""
    regions = []

    for label, model in models.items():
        results = model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                regions.append({
                    "label": label,
                    "confidence": confidence,
                    "box": (x1, y1, x2, y2),
                })

    return regions


def blur_regions(image: np.ndarray, regions: list) -> np.ndarray:
    """Verpixelt jede erkannte Region im Bild. Arbeitet auf einer KOPIE
    des Bildes, damit das Original unverändert bleibt."""
    output_image = image.copy()

    for region in regions:
        x1, y1, x2, y2 = region["box"]

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = output_image[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (0, 0), sigmaX=BLUR_STRENGTH)
        output_image[y1:y2, x1:x2] = blurred_roi

    return output_image


def anonymize_image(image: np.ndarray, models: dict) -> dict:
    """Kompletter Ablauf: Erkennen + Verpixeln in einem Schritt."""
    regions = detect_regions(image, models)
    output_image = blur_regions(image, regions)

    return {
        "output_image": output_image,
        "num_faces": sum(1 for r in regions if r["label"] == "face"),
        "num_plates": sum(1 for r in regions if r["label"] == "plate"),
        "regions": regions,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Nutzung: python3 src/anonymize.py <bildpfad>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    if image is None:
        print(f"Konnte Bild nicht laden: {image_path}")
        sys.exit(1)

    print("Lade Modelle ...")
    models = load_models()

    print("Erkenne und verpixele ...")
    result = anonymize_image(image, models)

    print(f"Gefundene Gesichter: {result['num_faces']}")
    print(f"Gefundene Kennzeichen: {result['num_plates']}")

    output_path = "data/test_output.jpg"
    Path("data").mkdir(exist_ok=True)
    cv2.imwrite(output_path, result["output_image"])
    print(f"Ergebnis gespeichert unter: {output_path}")