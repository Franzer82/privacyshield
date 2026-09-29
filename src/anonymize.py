from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

FACE_MODEL_PATH = Path("models/yolov8n-face.pt")
PLATE_MODEL_PATH = Path("models/best.pt")

# Ab welcher Erkennungssicherheit (0-1) ein Treffer als "echt" gilt,
# statt als unsicheres Rauschen ignoriert zu werden.
CONFIDENCE_THRESHOLD = 0.4

# Wie stark die Verpixelung ist (groesserer Wert = staerker verpixelt).
BLUR_STRENGTH = 25


def load_models():
    """Lädt beide vortrainierten YOLOv8-Modelle einmalig."""
    face_model = YOLO(str(FACE_MODEL_PATH))
    plate_model = YOLO(str(PLATE_MODEL_PATH))
    return {"face": face_model, "plate": plate_model}


def detect_regions(image: np.ndarray, models: dict) -> list:
    """Führt beide Modelle auf einem Bild aus und sammelt alle gefundenen
    Bounding Boxes (Gesichter + Kennzeichen) in einer gemeinsamen Liste."""
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
    des Bildes, damit das Original unverändert bleibt (wichtig, falls die
    aufrufende Funktion beide Versionen noch braucht)."""
    output_image = image.copy()

    for region in regions:
        x1, y1, x2, y2 = region["box"]

        # Sicherstellen, dass die Koordinaten innerhalb des Bildes liegen
        # (Modelle können knapp am Bildrand leicht ueberschiessende Boxen
        # liefern)
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