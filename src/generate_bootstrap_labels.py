from pathlib import Path

import cv2

from anonymize import load_models

DATASET_DIR = Path("data/retrain_dataset")
IMAGES_TRAIN_DIR = DATASET_DIR / "images" / "train"
IMAGES_VAL_DIR = DATASET_DIR / "images" / "val"
LABELS_TRAIN_DIR = DATASET_DIR / "labels" / "train"
LABELS_VAL_DIR = DATASET_DIR / "labels" / "val"

# WICHTIG: Beide Splits nutzen das Bild MIT Gesichtern (test_crowd.jpg),
# damit die Validierung ueberhaupt etwas zu messen hat. Mit nur einem
# Gesichts-Bild insgesamt waere Training/Validierung auf demselben Bild
# in einem echten Projekt fragwuerdig (Overfitting-Gefahr) - hier ist das
# aber ohnehin nur eine Pipeline-Demonstration, keine echte Trainingslauf-
# Bewertung, siehe Hinweis in retrain.py.
SOURCE_IMAGES = [
    (Path("data/test_crowd.jpg"), "train"),
    (Path("data/test_crowd.jpg"), "val"),
]


def to_yolo_format(box: tuple, img_width: int, img_height: int) -> tuple:
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return x_center, y_center, width, height


def build_dataset():
    for split_dir in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
        split_dir.mkdir(parents=True, exist_ok=True)

    print("Lade aktuelles Modell, um Bootstrap-Labels zu erzeugen ...")
    models = load_models()
    face_model = models["face"]

    for source_path, split in SOURCE_IMAGES:
        image = cv2.imread(str(source_path))
        img_height, img_width = image.shape[:2]

        results = face_model.predict(image, conf=0.4, verbose=False)

        image_target_dir = IMAGES_TRAIN_DIR if split == "train" else IMAGES_VAL_DIR
        label_target_dir = LABELS_TRAIN_DIR if split == "train" else LABELS_VAL_DIR

        image_target_path = image_target_dir / source_path.name
        cv2.imwrite(str(image_target_path), image)

        label_lines = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x_c, y_c, w, h = to_yolo_format((x1, y1, x2, y2), img_width, img_height)
                label_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        label_target_path = label_target_dir / f"{source_path.stem}.txt"
        label_target_path.write_text("\n".join(label_lines))

        print(f"{source_path.name} ({split}): {len(label_lines)} Gesichter als Bootstrap-Label gespeichert")

    dataset_yaml = f"""path: {DATASET_DIR.resolve()}
train: images/train
val: images/val
names:
  0: face
"""
    (DATASET_DIR / "dataset.yaml").write_text(dataset_yaml)
    print(f"\ndataset.yaml gespeichert unter: {DATASET_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    build_dataset()