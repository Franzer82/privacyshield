import sys
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from anonymize import anonymize_image, load_models

# Globale Variable, die einmalig beim Start der API mit den geladenen
# Modellen befuellt wird - siehe lifespan() weiter unten.
_models = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPIs empfohlener Weg, Code beim Start/Stopp der Anwendung
    auszufuehren. Wir laden die (vergleichsweise schweren) YOLO-Modelle
    HIER genau EINMAL - nicht bei jeder einzelnen Anfrage, das waere viel
    zu langsam. Der Code nach 'yield' liefe beim Herunterfahren der API,
    hier gibt es aber nichts aufzuraeumen."""
    global _models
    print("Lade Modelle beim API-Start ...")
    _models = load_models()
    print("Modelle geladen, API bereit.")
    yield


app = FastAPI(
    title="PrivacyShield API",
    description="Automatisierte Anonymisierung von Gesichtern und Kennzeichen in Bildern (Gaussian-Blur auf YOLOv8-Erkennungen).",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Einfacher Gesundheitscheck-Endpunkt - Standard-Praxis bei APIs,
    damit z.B. Docker oder ein Load Balancer pruefen kann, ob der Dienst
    laeuft UND einsatzbereit ist (Modelle tatsaechlich geladen)."""
    return {"status": "ok", "models_loaded": _models is not None}


@app.post("/anonymize")
async def anonymize_endpoint(file: UploadFile = File(...)):
    """Nimmt ein hochgeladenes Bild entgegen, fuehrt Gesichts-/Kennzeichen-
    erkennung + Verpixelung aus, und gibt das Ergebnisbild direkt als
    JPEG zurueck. Die Anzahl gefundener Objekte wird zusaetzlich als
    HTTP-Header mitgeliefert, damit Aufrufer diese Info bekommen, ohne
    den Bildinhalt selbst analysieren zu muessen."""
    if _models is None:
        raise HTTPException(status_code=503, detail="Modelle sind noch nicht geladen")

    contents = await file.read()
    np_array = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Ungültiges oder nicht lesbares Bildformat")

    result = anonymize_image(image, _models)

    success, encoded_image = cv2.imencode(".jpg", result["output_image"])
    if not success:
        raise HTTPException(status_code=500, detail="Fehler beim Kodieren des Ergebnisbildes")

    headers = {
        "X-Faces-Detected": str(result["num_faces"]),
        "X-Plates-Detected": str(result["num_plates"]),
    }

    return Response(content=encoded_image.tobytes(), media_type="image/jpeg", headers=headers)