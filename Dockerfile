# Offizielles, schlankes Python-Basisimage - "slim" enthaelt nur das
# Noetigste, haelt das finale Image kleiner als das volle Python-Image.
FROM python:3.13-slim

# Systemabhaengigkeiten, die opencv-python UND ultralytics (YOLO) intern
# benoetigen, aber nicht automatisch mitbringen - ohne diese wuerde der
# Import von cv2/ultralytics im Container fehlschlagen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Erst NUR requirements.txt kopieren und installieren, bevor der Rest des
# Codes reinkommt - Docker cached jeden Schritt einzeln. Aendert sich nur
# der Code (nicht die Abhaengigkeiten), muss der langsame pip-install-
# Schritt beim naechsten Bauen NICHT wiederholt werden.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Jetzt den restlichen Projektcode kopieren
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/

# Informativ: auf welchem Port die API im Container lauscht
EXPOSE 8000

# Startbefehl: WICHTIG - "0.0.0.0" statt "127.0.0.1", damit die API auch
# von AUSSERHALB des Containers erreichbar ist (127.0.0.1 waere nur von
# innerhalb des Containers selbst erreichbar - klassischer Docker-Stolperstein).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]