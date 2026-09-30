FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces verlangt, dass Container NICHT als root laufen,
# sondern als Nutzer mit UID 1000 - eine gute Sicherheitspraxis, die wir
# hier befolgen. "useradd -m" legt dabei automatisch ein Home-Verzeichnis
# an, das wir gleich fuer Modell-/Bibliothek-Caches brauchen.
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .

# PyTorch/torchvision GEZIELT ueber den offiziellen CPU-only-Paketindex
# installieren (statt aus requirements.txt via normalem PyPI) - das
# vermeidet die automatische Installation mehrerer Gigabyte an NVIDIA-
# GPU-Bibliotheken, die wir fuer reine CPU-Inferenz nicht brauchen.
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

# Besitzrechte an den neu kopierten Dateien auf unseren Nicht-Root-Nutzer
# uebertragen, bevor wir zu ihm wechseln - sonst haette "appuser" keinen
# Lese-/Schreibzugriff auf die Anwendungsdateien.
RUN chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser

# Port 7860 ist bei Hugging Face Spaces (Docker-SDK) FEST VORGESCHRIEBEN -
# ein anderer Port wuerde von der Plattform nicht erkannt werden.
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]