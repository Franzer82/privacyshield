---
title: PrivacyShield
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# PrivacyShield

🔗 **Live-Demo**: [privacyshield-0h0m.onrender.com/docs](https://privacyshield-0h0m.onrender.com/docs)

Automatisierte Anonymisierung von Gesichtern und Kfz-Kennzeichen in Bildern.

## Funktionsweise

Zwei YOLOv8-Modelle (Gesichtserkennung + Kennzeichenerkennung) identifizieren
sensible Bildbereiche, die anschließend automatisch verpixelt werden.

## API nutzen

Öffne `/docs` für die interaktive Swagger-Oberfläche, oder sende eine
POST-Anfrage an `/anonymize` mit einem Bild als multipart/form-data.

## Technischer Hintergrund

- **Erkennung**: YOLOv8n (Gesichter, MIT-lizenziert von lindevs) +
  YOLOv8n (Kennzeichen, MIT-lizenziert von Koushim)
- **API**: FastAPI
- **Containerisierung**: Docker
- **Experiment-Tracking**: MLflow
- **CI/CD**: GitHub Actions (automatisches Testen, Docker-Build,
  auslösbare Retraining-Pipeline)
- **Deployment**: Render.com

Vollständiger Quellcode und Modelle: alle Links unten.