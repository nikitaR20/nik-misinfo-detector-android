# Nik-Misinfo-Detector-Android

A mobile-first misinformation detection system with:
- **Frontend**: Android app in Kotlin
- **Backend**: FastAPI serving DistilBERT model
- **ML Model**: Fine-tuning scripts, datasets, and saved weights
- **Docs**: Project management and design documentation

## Structure
- `frontend/` → Android app code
- `backend/` → FastAPI backend with DistilBERT API
- `ml_model/` → AI/ML experiments, fine-tuning scripts
- `docs/` → Project documents
- `.github/` → CI/CD workflows

## Getting Started
1. Clone repo
2. Setup backend:
   ```bash
   cd backend
   pip install -r requirements.txt
   uvicorn app.main:app --reload
