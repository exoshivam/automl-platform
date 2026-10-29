# AutoML Platform

A no-code machine-learning platform that runs entirely on your own computer. Create a project, add data (upload, camera or a starter dataset), label it with AI help, train with one click, evaluate the model, test it in a playground, then export it to ONNX or deploy it behind an API key.

Supported project types: image classification, object detection (custom, ready-made COCO and hybrid), text classification, quick text and sentiment, tabular classification and regression, face recognition and speaker recognition.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React, TypeScript, Vite, TanStack Router and Query, Tailwind CSS |
| Backend | Python 3.12, FastAPI, SQLAlchemy, Alembic, Celery |
| Data | PostgreSQL 16 + pgvector, Redis, MinIO (S3-compatible), MLflow |
| ML | PyTorch, timm, torchvision, Hugging Face Transformers, LightGBM, Optuna, ONNX Runtime |

## Prerequisites

- **Docker Desktop** (runs PostgreSQL, Redis, MinIO and MLflow)
- **Python 3.12** and **[uv](https://docs.astral.sh/uv/)**
- **Node.js 20+** and npm
- macOS only: `brew install libomp` (needed by LightGBM)
- About 8 GB of free RAM and 5 GB of disk for the pretrained models

## Ports

Everything listens on `127.0.0.1` only.

| Service | Port |
|---|---|
| Frontend | 5180 |
| API | 8010 |
| PostgreSQL | 5442 |
| Redis | 6381 |
| MinIO API / console | 9010 / 9011 |
| MLflow | 5005 |

## Run locally

All commands start from the repository root.

### 1. Start the infrastructure

```sh
docker compose -f docker-compose.dev.yml up -d
docker compose -f docker-compose.dev.yml ps      # postgres, redis and minio should be (healthy)
```

### 2. Set up the backend

```sh
cd backend
cp .env.example .env                 # defaults already match docker-compose.dev.yml
uv sync                              # creates .venv with Python 3.12
uv run alembic upgrade head          # create the database schema
uv run python -m app.scripts.seed_dev   # demo user: demo@automl.dev / demo1234
```

### 3. Download the pretrained models (once)

The app loads models from the local Hugging Face cache and never downloads them during a request, so fetch them once:

```sh
uv run python - <<'EOF'
from huggingface_hub import snapshot_download
for repo, rev in [
    ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", None),
    ("tabularisai/multilingual-sentiment-analysis", "eea032081f8d247b4303ef3565e7cec1b6f201c9"),
    ("microsoft/wavlm-base-plus-sv", "feb593a6c23c1cc3d9510425c29b0a14d2b07b1e"),
    ("google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d"),
]:
    snapshot_download(repo, revision=rev)
    print("ok", repo)
EOF
```

Image-classification (timm) and detection (torchvision SSDlite) weights download automatically the first time you train.

Face recognition needs the InsightFace `buffalo_l` pack in `.model-cache/insightface/buffalo_l/` (the files `det_10g.onnx` and `w600k_r50.onnx`). Download `buffalo_l.zip` from the [InsightFace releases](https://github.com/deepinsight/insightface/releases) and unzip it there. For the optional small export pack, put `buffalo_s` next to it. These models are licensed for **non-commercial research only**.

### 4. Start the API and the workers

Use three terminals, all inside `backend/`:

```sh
# API on http://127.0.0.1:8010 (reloads on code changes)
APP_ENV=dev LLM_PROVIDER=stub uv run uvicorn app.main:app --host 127.0.0.1 --port 8010 --reload --reload-dir app
```

```sh
# default queue: ingest, validation, evaluation, export, deletion
APP_ENV=dev LLM_PROVIDER=stub uv run celery -A app.worker worker -P threads -c 4 -Q default -n default@%h --loglevel INFO
```

```sh
# train queue: one heavy job at a time (training, auto-labelling)
APP_ENV=dev LLM_PROVIDER=stub TRAIN_BATCH_SIZE=8 uv run celery -A app.worker worker -P threads -c 1 -Q train -n train@%h --loglevel INFO
```

Celery workers do not reload automatically. Restart them after changing task code.

### 5. Start the frontend

```sh
cd frontend
npm install
npm run dev -- --port 5180 --host 127.0.0.1
```

Open **http://127.0.0.1:5180** and log in with `demo@automl.dev` / `demo1234`.

### 6. Check that everything is up

```sh
curl -s http://127.0.0.1:8010/api/v1/health      # expect "database":"ok" and "redis":"ok"
cd backend && APP_ENV=dev uv run celery -A app.worker inspect ping   # expect 2 nodes online
```

## Optional data

- **Starter datasets** (Beans, Flowers and others): place the raw downloads in `.starter-cache/` and run `uv run python scripts/build_starters.py` from `backend/`.
- **Hybrid detection** (ready-made objects plus your own): needs a COCO replay set. Put the COCO val2017 subset in `.starter-cache/coco_val2017_subset/` and run `uv run python scripts/build_replay.py`.

## Tests

```sh
cd backend && uv run pytest          # backend (needs the Docker services running)
cd frontend && npm run typecheck && npm test
```

## Stop everything

Stop the API, workers and frontend with `Ctrl+C`, then:

```sh
docker compose -f docker-compose.dev.yml stop
```

Do not use `down -v`: it deletes the data volumes.

## Configuration notes

- `LLM_PROVIDER=stub` uses keyword rules to suggest a task setup from a plain-English description. To use a real language model, set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY` in `backend/.env`.
- On a 16 GB machine, keep `TRAIN_BATCH_SIZE=8` and the train worker at `-c 1`.
- Deployed models are served at `POST http://127.0.0.1:8010/v1/predict` with an API key created in the project's **Deploy** tab.

## Licences of pretrained models

| Model | Licence |
|---|---|
| MobileNetV3 / EfficientNet (timm), MiniLM, OWLv2 | Apache-2.0 |
| SSDlite320-MobileNetV3 (torchvision) | BSD-3-Clause |
| WavLM base-plus-sv | CC BY-SA 3.0 |
| tabularisai multilingual sentiment | CC BY-NC 4.0 (non-commercial) |
| InsightFace buffalo_l / buffalo_s | Non-commercial research only |

The product requirements are in [PRD.md](PRD.md).
