# AutoML Platform: Product Requirements Document

| Field | Value |
|---|---|
| Product | AutoML Platform (working name) |
| Document | Product Requirements Document (PRD) |
| Version | 0.3 (Draft for engineering review) |
| Date | 22 October 2026 |
| Owner | Shivam Yadav |
| Status | Draft; open questions in §15.2 |

**Changes in v0.3 (current build target):** the first build is a **local, single-machine MVP** that the owner can run and test end to end. The flow is:

1. create a project
2. upload or camera-capture data
3. label it (including the box tool)
4. validate the data
5. train (Quick budget, on CPU or Apple MPS)
6. evaluate, and try the model in the playground
7. download the export (ONNX first)

Moved to **Phase 2**:
- hosted deployment, i.e. users deploying models on the platform itself (§6.9 DEP-*)
- billing and quotas (§6.11)
- Google login
- real email sending
- team members and invites
- production infrastructure and hardening

The production requirements below stay as the long-term target.

**Changes in v0.2:**
- Users start by choosing a **category** (Image, Text/Sentiment, Audio, Object detection, Tabular). The plain-English description becomes the secondary path.
- **Object detection moves into the MVP**, with the bounding-box tool and AI-assisted box suggestions.
- **Camera capture** (burst mode, rear camera on phones) is P0.
- The MVP grows from ~3 to **~4 months**.

---

## Table of contents

1. [Overview](#1-overview)
2. [Goals, non-goals and success metrics](#2-goals-non-goals-and-success-metrics)
3. [Target users and personas](#3-target-users-and-personas)
4. [What the platform produces](#4-what-the-platform-produces)
5. [Core user flow](#5-core-user-flow)
6. [Functional requirements](#6-functional-requirements)
7. [Non-functional requirements](#7-non-functional-requirements)
8. [Tech stack decisions](#8-tech-stack-decisions)
9. [System architecture](#9-system-architecture)
10. [Data model](#10-data-model)
11. [REST API overview](#11-rest-api-overview)
12. [Responsible AI and face recognition policy](#12-responsible-ai-and-face-recognition-policy)
13. [Pricing tiers](#13-pricing-tiers)
14. [Roadmap](#14-roadmap)
15. [Risks, mitigations and open questions](#15-risks-mitigations-and-open-questions)
- [Appendix A: Task spec JSON schema (draft)](#appendix-a-task-spec-json-schema-draft)
- [Appendix B: Glossary](#appendix-b-glossary)

---

## 1. Overview

### 1.1 Vision
Anyone with data and a problem should be able to get a production-grade machine-learning model in under an hour, without writing training code, and deploy it wherever their product runs.

### 1.2 Problem
- **Building a custom model is still slow and expert-heavy.** It needs data cleaning, model selection, hyperparameter tuning, evaluation, and export to the target runtime, and each step has its own tooling and failure modes.
- **Existing no-code tools stop at demos.** Teachable Machine-style tools are great for learning, but they offer few task types, no data validation, weak evaluation, no versioning or reproducibility, and no path to a hosted API with keys, quotas and monitoring.
- **Cloud AutoML suites are expensive, opaque and cloud-locked.** They are hard to use for small teams, students and Indian SMBs, and exports to mobile or browser are often limited.
- **Small datasets defeat from-scratch training.** Most real users have hundreds, not millions, of examples.

### 1.3 Solution
A platform where users upload their own data, describe the task (from a template or in plain English), and receive a trained, tested, documented, ready-to-deploy model.

Five things make it work:
1. **Never train from scratch.** Every model fine-tunes a strong pretrained backbone (timm image models, Hugging Face transformers, audio encoders) or uses gradient boosting for tabular data. Training is fast and cheap, and accurate on small data.
2. **Automated search.** Hyperparameter search (Optuna; Ray Tune at scale), with early stopping and pruning, runs within a user-chosen budget.
3. **Guard rails before training.** Data validation catches corrupt files, leakage, imbalance and duplicates, and suggests fixes before any GPU time is spent.
4. **Honest evaluation.** A held-out test set, task-appropriate metrics, per-class results, worst mistakes, and concrete advice on improving the model.
5. **Deploy anywhere.** Export to ONNX, TFLite, Core ML and TF.js with ready-to-paste code, or deploy as a hosted inference API with keys and usage metering.

### 1.4 One-line pitch
> Upload your data, say what you want, and get a production-ready custom AI model in under 30 minutes, for your app, your website or your API.

---

## 2. Goals, non-goals and success metrics

### 2.1 Goals
| ID | Goal |
|---|---|
| G1 | A non-ML user gets a usable, evaluated model from raw data in under 30 minutes (Quick budget). |
| G2 | Model quality comes within 3% of a hand-tuned baseline built by an ML engineer on the same data. |
| G3 | Every model is reproducible: same dataset version, config, seeds and library versions give the same result within tolerance. |
| G4 | Every model can be exported to at least one on-device format and deployed as a hosted API in one click. |
| G5 | GPU cost per job stays low enough for a sustainable free tier and healthy paid margins. |
| G6 | Face recognition is offered safely: consent capture, licence-clean models and biometric-grade data protection. |

### 2.2 Non-goals (v1)
- Training foundation models or any model from scratch.
- Generative image, video or audio models (diffusion, GANs, TTS voice cloning).
- On-premise or air-gapped installation. Cloud SaaS only in v1; on-prem is an Enterprise discussion for later.
- Full MLOps for externally trained models (bring-your-own-model hosting). Possibly later.
- Real-time video stream processing as a hosted service (single-frame and batch inference only in v1).
- General-purpose data labeling marketplace or human labeling workforce.
- Training on users' data for any purpose other than their own models (no pooled training).

### 2.3 Success metrics
| Metric | Definition | Target (6 months after GA) |
|---|---|---|
| Time to first model | Median time from project creation to first evaluated model (Quick budget, Phase 1 tasks) | **< 30 min** (stretch: < 15 min) |
| Job success rate | Training jobs that end in an evaluated model ÷ jobs started (excluding user cancellations) | **> 97%** |
| Model quality | Median gap versus a hand-tuned baseline on an internal benchmark suite (§14.2), using the primary metric | **≤ 3%** absolute |
| Validation catch rate | Benchmark datasets with seeded defects (leakage, duplicates, corrupt files) where validation flags the defect | **> 95%** |
| Export/deploy rate | Trained models that are exported or deployed at least once | **> 40%** |
| GPU cost per job | Median GPU spend per completed job, by budget mode | Quick **< ₹15 (~$0.18)**; Standard **< ₹80 (~$1)**; Thorough **< ₹600 (~$7)** |
| Free-to-paid conversion | Free accounts that upgrade within 60 days | **≥ 3%** (stretch 5%) |
| Activation | New sign-ups that complete at least one training job within 7 days | **≥ 35%** |
| Hosted API availability | Monthly uptime of the inference endpoints | **≥ 99.9%** (Pro); **≥ 99.95%** (Enterprise SLA) |
| Support load | Support tickets per 100 training jobs | **< 2** |

---

## 3. Target users and personas

| Persona | Who | Goals | Pain points today | What they need from us |
|---|---|---|---|---|
| **P1: App developer** | Mobile or web developer at a startup or agency | Add "smart" features (classify photos, detect objects, sentiment) to an app | Doesn't know PyTorch; exporting to Core ML or TFLite is painful | Clean exports, copy-paste snippets, small fast models, a hosted API fallback |
| **P2: SMB / ops team** | Quality inspector, store manager, support lead at a small business or factory | Automate a repetitive judgement (defect yes/no, ticket routing, attendance) | No ML staff; vendor solutions are expensive and generic | Templates, plain-English setup, guidance on data, a reliable hosted API, simple pricing (INR) |
| **P3: Data analyst** | Analyst comfortable with Excel/SQL, maybe some Python | Predict churn, fraud or demand from tabular data | Hand-tuning models is slow; unsure which model or metric | CSV/Excel/Parquet ingestion, leakage checks, feature importance, a batch prediction API |
| **P4: Student / educator** | College student, bootcamp learner, teacher | Learn ML concepts with real results; class projects | Teachable Machine is too limited; Colab setups break | A generous free tier, explanations, visible experiments, Hinglish/Indian-language support |
| **P5: Startup ML team** | 1–3 ML engineers with too much to do | Fast strong baselines, reproducible experiments, deployment without building infrastructure | Rebuilding the same pipelines; GPU orchestration | MLflow access, configurable search, artefact downloads, API automation, team workspaces |

**Primary personas for MVP:** P1, P2 and P3. P4 drives top-of-funnel growth. P5 drives Team-tier revenue in Phase 2.

---

## 4. What the platform produces

### 4.1 Task templates by data type and phase

| Data type | Task template | Phase | Example use cases | Default backbone families (licence-checked, see §12.4) |
|---|---|---|---|---|
| Image | Image classification (single- and multi-label) | **1** | Manufacturing defect detection; plant disease identification; product categorisation; X-ray triage (research use); recyclable sorting | timm (ConvNeXt, EfficientNet, ViT, MobileNetV3/V4 for edge) |
| Text | Text classification | **1** | Support ticket routing; spam/abuse filtering; intent detection; topic tagging; Hinglish complaint categorisation | Hugging Face encoders (DeBERTa-v3, XLM-R, MuRIL, IndicBERT; DistilBERT/MiniLM for edge) |
| Text | Sentiment analysis (ready-made and custom) | **1** | Product review sentiment; social media monitoring; NPS comment analysis; custom labels such as *angry / confused / happy / churn-risk* | Same as text classification; a pretrained multilingual sentiment model for the ready-made option |
| Tabular | Tabular classification | **1** | Customer churn; loan default; lead scoring; fraud flagging | LightGBM, XGBoost, CatBoost (+ linear baseline) |
| Tabular | Tabular regression | **1** | House price; delivery ETA; demand per store; insurance claim size | LightGBM, XGBoost, CatBoost (+ linear baseline) |
| Image | Object detection | **1** | Safety gear (helmet/vest) detection; shelf stock counting; vehicle counting; PCB component detection | RT-DETR / DETR-family (Apache-2.0), YOLOX, torchvision detectors. *Avoid AGPL-licensed YOLO weights unless commercially licensed.* |
| Image | Face recognition (special template, §12) | 2 | Attendance with consent; employee access check-in; photo library tagging | Face detector + face-embedding model + gallery matching (not classification) |
| Audio | Audio / sound classification | 2 | Machine fault sounds; baby cry detection; wildlife calls; cough classification (research) | Audio Spectrogram Transformer, BEATs-style encoders, YAMNet-style models for edge |
| Audio | Keyword spotting | 2 | Custom wake word; voice command for kiosks; Hindi/English command words | Small CNN/transformer on log-mel features, fine-tuned from speech-commands pretraining |
| Text | Named entity extraction | 2 | Invoice fields from text; PII detection; medical entity tagging; address parsing (Indian addresses) | Token-classification transformers (DeBERTa, XLM-R, MuRIL) |
| Text | Aspect-based sentiment | 2 | "Battery good, camera bad" from reviews; restaurant review aspects (food/service/price) | Encoder fine-tuned for aspect extraction + polarity |
| Time series | Forecasting | 2 | Sales/demand forecasting; energy load; footfall; inventory planning | Gradient boosting with lag features; pretrained forecasting models (Chronos-style); statistical baselines |
| Tabular / time series | Anomaly detection | 2 | Transaction anomalies; sensor drift; server metric anomalies | Isolation Forest, ECOD/COPOD, autoencoders; forecast-residual methods |
| Image | Semantic / instance segmentation | 3 | Crop/weed segmentation; road damage area; medical region outlining (research); background removal | Mask2Former, SAM-assisted labeling + fine-tuned segmenters (licence-checked) |
| Image | Pose / keypoint detection | 3 | Exercise form checking; ergonomics monitoring; sports analytics | ViTPose / RTMPose-family |
| Document | OCR / document extraction | 3 | Invoice and receipt fields; KYC forms (with consent); handwritten forms | PaddleOCR / docTR + layout-aware extractors (avoid non-commercial weights) |
| Text | Small LLM fine-tuning (LoRA) | 3 | Domain-specific support replies; structured extraction; style-specific summarisation | 0.5B–8B open-weight models with permissive licences, PEFT/LoRA, QLoRA |
| Video | Action recognition | 3 | Shoplifting/fall detection (with policy review); workout rep counting; factory step verification | Video transformers / X-CLIP-style models |
| Sensor | IMU / activity recognition | 3 | Fall detection in wearables; gesture recognition; machine vibration states | 1D CNN / transformer on windowed signals; pretrained HAR encoders where licences allow |

**Indian languages and Hinglish.** All text tasks support Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi and Odia in native scripts, plus romanised and code-mixed Hinglish.
- Language detection runs at ingestion, and the dataset report shows the script and language mix.
- Model selection prefers multilingual or Indic encoders (MuRIL, IndicBERT, XLM-R) when non-English or code-mixed text is above 10% of the samples.
- An optional transliteration normalisation step (romanised to native script, or the reverse) is offered for Hinglish datasets, and the system keeps whichever variant scores better on validation.
- The ready-made sentiment model must be evaluated on an internal Hinglish and Indic benchmark before launch (§14.2).

### 4.2 Standard deliverables per model
Every completed training job produces:

| Deliverable | Contents |
|---|---|
| **Model artefacts** | The best checkpoint in its native format plus the requested exports (ONNX, TFLite, Core ML, TF.js), with optional quantised variants (§6.9) |
| **Labels and metadata file** | `model.json`: task type, class labels and order, input shape and preprocessing (resize, normalisation, tokenizer), output schema, thresholds, model version, dataset version hash, licence notices of upstream weights |
| **Model card** | Auto-generated Markdown/PDF: intended use, training data summary, metrics, per-class results, known limitations, fairness notes (mandatory for face models), licence of base model |
| **Evaluation report** | Test-set metrics, confusion matrix, PR/ROC curves where relevant, per-class table, worst mistakes gallery, calibration, latency and size per export |
| **Improvement recommendations** | Ranked, actionable suggestions (e.g., "Class *rust* has 38 images and 0.61 recall; add about 100 more, especially close-ups") |
| **Reproducibility manifest** | Dataset version ID, split seed, task spec, search space, best trial config, random seeds, Docker image digest, library versions (torch, timm, transformers, lightgbm, onnx…) |
| **Code snippets** | Copy-paste inference code per target: Python (ONNX Runtime), Android (Kotlin + TFLite), iOS (Swift + Core ML), Web (TF.js / onnxruntime-web), REST (curl, JS, Python) |
| **Experiment record** | Link to the MLflow run with all trials, parameters, metrics and artefacts (visible to Pro/Team users) |

---

## 5. Core user flow

1. **Sign up and create a workspace.** Sign in with email or Google. A personal workspace is created automatically; Team users can invite members.
2. **Choose a category (primary entry point).** The first screen shows category cards:
   - **Image classification**
   - **Object detection**
   - **Text / Sentiment**
   - **Tabular** (classification / regression)
   - **Audio** (Phase 2)
   Phase 2/3 categories appear as "Coming soon".
   - The user names the project and enters the class names (e.g., `helmet`, `vest`, `no_helmet`).
   - A pre-filled **task spec** (JSON, Appendix A) is created from the category template. The user can adjust the target platform, max model size, max latency and budget in a form view, with raw JSON available.
   - **Secondary path:** "Not sure? Describe your task." The user types a description in plain English, e.g. *"Detect whether workers are wearing helmets and vests in CCTV photos. It must run on an Android phone offline."* An LLM converts it into the same task spec and picks the category; the user reviews and edits it.
3. **Bring data.** The data screen adapts to the category. Every category offers **Upload** and, where it applies, **Capture**.
   - **Upload:**
     - Folder or zip: subfolder names become labels for classification.
     - CSV/Excel/Parquet with column mapping, for text and tabular.
     - Images plus COCO/YOLO/VOC annotation files, for detection.
   - **Capture (camera):** opens the device camera in the browser (rear camera by default on phones).
     - For image classification, the user picks a class and captures in **burst mode** (e.g., 10 photos in 3 s), and the photos go straight into that class.
     - For object detection, the photos are captured unlabelled and go to the box-drawing step.
     - A live per-class counter (e.g., "helmet: 34/50") and tips ("change angle, lighting and background") guide the user towards varied data.
   - **Capture (microphone):** records fixed-length clips per class (Audio, Phase 2).
   Uploads and captures go straight to object storage through resumable presigned uploads. Each batch creates a **dataset version**.
4. **Label (if needed).**
   - **Classification:** quick bulk labeling for uploads without folders.
   - **Object detection:** every image needs a box around each object, with its class. The **box tool** supports draw, resize, copy-to-next and keyboard shortcuts.
   - **AI-assisted boxes:** an open-vocabulary detector proposes boxes from the class names, and the user accepts, adjusts or deletes them. This is expected to cut labeling time 5–10×.
   - Validation blocks training until every class has the minimum number of boxes.
5. **Validate.** An automatic validation report lists blocking errors and warnings, each with a suggested fix (auto-fix where safe, e.g., remove exact duplicates). The user fixes the issues or acknowledges the warnings.
6. **Choose a budget and train.**
   - Pick Quick (~10 min), Standard (~30–60 min) or Thorough (several hours).
   - The system shows an estimated duration and GPU-minute cost, then queues the job.
   - The user watches live progress (trials, best metric so far, ETA) and can cancel at any time.
7. **Evaluate.**
   - Review the test metrics, confusion matrix, per-class results and worst mistakes.
   - Try the model in the playground: upload or paste input, or use the webcam/mic. For detection, a **live camera mode** draws the predicted boxes on the video frames in the browser.
   - Read the improvement recommendations. Optionally add data and retrain, which creates a new model version.
8. **Export or deploy.**
   - Download exports with snippets, or click **Deploy** to get a hosted HTTPS endpoint.
   - Create API keys, view usage, and roll back versions.
9. **Monitor and iterate (Phase 2).** View endpoint latency, error rate and prediction distribution. Low-confidence predictions can be sent to a labeling queue (active learning), then retrain.

---

## 6. Functional requirements

Priorities: **P0** = required for MVP launch; **P1** = shortly after MVP or late MVP if capacity allows; **P2** = later phase.

### 6.1 Accounts, workspaces, API keys and usage (ACC)
| ID | Requirement | Priority |
|---|---|---|
| ACC-1 | Sign up and sign in with email + password (verified email) and Google OAuth (OIDC). | P0 |
| ACC-2 | Short-lived JWT access tokens (15 min) and rotating refresh tokens in httpOnly, Secure, SameSite cookies; refresh-token reuse detection revokes the session family. | P0 |
| ACC-3 | Each user gets a personal workspace (organisation). Users can create additional workspaces. | P0 |
| ACC-4 | Workspace roles: Owner, Admin, Member, Viewer. Role checks are enforced in the API for every resource. | P0 |
| ACC-5 | Invite members by email; invitations expire after 7 days; Owners can remove members and transfer ownership. | P1 |
| ACC-6 | Workspace-scoped API keys with name, scopes (`inference`, `projects:read`, `projects:write`, `training`), optional expiry, and last-used timestamp. Only a hash is stored; the full key is shown once. | P0 |
| ACC-7 | Usage dashboard: GPU minutes used versus quota, storage used, inference calls, by project and by day. | P0 |
| ACC-8 | Account deletion and workspace deletion with a 14-day soft-delete window, then hard deletion of data, models and object-storage artefacts. | P0 |
| ACC-9 | Two-factor authentication (TOTP). | P1 |
| ACC-10 | SSO (SAML/OIDC) and SCIM provisioning for Enterprise. | P2 |
| ACC-11 | Audit log of security-relevant actions (logins, key creation, role changes, exports, face gallery access). | P1 (P0 for face features) |

### 6.2 Projects and task specs (PRJ)
| ID | Requirement | Priority |
|---|---|---|
| PRJ-1 | Create a project by choosing a **category** card (Image classification, Object detection, Text/Sentiment, Tabular; Audio and later categories shown as "Coming soon"), entering class names, and getting a pre-filled task spec from that category's template. This is the primary entry point. | P0 |
| PRJ-2 | Secondary path: create a project from a plain-English description. An LLM returns a task spec (including the category) that conforms to the JSON Schema (Appendix A). Invalid output is retried up to 2 times, then the user falls back to category selection. | P0 |
| PRJ-3 | The task spec is editable in a form UI and as raw JSON with live schema validation. Every edit is versioned. | P0 |
| PRJ-4 | The LLM receives only the user's description, plus column names and up to 5 sample rows when the user opts in. It never receives full datasets. Prompts and outputs are logged for 30 days for debugging, and are excluded from any vendor training. | P0 |
| PRJ-5 | The LLM flags unsupported or ambiguous requests (e.g., "generate images", "predict stock prices with certainty") and explains what is supported instead. | P0 |
| PRJ-6 | Constraints in the spec (target platform, max size MB, max latency ms, budget mode) drive model selection and export options. | P0 |
| PRJ-7 | Project list with status (Draft, Data needed, Validating, Ready to train, Training, Trained, Deployed), search and archive. | P0 |
| PRJ-8 | Duplicate a project (spec + dataset version reference) to try a variant. | P1 |

### 6.3 Data ingestion (DAT)
| ID | Requirement | Priority |
|---|---|---|
| DAT-1 | Folder upload (drag and drop, `webkitdirectory`) and zip upload. Subfolder names become labels. | P0 |
| DAT-2 | CSV, Excel (.xlsx) and Parquet upload with a column-mapping step (target, features, text column, ID column, timestamp, ignore). Types are inferred and can be overridden. | P0 |
| DAT-3 | Resumable direct-to-storage uploads: S3 multipart with presigned part URLs. Interrupted uploads resume from the last part; uploads up to 20 GB per file and 50k files per batch. | P0 |
| DAT-4 | Server-side checksum (SHA-256) per object for deduplication and integrity. | P0 |
| DAT-5 | Dataset versioning: each upload, labeling session or fix creates an immutable dataset version (manifest + parent pointer). Training always references a specific version. | P0 |
| DAT-6 | Automatic stratified train/val/test split (default 70/15/15; configurable; minimum test count per class enforced). Grouped splits by a user-chosen group column (e.g., patient ID, customer ID) to prevent leakage. | P0 |
| DAT-7 | Time-based splits for time series and time-aware tabular data. | P1 |
| DAT-8 | Import COCO JSON, YOLO txt and Pascal VOC XML annotations for detection. | P0 |
| DAT-9 | **Camera capture in the browser** (getUserMedia):
- rear camera by default on mobile, with a front/rear switch
- single shot and **burst mode** (configurable count and interval)
- capture straight into a class (classification) or unlabelled into the box-drawing queue (detection)
- a live per-class counter against the recommended minimum, and variety tips
- an upload queue that survives tab sleep and resumes on reconnect

Works on desktop Chrome/Edge/Firefox/Safari and mobile Chrome/Safari. | P0 |
| DAT-9a | Microphone capture (fixed-length clips per class, 16 kHz). | P1 (with Audio, Phase 2) |
| DAT-10 | Image formats: JPEG, PNG, WebP, BMP, TIFF, HEIC (converted on ingest). Audio: WAV, MP3, FLAC, OGG, M4A (resampled to 16 kHz mono). Text: UTF-8 CSV/TSV/JSONL/TXT folders. | P0 (image, text, tabular); P1 (audio) |
| DAT-11 | Thumbnail and preview generation; a paginated dataset browser with filter by label, split and validation flag. | P0 |
| DAT-12 | Connectors: Google Drive, S3 bucket, public URL list. | P2 |
| DAT-13 | **Starter dataset library (local MVP).** A curated catalog of ready-made public datasets per category, at least 2 per category: Image classification, Object detection, Text/Sentiment (including Hindi/Hinglish), Tabular classification and Tabular regression.
- Each catalog entry records its source, licence and size.
- Each dataset is subsampled to fit Quick training on a laptop: ≤1,000 images, ≤3,000 text rows.
- One click imports a starter dataset into a project's draft through the normal ingest pipeline (validation, versions and training work unchanged), pre-fills the classes, and can be combined with the user's own data.
- Catalog licences are recorded but not enforced in the local build. They must be reviewed before any public launch. | P0 (local MVP) |

### 6.4 Labeling (LBL)
| ID | Requirement | Priority |
|---|---|---|
| LBL-1 | Classification labeling UI: keyboard shortcuts, multi-select bulk labeling, label creation and merge. | P0 |
| LBL-2 | Bounding-box labeling (object detection):
- draw, resize, move and delete boxes
- copy boxes to the next image
- class picker with number-key shortcuts
- zoom and pan
- "image has no objects" flag
- autosave

Exports to COCO/YOLO. Works with touch on tablets. | P0 |
| LBL-3 | Labeling progress and label distribution chart (images per class; boxes per class for detection). | P0 |
| LBL-4a | **AI-assisted box suggestions for detection.** An open-vocabulary detector (licence-cleared, e.g., OWL-ViT/Grounding-DINO-class) proposes boxes from the class names. The user accepts all, accepts per box, adjusts or deletes. After the first trained model exists, suggestions come from the user's own model. | P0 |
| LBL-4b | AI-assisted pre-labeling for classification and text: zero-shot (CLIP-style for images, NLI or an LLM for text). The user accepts or rejects. | P1 |
| LBL-5 | Active learning: suggest the next samples to label by uncertainty and diversity; a queue fed from low-confidence production predictions. | P2 |
| LBL-6 | Multiple labelers with review/approve and inter-annotator agreement. | P2 |
| LBL-7 | Polygon/mask labeling with SAM-assisted clicks (for segmentation). | P2 |

### 6.5 Data validation (VAL)
Validation runs automatically after each dataset version is created. The report has **Errors** (block training), **Warnings** (must be acknowledged) and **Info**. Each item shows the count, examples, the reason it matters, and a suggested fix, with a one-click auto-fix where safe.

| ID | Check | Default severity | Suggested fix | Priority |
|---|---|---|---|---|
| VAL-1 | Corrupt or unreadable files (decode failure, zero bytes, truncated) | Error | Exclude the files (auto-fix) | P0 |
| VAL-2 | Too few samples: fewer than 10 per class (image/text classification); fewer than 50 rows total (tabular) | Error | Add data; merge rare classes | P0 |
| VAL-3 | Low samples: fewer than 30 per class | Warning | Add data; the estimated achievable accuracy band is shown | P0 |
| VAL-4 | Class imbalance: ratio > 10:1 between largest and smallest class | Warning | Class weights/oversampling (applied automatically) and per-class metrics emphasised; suggest collecting more of the minority class | P0 |
| VAL-5 | Exact duplicates (same hash) | Warning | Deduplicate (auto-fix) | P0 |
| VAL-6 | Near-duplicates (perceptual hash / embedding similarity > 0.95 for images; MinHash for text) | Warning | Review the gallery; keep one | P0 |
| VAL-7 | **Leakage across splits:** duplicates or near-duplicates in both train and test | Error | Re-split with duplicate groups kept together (auto-fix) | P0 |
| VAL-8 | Conflicting labels: the same or near-same sample with different labels | Warning | Review the conflicts | P0 |
| VAL-9 | Tabular: missing values per column; columns > 50% missing | Warning | Impute or drop (choose) | P0 |
| VAL-10 | Tabular: ID-like columns (near-unique integers/strings, UUIDs) | Warning | Exclude from features (auto-fix, default on) | P0 |
| VAL-11 | Tabular: **target leakage**: a feature with suspiciously high single-feature predictive power (e.g., AUC > 0.98) or a name like `*_outcome`, `cancelled_date` | Error until acknowledged | Review and exclude | P0 |
| VAL-12 | Tabular: constant or near-constant columns; high-cardinality categoricals | Info/Warning | Drop or encode | P0 |
| VAL-13 | Image: very small images (< 32 px), extreme aspect ratios, mixed grayscale/RGB | Info/Warning | Resize policy note | P0 |
| VAL-14 | Text: empty or very short texts; language/script mix (Indic, Hinglish) | Info | Suggest Indic/multilingual backbones | P0 |
| VAL-15 | Label noise estimation (confident-learning style) after the first training run | Info | Review the top suspected mislabels | P1 |
| VAL-16 | Detection: boxes out of bounds, zero-area or tiny boxes (< 8 px), images without boxes (not flagged "no objects"), fewer than 50 boxes for a class (Error below 15) | Warning / Error | Fix, label more, or exclude | P0 |
| VAL-17 | Face datasets: missing consent records; faces not detected; multiple faces in an enrolment photo | Error | Upload consent; replace photos | P0 for the face template |

### 6.6 Model selection and training (TRN)
| ID | Requirement | Priority |
|---|---|---|
| TRN-1 | **Rules-based model selection** picks candidate backbones from a curated, licence-approved **model registry**, based on task type, constraints (platform, size, latency), dataset size and language. Example: edge plus < 10 MB gives MobileNetV3/EfficientNet-Lite; server plus fewer than 2k images gives ConvNeXt-Tiny or ViT-S with strong augmentation. | P0 |
| TRN-2 | Every model starts from pretrained weights. Training from random initialisation is not offered. | P0 |
| TRN-3 | Hyperparameter search with Optuna (TPE sampler, median/ASHA-style pruner). The search space covers backbone choice, learning rate, weight decay, augmentation strength, freeze/unfreeze schedule, batch size and epochs (bounded by budget). Tabular uses Optuna over the GBDT parameters. | P0 |
| TRN-4 | **Budget modes:** Quick (~10 min wall-clock, few trials, frozen-backbone + head first), Standard (~30–60 min), Thorough (several hours, wider space, ensembling allowed). Each mode has a hard GPU-minute ceiling. | P0 |
| TRN-5 | Early stopping on the validation metric. Trials are pruned when they fall behind the median at checkpoints. | P0 |
| TRN-6 | Final model = the best trial's config retrained on train+val (configurable), evaluated **once** on the untouched test split. | P0 |
| TRN-7 | **Live progress** over WebSocket (SSE fallback): phase, trial n/N, current and best metric, loss curves, ETA, GPU-minutes consumed. | P0 |
| TRN-8 | **Cancellation:** the user can cancel at any time. Workers stop within 30 seconds, and the best model so far is kept if at least one trial finished. Only the minutes actually used are billed. | P0 |
| TRN-9 | **Retries:** infrastructure failures (preempted spot instance, OOM, worker crash) are retried automatically up to 2 times from the last checkpoint. User-data errors fail fast with a clear message. | P0 |
| TRN-10 | **Reproducibility:** every run records the dataset version ID, split seed, task spec version, search space, all trial params, random seeds, container image digest and library versions. Re-running a model version with "reproduce" gives metrics within ±1% (GPU non-determinism tolerated). | P0 |
| TRN-11 | **Experiment tracking with MLflow:** one MLflow experiment per project and one run per trial (parameters, metrics, artefacts), linked from the UI. | P0 |
| TRN-12 | Mixed precision (bf16/fp16), gradient accumulation, and automatic batch-size finding to avoid OOM. | P0 |
| TRN-13 | Class-imbalance handling (weighted loss / balanced sampling) applied automatically when VAL-4 fires. | P0 |
| TRN-14 | Custom training options for advanced users (P5): choose a backbone, edit the search space, set epochs and the metric to optimise. | P1 |
| TRN-15 | Ensembles (Thorough mode, server deployment only). | P2 |
| TRN-16 | Distributed search with Ray Tune across multiple GPUs for Thorough mode and Team tier. | P2 |
| TRN-17 | Job queue priority by tier (Enterprise > Team > Pro > Free), with fair-share scheduling per workspace. | P0 |

### 6.7 Evaluation (EVL)
| ID | Requirement | Priority |
|---|---|---|
| EVL-1 | Metrics per task: **classification**: accuracy, macro F1, per-class precision/recall/F1, ROC-AUC/PR-AUC (binary); **regression**: RMSE, MAE, R², MAPE where safe; **detection**: mAP@0.5, mAP@0.5:0.95, per-class AP; **NER**: entity-level F1; **forecasting**: MASE, sMAPE; **face**: TAR@FAR, identification accuracy, unknown rejection rate. | P0 (Phase 1 tasks) |
| EVL-2 | Confusion matrix (counts and normalised), with click-through to the samples in each cell. | P0 |
| EVL-3 | Worst-mistakes gallery: highest-confidence wrong predictions on the test set. | P0 |
| EVL-4 | Threshold tuning UI for binary and multi-label tasks (precision/recall trade-off) and calibration (reliability diagram, ECE). | P1 |
| EVL-5 | **Playground:** upload a file, paste text, fill a tabular form, or use the webcam/mic. Shows top-k predictions with confidence and latency. | P0 |
| EVL-6 | **Improvement recommendations engine:** rules over the metrics, data stats and errors, e.g. per-class data shortage, confusion between specific pairs, overfitting gap (train ≫ val), suspected label noise, a suggestion to try the Standard budget, or an imbalance warning. | P0 |
| EVL-7 | Explainability: feature importance (SHAP for tabular), saliency/Grad-CAM for images, token attributions for text. | P1 (tabular SHAP P0) |
| EVL-8 | Baseline comparison: show the trivial baseline (majority class / mean predictor) next to the model. | P0 |
| EVL-9 | Model version comparison (side-by-side metrics and per-class deltas). | P1 |
| EVL-10 | Slice metrics by a user-chosen column (e.g., region, device, language) and, for face models, by available demographic or capture-condition tags. | P1 (P0 for the face template) |

### 6.8 Export (EXP)
| ID | Requirement | Priority |
|---|---|---|
| EXP-1 | Export formats per task, from a maintained **export matrix**. MVP:
- image classification: ONNX, TFLite, Core ML and TF.js
- **object detection: ONNX, TFLite and Core ML**, with NMS included in the graph or documented as post-processing in the snippet; TF.js is best effort
- text classification: ONNX, plus TFLite/Core ML for small encoders (best effort)
- tabular: ONNX (via onnxmltools/skl2onnx) plus native LightGBM/XGBoost files | P0 |
| EXP-2 | Optional quantisation: dynamic INT8 (ONNX), full-integer INT8 with a calibration set (TFLite), FP16 (Core ML/TFLite). Accuracy is re-measured on the test set after quantisation and the delta is shown. | P0 (dynamic/FP16), P1 (full-integer) |
| EXP-3 | Every export is verified automatically: run the test subset through the exported model and check that its outputs agree with the native model (tolerance per format). A mismatch blocks the download. | P0 |
| EXP-4 | Export bundle: model file(s), `model.json` labels/metadata, preprocessing spec, model card, licence notices. | P0 |
| EXP-5 | Copy-paste snippets per platform (§4.2), rendered with the actual input shape, labels and normalisation values. | P0 |
| EXP-6 | Exports are generated asynchronously, cached per model version and format, and downloaded through presigned URLs (15 min expiry). | P0 |
| EXP-7 | Size and latency estimates per export (measured on a reference CPU; device estimates from benchmarks). | P1 |

### 6.9 Deployment and hosted inference (DEP), Phase 2 (v0.3)
> Deferred to Phase 2. Users will deploy their trained models on the platform itself as hosted endpoints. In the local MVP, models are downloaded as export files and tried in the in-app playground.
| ID | Requirement | Priority |
|---|---|---|
| DEP-1 | One-click deploy of a model version to a hosted HTTPS endpoint: `POST /v1/predict/{deployment_id}`. | P0 |
| DEP-2 | Authentication by API key (`Authorization: Bearer <key>`), scoped to the workspace with the `inference` scope. | P0 |
| DEP-3 | Input via JSON (base64 image, text, feature object) or multipart. Batch requests up to 64 items. | P0 |
| DEP-4 | Version pinning, rollback to a previous version, and alias tags (`production`, `staging`). | P0 (rollback), P1 (aliases) |
| DEP-5 | Rate limits and quotas per key and per workspace by tier. Returns 429 with `Retry-After`. | P0 |
| DEP-6 | Serving through ONNX Runtime in a shared CPU pool for small models; dedicated or GPU pools (Triton) for large models and Team/Enterprise. | P0 (CPU shared), P2 (dedicated GPU) |
| DEP-7 | Cold start under 5 seconds for scale-to-zero deployments (Free tier); always-warm for Pro and above. | P0 |
| DEP-8 | Endpoint dashboard: request count, p50/p95 latency, error rate, prediction class distribution over time. | P1 |
| DEP-9 | Optional request logging, off by default and opt-in, with retention per tier, used for active learning and drift detection. | P2 |
| DEP-10 | Async batch prediction jobs (upload a CSV or folder, download the results). | P1 |
| DEP-11 | Webhook on prediction-job completion. | P2 |

### 6.10 Notifications (NTF)
| ID | Requirement | Priority |
|---|---|---|
| NTF-1 | In-app notification centre (job finished, failed, export ready, quota at 80%/100%). | P0 |
| NTF-2 | Email notifications for job completion or failure and quota alerts (per-user opt-out). | P0 |
| NTF-3 | Webhooks per workspace (job.completed, job.failed, deployment.updated) with HMAC signature and retries. | P2 |
| NTF-4 | Browser push notification when a long job finishes. | P2 |

### 6.11 Billing and quotas (BIL)
| ID | Requirement | Priority |
|---|---|---|
| BIL-1 | Free tier with a monthly GPU-minute quota, a storage cap, a project cap, an inference call cap and Quick budget only (see §13). | P0 |
| BIL-2 | Paid plans as subscriptions (Razorpay for INR, Stripe for international), with prorated upgrades and downgrades. | P0 |
| BIL-3 | GPU minutes metered per job from **actual worker GPU time** (not wall-clock queue time), written as `usage_events` in the same transaction that updates the job state. | P0 |
| BIL-4 | Quota enforcement before job start (an estimated cost must fit the remaining quota) and during the job (hard stop at quota with the best model kept). Uses Postgres row-level locks or atomic counters to prevent double-spend under concurrency. | P0 |
| BIL-5 | Pay-as-you-go GPU-minute top-ups on paid plans, and an optional monthly spend cap. | P1 |
| BIL-6 | Invoices with GST (India) and tax handling per region; downloadable invoice PDFs. | P0 for paid launch |
| BIL-7 | Usage export (CSV) and a billing API for Team/Enterprise. | P1 |
| BIL-8 | Abuse controls: free-tier sign-up throttling, disposable-email blocking, payment-method verification for GPU-heavy usage. | P0 |

### 6.12 Face recognition template (FACE)
See §12 for the policy. Functional requirements:

| ID | Requirement | Priority (Phase 2) |
|---|---|---|
| FACE-1 | Pipeline: face detection (RetinaFace/MTCNN-class detector) → alignment → embedding model (ArcFace/FaceNet-style, licence-cleared) → gallery matching with cosine similarity. This is **not** a classifier. | P0 |
| FACE-2 | Galleries: create a gallery; enrol an identity with 3–10 photos. Quality checks: one face per photo, minimum face size 112 px, blur/pose/illumination score. Embeddings are stored as one vector per photo, plus a mean template. | P0 |
| FACE-3 | **Instant add/remove of people** without retraining: enrolment and deletion update only the gallery rows (pgvector). | P0 |
| FACE-4 | Matching returns the best identity, its score, and **"Unknown"** when the score is below the threshold. Top-k candidates are optional. | P0 |
| FACE-5 | A threshold tuning UI on the user's own labelled verification set (genuine/impostor pairs) shows TAR/FAR and lets the user pick an operating point. A conservative default threshold is preset. | P0 |
| FACE-6 | Enrolment is blocked unless a **consent record** exists for the identity (§12.3). | P0 |
| FACE-7 | Embeddings and photos are encrypted at rest with per-workspace keys; there is no export of raw embeddings in v1. | P0 |
| FACE-8 | Deletion of an identity removes the photos, embeddings and derived templates within 24 hours, including from backups within the backup retention window, with a deletion certificate in the audit log. | P0 |
| FACE-9 | Liveness/anti-spoofing (passive) is offered as an add-on for access-control use cases. | P1 |
| FACE-10 | Evaluation includes slice metrics by capture condition (lighting, camera, angle) and, where the user provides them, demographic tags. A pre-deployment checklist must be completed (§12.5). | P0 |

---

## 7. Non-functional requirements

### 7.1 Performance
| ID | Requirement |
|---|---|
| NFR-P1 | API p95 latency < 300 ms for CRUD endpoints; < 1 s for dataset browser pages of 100 items. |
| NFR-P2 | Upload throughput limited only by the client's bandwidth (direct to object storage); the UI stays responsive with 50k-file uploads. |
| NFR-P3 | Validation of 10k images completes in < 5 min; 1M-row CSV in < 3 min. |
| NFR-P4 | Queue wait p90 < 2 min for paid tiers and < 15 min for the free tier under normal load. |
| NFR-P5 | Hosted inference p95 (server-side, excluding network) for MVP-size models: image classification < 80 ms, text classification < 60 ms, tabular < 20 ms (CPU pool). |
| NFR-P6 | Progress events reach the UI within 2 s of the worker emitting them. |

### 7.2 Scalability
| ID | Requirement |
|---|---|
| NFR-S1 | Stateless API pods scale horizontally behind a load balancer. |
| NFR-S2 | MVP: at least 50 concurrent training jobs (Celery workers on a GPU pool). Phase 2: 500+ via Kubernetes GPU node pools with an autoscaler (KEDA on queue depth) and spot instances for Free/Quick jobs. |
| NFR-S3 | Inference: autoscaling CPU pool to 1,000 RPS aggregate at MVP; per-deployment autoscaling later. |
| NFR-S4 | Object storage is the system of record for samples and artefacts; Postgres holds only metadata and manifests (pointers). |

### 7.3 Reliability
| ID | Requirement |
|---|---|
| NFR-R1 | Training job success rate > 97% (§2.3); infrastructure failures retried automatically (TRN-9). |
| NFR-R2 | Jobs are idempotent and resumable from checkpoints, which are written every epoch and at least every 5 minutes. |
| NFR-R3 | Platform API availability ≥ 99.9% monthly; hosted inference ≥ 99.9% (Pro) and ≥ 99.95% (Enterprise SLA). |
| NFR-R4 | Postgres: managed, multi-AZ, point-in-time recovery (7 days at MVP, 30 days later). RPO ≤ 5 min, RTO ≤ 1 h. |
| NFR-R5 | Graceful degradation: if the LLM provider is down, task creation falls back to templates; if MLflow is down, jobs keep running and log to a local buffer that syncs later. |

### 7.4 Security
| ID | Requirement |
|---|---|
| NFR-SEC1 | TLS 1.2+ everywhere; HSTS; secure cookies; CSRF protection for cookie-authenticated endpoints. |
| NFR-SEC2 | Encryption at rest for Postgres, Redis persistence and object storage (SSE-KMS). Biometric data uses per-workspace data keys (envelope encryption). |
| NFR-SEC3 | Strict tenant isolation: every query is scoped by `organization_id`; object keys are prefixed by organisation; presigned URLs are short-lived and path-scoped. Automated tests cover cross-tenant access. |
| NFR-SEC4 | Training workers run user-influenced configs only. **No user-supplied code executes** in v1. Workers run in isolated containers without outbound internet except to an allowlist (artefact store, MLflow, model mirror). |
| NFR-SEC5 | Pretrained weights are pulled from an **internal mirror** of the approved registry (pinned hashes), never directly from the internet at job time. |
| NFR-SEC6 | File parsing is hardened: zip bomb and path traversal checks, size limits, image decompression-bomb limits, malware scanning of uploads (ClamAV or a cloud equivalent). |
| NFR-SEC7 | API keys are hashed (argon2/SHA-256 with prefix lookup); secrets live in a secrets manager; keys rotate. |
| NFR-SEC8 | OWASP ASVS L2 baseline; annual third-party penetration test before Enterprise launch; a dependency and container scanning pipeline. |
| NFR-SEC9 | Prompt-injection hygiene for the LLM task-spec step: the LLM output is data validated against a schema, and it never triggers actions directly. |

### 7.5 Privacy
| ID | Requirement |
|---|---|
| NFR-PR1 | Customer data is used only to provide the customer's own models. It is never used to train shared or platform models, and never shared with other customers. |
| NFR-PR2 | Data processing agreement (DPA) available; subprocessor list is public. |
| NFR-PR3 | Data residency: India region (Mumbai) by default for Indian customers; EU region at Team/Enterprise (Phase 2). |
| NFR-PR4 | PII detection on text/tabular uploads (Info-level warning with column/row hints). |
| NFR-PR5 | Biometric data rules in §12. |

### 7.6 Data retention
| Data | Default retention |
|---|---|
| Raw uploads and dataset versions | Until the user deletes them. Free tier: deleted after 90 days of workspace inactivity, with 2 warning emails |
| Model artefacts and exports | Until the user deletes them; cached exports regenerated on demand |
| Training logs and MLflow runs | 180 days (Free), 1 year (Pro), configurable (Team/Enterprise) |
| Inference request payloads | Not stored by default; opt-in logging for 30 days (Pro), configurable (Enterprise) |
| LLM prompt/response logs | 30 days |
| Face photos and embeddings | Until the identity is removed, or until the purpose ends (the user sets a mandatory retention date per gallery, max 3 years). Removed within 24 h of a deletion request |
| Backups | 7–30 days rolling; deletions propagate when backups expire |
| Billing records | As required by tax law (8 years for India GST records) |

### 7.7 Compliance
- **India DPDP Act 2023:**
  - The platform acts as a **Data Processor** for customer data, and the customer is the **Data Fiduciary**.
  - The platform is the Data Fiduciary for its own account data, with notice, consent and grievance-officer contact.
  - Breach notification procedures are aligned with the DPDP Rules once notified.
- **GDPR** (EU customers, Phase 2): DPA with SCCs, Article 28 processor terms, data-subject request tooling. Face data is **special-category biometric data** (Art. 9), so the customer must have a lawful basis plus an Art. 9 condition.
- **Illinois BIPA** (and similar US state laws such as Texas CUBI and Washington): written notice and **written release** from each enrolled person, a public retention and destruction schedule, and no sale of biometric data. Because of private-right-of-action exposure, face features may be **geo-restricted for Illinois customers** until legal review completes (open question Q4).
- **SOC 2 Type II** targeted within 12 months of GA (Enterprise requirement); ISO 27001 later.
- **Accessibility:** WCAG 2.1 AA for the web app.

---

## 8. Tech stack decisions

### 8.1 Frontend
| Concern | Choice | Reason |
|---|---|---|
| Language/framework | **React 18+ with TypeScript (strict)** | Team familiarity, ecosystem, typed API client |
| Build tool | **Vite** | Fast dev server and builds |
| Routing | **TanStack Router** (or React Router v6+ if the team prefers) | Type-safe routes and search params, which suit filter-heavy dataset views |
| Server state | **TanStack Query** | Caching, retries, pagination and optimistic updates for REST |
| Client state | **Zustand** (small, local UI state only) | Minimal boilerplate; most state is server state |
| UI components | **shadcn/ui (Radix primitives) + Tailwind CSS** | Accessible primitives; components owned in-repo |
| Forms and validation | **react-hook-form + Zod**, with Zod schemas generated from the backend's OpenAPI | One source of truth for the task spec form |
| Charts | **Recharts** for standard charts; **ECharts** for heavy charts (large confusion matrices, loss curves with many trials) | Ease of use versus performance |
| Labeling canvas | **Konva (react-konva)** for boxes; polygons/masks later | Mature 2D canvas, good hit-testing |
| Uploads | **Uppy** with the S3 multipart plugin | Resumable chunked direct uploads, folder support |
| Realtime | Native WebSocket with reconnect; SSE fallback | Training progress |
| In-browser inference (playground, TF.js/onnxruntime-web demos) | **onnxruntime-web** (WASM/WebGPU) | Same ONNX artefact as the server |
| API client | **openapi-typescript** generated types + a thin fetch wrapper | Type safety against FastAPI's OpenAPI schema |
| Testing | Vitest, Testing Library, Playwright (E2E) | Standard |

### 8.2 Backend language: comparison

| Criterion | Python (FastAPI) | Node.js (NestJS/Fastify) | Go |
|---|---|---|---|
| ML ecosystem | **Best:** PyTorch, timm, transformers, Optuna, MLflow, ONNX tooling, pandas, LightGBM are all native | Weak; ML must be called through Python services | Weak; ML must be called through Python services |
| Code sharing with training workers | **Shared** schemas (Pydantic), validation, preprocessing specs, dataset readers | None (schemas duplicated in TS and Python) | None (schemas duplicated) |
| Raw API performance | Good with async (uvicorn, uvloop); slower than Go for CPU-bound work | Good I/O performance | **Best:** low latency, low memory, high concurrency |
| Async support | Native async/await (asyncio); some libraries are sync and need thread pools | Native, mature event loop | Goroutines; excellent |
| Developer speed | **High:** Pydantic models give validation + OpenAPI for free | High | Medium (more boilerplate) |
| Hiring (India) | Large pool of Python backend and ML engineers | Very large pool of JS/TS developers | Smaller, more senior pool |
| Typing | Pydantic + mypy/pyright (good, not compile-time strict) | TypeScript (strong) | Static, compiled |
| Ops footprint | Moderate (larger images, GIL for CPU-bound work) | Moderate | **Small** single binaries |
| Fit for this product | **Strong:** the product's core is ML orchestration | Medium: good for the web tier, but creates a two-language split for no benefit | Medium for the control plane; **strong for a future inference gateway** |

**Recommendation: Python + FastAPI for the API and control plane.**
- The training workers, validation, export pipeline and MLflow integration are all Python. One language means:
  - shared Pydantic models for the task spec, job config and dataset manifests
  - one hiring profile
  - no translation layer between the API and the workers
- FastAPI's async performance is more than enough for a control-plane API whose heavy work is offloaded to queues.
- Ultimately CPU-bound API paths (e.g., large manifest diffing) run in worker processes, not in request handlers.

**Where Go may make sense later:** a **high-throughput inference gateway** in Phase 3. It would handle API-key auth, rate limiting, request routing to model servers, and usage metering at tens of thousands of RPS. Go's latency, memory footprint and concurrency model fit that layer well; the model runtime behind it stays ONNX Runtime/Triton. Revisit this when inference traffic exceeds roughly 2–3k RPS sustained, or when gateway p99 overhead exceeds 10 ms on FastAPI.

**Node.js** is not recommended for the backend: it would add a second language without a capability we need. The TypeScript frontend still benefits from generated API types.

### 8.3 Database: comparison

| Criterion | PostgreSQL | MongoDB |
|---|---|---|
| Relational data (users, orgs, memberships, projects, jobs, billing) | **Native:** foreign keys, joins, constraints | Manual references; joins via `$lookup`; weaker integrity |
| Transactions for quotas and billing | **ACID** with row-level locks (`SELECT … FOR UPDATE`), serialisable isolation where needed | Multi-document transactions exist but are more limited and costlier; harder to reason about for money |
| Flexible configs (task specs, search spaces, metrics blobs) | **JSONB** with GIN indexes and JSON Schema validation in the app | Native document model (a strength) |
| Vector search (face embeddings, near-duplicate detection) | **pgvector** (HNSW/IVFFlat) in the same database and transaction as consent and identity rows | Atlas Vector Search (managed Atlas only) |
| Reporting and analytics (usage, billing) | SQL, window functions, materialised views | Aggregation pipelines (more verbose) |
| Operational maturity / managed options | RDS, Cloud SQL, Neon, Supabase, AlloyDB | Atlas, DocumentDB (partial compatibility) |
| Fit | **Strong** | Medium |

**Recommendation: PostgreSQL (16+) with JSONB and pgvector.**
- The core domain is relational and money-adjacent (quotas, GPU-minute metering, subscriptions), which needs reliable transactions and constraints.
- JSONB covers the genuinely flexible parts (task specs, search spaces, per-task metrics) without giving up integrity elsewhere.
- pgvector keeps face embeddings in the same transactional boundary as consent records and identity deletion. Deleting an identity and its vectors happens atomically, which matters for biometric compliance.
- For very large galleries (> ~5M vectors per workspace), revisit a dedicated vector store (Qdrant/Milvus). That is not expected in v1.

**Datasets and model files are not stored in the database.** Raw samples, dataset manifests (Parquet), checkpoints, exports and reports live in **object storage** (S3 in production; MinIO for local development and possible on-prem later). Postgres stores metadata and object keys only.

### 8.4 Other components
| Component | MVP choice | Later / scale choice | Notes |
|---|---|---|---|
| Job queue | **Celery + Redis** (separate queues: `validate`, `train-gpu`, `export`, `notify`) | Kubernetes Jobs / Argo Workflows or Ray on K8s with GPU node pools, KEDA autoscaling on queue depth; Redis Streams or RabbitMQ if Celery limits bite | Workers are pinned to GPU types per budget mode |
| Object storage | **AWS S3** (ap-south-1) with SSE-KMS, lifecycle rules | Same; multi-region for Enterprise | MinIO in dev/CI |
| Uploads | **S3 multipart presigned uploads** (Uppy) | Same; optionally tus for non-S3 connectors | Completion callback validates checksums |
| Experiment tracking | **MLflow** (tracking server + Postgres backend + S3 artefacts) | Same, sharded per region | The UI links to runs; Pro+ gets read-only MLflow access |
| Hyperparameter search | **Optuna** (RDB storage in Postgres) | **Ray Tune** for multi-GPU/distributed Thorough runs | |
| Training libraries | PyTorch, timm, Hugging Face transformers/datasets/PEFT, LightGBM/XGBoost/CatBoost, scikit-learn | + Ultralytics only with a commercial licence; MMDetection/MMPose (Apache) | All weights from the approved registry |
| Export tooling | `torch.onnx`/dynamo export, onnxruntime quantisation, coremltools, ai-edge-torch or onnx2tf for TFLite, tfjs-converter, onnxmltools/skl2onnx | | Verified by EXP-3 |
| Inference serving | **ONNX Runtime** in a FastAPI-based model server (CPU pool) | **NVIDIA Triton** (GPU, dynamic batching) or **BentoML** for packaging; Go gateway (§8.2) | |
| Auth | JWT (access + refresh rotation) + Google OAuth via OIDC (Authlib) | SAML/OIDC SSO (WorkOS or Keycloak) | |
| LLM (task spec) | Hosted LLM API with structured JSON output validated against a JSON Schema (e.g., Anthropic Claude) | Provider abstraction for fallback | §6.2 data minimisation |
| Monitoring | **Prometheus + Grafana** (metrics), **OpenTelemetry** traces, **Sentry** (errors), **Loki** or CloudWatch (logs) | Same + GPU metrics via DCGM exporter | SLO dashboards per §7 |
| Infra / IaC | Docker, Terraform, managed Postgres (RDS), ElastiCache Redis, ECS or EKS | EKS with GPU node groups (spot + on-demand) | |
| CI/CD | GitHub Actions; container scanning (Trivy); migrations via Alembic | Same + canary deploys | |
| Payments | Razorpay (INR, GST), Stripe (international) | | |
| Email | Amazon SES or Postmark | | |

---

## 9. System architecture

### 9.1 Components

```
┌──────────────────────────┐        ┌───────────────────────────────────────────────┐
│ React SPA (Vite, TS)     │ HTTPS  │ API service (FastAPI, stateless, autoscaled)  │
│ - projects, data, label  ├───────►│ - auth, RBAC, projects, datasets, jobs        │
│ - training live view ◄───┼── WS ──┤ - presigned URLs, quotas, billing webhooks    │
│ - eval, playground       │        │ - WebSocket/SSE progress fan-out              │
└────────────┬─────────────┘        └───┬───────────┬───────────┬──────────────┬────┘
             │ presigned PUT/GET        │           │           │              │
             ▼                          ▼           ▼           ▼              ▼
┌──────────────────────────┐  ┌────────────┐ ┌──────────┐ ┌──────────────┐ ┌─────────────┐
│ Object storage (S3/MinIO)│  │ PostgreSQL │ │  Redis   │ │ LLM provider │ │ Payments    │
│ raw samples, manifests,  │  │ metadata,  │ │ broker,  │ │ (task spec)  │ │ (Razorpay/  │
│ checkpoints, exports     │  │ JSONB,     │ │ pub/sub, │ └──────────────┘ │  Stripe)    │
└────────────▲─────────────┘  │ pgvector   │ │ rate lim │                  └─────────────┘
             │                └─────▲──────┘ └────┬─────┘
             │                      │             │ Celery queues
             │        ┌─────────────┴─────────────▼───────────────────────────┐
             │        │ Workers                                                │
             ├────────┤ - CPU: validation, manifests, thumbnails, exports      │
             │        │ - GPU: training (Optuna trials), embedding, eval       │
             │        └──────────┬───────────────────────────────┬────────────┘
             │                   │ params/metrics/artefacts      │ model registry mirror
             │                   ▼                               ▼
             │        ┌──────────────────────┐        ┌──────────────────────┐
             │        │ MLflow tracking      │        │ Approved weights     │
             │        │ (Postgres + S3)      │        │ mirror (pinned hash) │
             │        └──────────────────────┘        └──────────────────────┘
             │
┌────────────┴───────────────────────────────────────────────────────────────────┐
│ Inference plane: gateway (FastAPI now, Go later) → ONNX Runtime pool / Triton  │
│ - API-key auth, rate limits, usage events → Postgres (batched)                 │
└────────────────────────────────────────────────────────────────────────────────┘
Observability: OpenTelemetry → Prometheus/Grafana, Sentry, Loki (all services)
```

### 9.2 Training job flow
1. **Submit.**
   - The React app calls `POST /projects/{id}/training-jobs` with a dataset version, budget mode and task spec version.
   - The API checks RBAC, checks that validation passed (or that the warnings were acknowledged), and estimates the GPU minutes.
   - In **one Postgres transaction** it locks the workspace quota row, reserves the estimated minutes, inserts `training_jobs` (status `queued`), and commits.
2. **Enqueue.** The API publishes a Celery task (`train-gpu` queue, priority by tier) carrying only the job ID. Workers read everything else from Postgres and S3.
3. **Prepare.** A GPU worker claims the job and sets the status to `running`.
   - It loads the dataset manifest (Parquet in S3) and streams samples from S3 with a local NVMe cache.
   - It resolves the candidate backbones from the model registry and pulls the pinned weights from the internal mirror.
   - It records a reproducibility manifest.
4. **Search.**
   - Optuna creates a study (Postgres RDB storage).
   - Each trial logs to MLflow, writes `trials` rows, and publishes progress to Redis pub/sub (`job:{id}`). The API fans the progress out to subscribed WebSocket clients.
   - Weak trials are pruned, and checkpoints are written to S3 so the job can resume.
5. **Finalise.** The worker retrains the best config (train+val, if configured) and evaluates **once** on the test split. It writes the evaluation report, the recommendations and the model card, then registers a `models`/`model_versions` row and the MLflow model.
6. **Export.** Default exports for the task are queued on the `export` (CPU) queue. Each export is generated, verified against the native model (EXP-3), and stored in S3, with a row in `exports`.
7. **Settle billing.** In one transaction: record the actual GPU seconds as `usage_events`, release the unused reservation, and set the job to `succeeded`/`failed`/`cancelled`. Notifications are sent.
8. **Deploy (optional).** `POST /deployments` loads the chosen ONNX artefact into the inference pool. The gateway routes `/v1/predict/{deployment_id}` there, enforces API-key auth and rate limits, and writes batched `usage_events`.

**Cancellation.** The API sets `cancel_requested=true` and publishes on `job:{id}:control`. The worker checks between batches and trials (≤ 30 s), stops, keeps the best completed trial, and settles billing for the minutes used.

**Failure handling.**
- Preemption or worker loss is detected by a heartbeat timeout (60 s). The job is requeued from the last checkpoint, up to 2 times.
- Deterministic user-data errors (e.g., a label missing from all training samples after the split) fail fast with an actionable message and no retry.

---

## 10. Data model

All tables have `id` (UUIDv7), `created_at`, `updated_at`. Tenant-owned tables carry `organization_id` (indexed; part of every query predicate). Soft-deletable tables have `deleted_at`.

### 10.1 Entity overview

```
users ─< memberships >─ organizations ─< projects ─< datasets ─< dataset_versions ─< samples
                              │             │                         │                │
                              │             │                         │              labels
                              │             ├─< training_jobs ─< trials
                              │             │         │
                              │             ├─< models ─< model_versions ─< exports
                              │             │                  │
                              │             │                  └─< deployments
                              │             └─< face_galleries ─< face_identities ─< face_embeddings
                              │                                        │
                              │                                        └─< consent_records
                              ├─< api_keys
                              ├─< usage_events
                              ├─< subscriptions / quotas
                              └─< audit_logs
```

### 10.2 Tables

| Table | Key columns | Notes |
|---|---|---|
| `users` | email (unique, citext), password_hash (nullable), google_sub (nullable, unique), name, email_verified_at, totp_secret_enc, last_login_at | |
| `organizations` | name, slug (unique), plan, region, data_residency, created_by | Every user gets a personal org |
| `memberships` | user_id → users, organization_id → organizations, role (owner/admin/member/viewer) | Unique (user_id, organization_id) |
| `projects` | organization_id, name, task_type, status, current_spec (JSONB), spec_version, created_by, archived_at | GIN index on `current_spec` |
| `task_spec_versions` | project_id, version, spec (JSONB), source (template/llm/user_edit), llm_prompt_ref | Immutable history |
| `datasets` | project_id, organization_id, name, modality (image/text/tabular/audio/...), current_version_id | One logical dataset per project at MVP |
| `dataset_versions` | dataset_id, version_no, parent_version_id, manifest_uri (S3 Parquet), sample_count, label_stats (JSONB), split_seed, split_strategy, validation_status, validation_report (JSONB), content_hash | Immutable after `sealed_at` |
| `samples` | dataset_version_id, object_key, sha256, phash/simhash, mime, bytes, width/height/duration, split (train/val/test), metadata (JSONB) | At very large scale, rows live in the Parquet manifest and Postgres keeps only an index; for MVP, rows up to ~1M per version |
| `labels` | sample_id, label_set_id, class_name, bbox (JSONB: x,y,w,h), source (user/import/ai_prelabel), confidence, labeled_by, reviewed_by | Multiple labels per sample (multi-label, detection) |
| `label_sets` | project_id, classes (JSONB ordered list), version | The class order used by the model |
| `training_jobs` | project_id, organization_id, dataset_version_id, spec_version_id, budget_mode, status (queued/running/succeeded/failed/cancelled), cancel_requested, priority, reserved_gpu_seconds, used_gpu_seconds, attempt, worker_id, heartbeat_at, error_code, error_message, mlflow_experiment_id, repro_manifest (JSONB), started_at, finished_at | Index (status, priority, created_at) |
| `trials` | training_job_id, trial_no, params (JSONB), state (running/complete/pruned/failed), val_metric, metrics (JSONB), gpu_seconds, mlflow_run_id, checkpoint_uri | |
| `models` | project_id, organization_id, name, task_type, production_version_id | |
| `model_versions` | model_id, version_no, training_job_id, best_trial_id, base_model (registry ID + licence), metrics (JSONB), eval_report_uri, model_card_uri, artifact_uri, label_set_id, input_spec (JSONB), status | Immutable |
| `exports` | model_version_id, format (onnx/tflite/coreml/tfjs/native), quantization (none/fp16/int8_dynamic/int8_full), artifact_uri, size_bytes, verified (bool), verify_report (JSONB), accuracy_delta | Unique (model_version_id, format, quantization) |
| `deployments` | organization_id, model_id, model_version_id, name, status, runtime (onnxruntime/triton), hardware (cpu-shared/cpu-dedicated/gpu), min_replicas, max_replicas, endpoint_slug, alias | |
| `api_keys` | organization_id, name, prefix (indexed), key_hash, scopes (text[]), expires_at, last_used_at, created_by, revoked_at | |
| `usage_events` | organization_id, project_id (nullable), kind (gpu_seconds/storage_gb_day/inference_calls/llm_tokens/export), quantity, unit, ref_type, ref_id, occurred_at, idempotency_key (unique) | Append-only; partitioned by month |
| `quotas` | organization_id, period_start, gpu_seconds_limit, gpu_seconds_used, gpu_seconds_reserved, storage_bytes_limit, inference_calls_limit | Row locked during reservation |
| `subscriptions` | organization_id, plan, provider (razorpay/stripe), provider_subscription_id, status, current_period_end | |
| `face_galleries` | organization_id, project_id, name, threshold, embedding_model_id, purpose, retention_until, region | |
| `face_identities` | gallery_id, organization_id, display_name_enc, external_ref, status (active/deleted), consent_record_id (NOT NULL) | Enrolment is blocked without consent |
| `face_embeddings` | identity_id, gallery_id, organization_id, embedding vector(512), quality_score, photo_object_key_enc, is_template | HNSW index (cosine) per gallery; cascade delete with identity |
| `consent_records` | organization_id, gallery_id, subject_ref, consent_method (written/e-sign/checkbox_attested), document_object_key, purpose, jurisdiction, granted_at, expires_at, withdrawn_at, attested_by_user_id | Required for every identity (§12.3) |
| `audit_logs` | organization_id, actor_user_id/api_key_id, action, target_type, target_id, ip, user_agent, details (JSONB) | Append-only; retained 1 year+ |
| `notifications` | user_id, organization_id, type, payload (JSONB), read_at | |

---

## 11. REST API overview

Base path `/api/v1` for the app API (cookie/JWT auth), and `/v1` on the inference host (API-key auth). JSON throughout, OpenAPI 3.1 generated by FastAPI. Cursor pagination (`?cursor=&limit=`). Errors use RFC 9457 Problem Details. Mutations accept an `Idempotency-Key` header where retries are likely.

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register` | Create an account (email/password) |
| POST | `/auth/login` | Log in; sets the refresh cookie and returns an access token |
| POST | `/auth/refresh` | Rotate the refresh token, issue an access token |
| POST | `/auth/logout` | Revoke the session |
| GET | `/auth/google/start` · `/auth/google/callback` | Google OAuth (OIDC) |
| GET | `/me` | Current user and memberships |
| GET/POST | `/orgs` | List / create workspaces |
| GET/PATCH/DELETE | `/orgs/{orgId}` | Workspace details / update / delete |
| GET/POST/DELETE | `/orgs/{orgId}/members[/{userId}]` | Members and invitations |
| GET/POST/DELETE | `/orgs/{orgId}/api-keys[/{keyId}]` | Manage API keys |
| GET | `/orgs/{orgId}/usage?from=&to=&group_by=` | Usage and quota |
| GET | `/templates` | Task templates and their default specs |
| POST | `/orgs/{orgId}/projects` | Create a project from a template or spec |
| POST | `/orgs/{orgId}/projects/from-description` | Plain English → task spec (LLM), returns a draft spec |
| GET/PATCH/DELETE | `/projects/{projectId}` | Project details / update / archive |
| PUT | `/projects/{projectId}/spec` | Save a new task spec version (validated) |
| POST | `/projects/{projectId}/datasets/{datasetId}/uploads` | Start an upload session; returns multipart presigned URLs |
| POST | `/uploads/{uploadId}/complete` | Finalise the upload (checksums); triggers ingestion |
| POST | `/projects/{projectId}/datasets/{datasetId}/imports` | Import COCO/YOLO/VOC/CSV with a column mapping |
| GET | `/datasets/{datasetId}/versions` | List dataset versions |
| GET | `/dataset-versions/{versionId}` | Version details, label stats, split summary |
| GET | `/dataset-versions/{versionId}/samples?label=&split=&flag=` | Browse samples (presigned thumbnail URLs) |
| POST | `/dataset-versions/{versionId}/labels:batch` | Create/update labels in bulk (creates a new version on seal) |
| POST | `/dataset-versions/{versionId}/seal` | Freeze a version and run validation |
| GET | `/dataset-versions/{versionId}/validation` | Validation report |
| POST | `/dataset-versions/{versionId}/validation/fixes` | Apply auto-fixes (creates a new version) |
| POST | `/projects/{projectId}/training-jobs` | Start training (dataset version, budget, spec version); returns a cost estimate and job |
| GET | `/training-jobs/{jobId}` | Status, progress, best metric |
| POST | `/training-jobs/{jobId}/cancel` | Cancel the job |
| GET | `/training-jobs/{jobId}/trials` | Trial list with params and metrics |
| WS | `/training-jobs/{jobId}/events` | Live progress stream (SSE fallback: `GET …/events` with `text/event-stream`) |
| GET | `/projects/{projectId}/models` | Models and versions |
| GET | `/model-versions/{versionId}` | Metrics, eval report, recommendations, model card |
| POST | `/model-versions/{versionId}/predict` | Playground prediction (session auth; low rate limit) |
| POST | `/model-versions/{versionId}/exports` | Request an export (format, quantisation) |
| GET | `/exports/{exportId}` | Export status; presigned download URL when ready |
| POST | `/orgs/{orgId}/deployments` | Deploy a model version |
| GET/PATCH/DELETE | `/deployments/{deploymentId}` | Status; switch version / rollback; delete |
| GET | `/deployments/{deploymentId}/metrics` | Latency, errors, class distribution |
| POST | `/v1/predict/{deploymentId}` (inference host) | Hosted inference (API key) |
| POST | `/v1/batch-predictions` (inference host) | Async batch prediction job |
| POST | `/projects/{projectId}/face-galleries` | Create a face gallery (Phase 2) |
| POST | `/face-galleries/{galleryId}/consents` | Upload/attest a consent record |
| POST | `/face-galleries/{galleryId}/identities` | Enrol an identity (3–10 photos, consent ID required) |
| DELETE | `/face-galleries/{galleryId}/identities/{identityId}` | Remove an identity and all its biometric data |
| POST | `/v1/face/{galleryId}/identify` (inference host) | Identify a face → identity or `unknown` |
| POST | `/v1/face/{galleryId}/verify` (inference host) | 1:1 verification against a claimed identity |
| GET | `/orgs/{orgId}/billing` · POST `/orgs/{orgId}/billing/checkout` | Plan, invoices, upgrade checkout |
| POST | `/webhooks/razorpay` · `/webhooks/stripe` | Payment provider webhooks (signature-verified) |
| GET | `/orgs/{orgId}/audit-logs` | Audit trail (Admin+) |

---

## 12. Responsible AI and face recognition policy

### 12.1 General responsible-AI requirements
- **Model cards for every model** (§4.2), including intended use, out-of-scope use, data summary and limitations.
- **Prohibited uses (Acceptable Use Policy):** covert surveillance of the public, emotion recognition for employment or education decisions, inferring protected attributes (religion, caste, sexual orientation, health) from faces or voice, social scoring, and weapons targeting. Using a model as the sole basis for decisions with legal or similarly significant effects (credit, hiring, policing) is also prohibited without human review.
- **Misuse detection:**
  - The LLM task-spec step and the project templates flag high-risk descriptions for review.
  - Face features need a verified business account (Pro+) and an attested use case.
  - Projects can be reported, and accounts suspended.
- **Bias awareness:** the evaluation supports slice metrics (EVL-10). The UI warns when a slice underperforms the overall metric by more than 5 points.
- **Transparency:** exports embed the base-model licence and a "trained on customer data" notice. The platform never claims accuracy beyond the test results.

### 12.2 Why face recognition is not plain classification
A classifier trained on N people can only output one of those N people:
- **It cannot say "Unknown".** A stranger is forced into the closest class, often with high confidence, so it fails exactly when it matters (access control, attendance fraud).
- **Every new person needs retraining.** Adding or removing someone means a new training job, new GPU cost and a new model version. That is slow and expensive, and deletion requests become hard.
- **It needs many photos per person** to work at all, whereas metric-learning embeddings work with 3–10.

**Required design: detection → embedding → gallery matching.**
1. **Detect and align:** a RetinaFace- or MTCNN-class detector finds faces and landmarks, then aligns and crops to 112×112.
2. **Embed:** an ArcFace/FaceNet-style model maps each face to an L2-normalised 512-d vector. The model is pretrained on large face datasets and is **not** retrained per customer.
3. **Gallery:** enrolment stores 3–10 embeddings per identity (plus a mean template) in **pgvector** with an HNSW cosine index.
4. **Match:**
   - **Identify (1:N):** the nearest identity, accepted only if cosine similarity ≥ the gallery threshold; otherwise the result is **"Unknown"**.
   - **Verify (1:1):** compare against the claimed identity only.
5. **Tune:** users set the threshold from TAR/FAR curves on their own genuine/impostor pairs (FACE-5). The default is conservative (low FAR).
6. **Add/remove instantly:** galleries are just rows, so there is no retraining (FACE-3).

### 12.3 Consent and data protection
- Each enrolled person must have a **consent record** before enrolment (FACE-6):
  - The customer uploads a signed form or e-sign record, or attests a checkbox for low-risk jurisdictions where the customer's counsel permits it.
  - The record includes the purpose, retention period and withdrawal contact.
  - **BIPA-covered subjects need written releases.**
- **Withdrawal:** withdrawing consent automatically deletes that identity's biometric data (FACE-8).
- **Encryption:** face photos and embeddings are encrypted at rest with per-workspace keys. Access is limited to the inference path and authorised workspace roles, and every gallery access is audit-logged.
- **Retention:** every gallery has a mandatory retention date. Data is deleted automatically at expiry, after a reminder.
- **No secondary use:** biometric data is never used to improve platform models, never sold, and never shared across workspaces.
- **Deletion on request:** an API and UI deletion path with confirmation. Deletion completes within 24 h, and backups age out within the backup window.

### 12.4 Model licensing (all tasks, critical for face)
- A **model registry** lists every backbone with: source, licence, commercial-use flag, attribution text, weight hash and approval date.
- **Only commercially-cleared weights can be used for paid-tier models.** Non-commercial weights may be offered only in an explicitly labelled "research/education" mode on the Free tier, or excluded entirely (open question Q5).
- Known cautions to verify during registry review:
  - **InsightFace pretrained face models** (e.g., the buffalo/antelope packs) are licensed for **non-commercial research only**. Commercial use needs a licence from the vendor, or a different model trained on licence-clean data. The InsightFace *code* is permissively licensed; the *weights* are not.
  - **Face training datasets** behind public weights (e.g., MS-Celeb-1M, VGGFace2) have consent and licence issues of their own. Prefer vendors that document training-data provenance.
  - **Ultralytics YOLO** (v5/v8/11) is **AGPL-3.0**. Hosting it as a service or shipping it in exports creates copyleft obligations unless an enterprise licence is bought. Prefer Apache/MIT detectors (RT-DETR via transformers, YOLOX, torchvision) by default.
  - Some popular checkpoints carry **non-commercial licences**. Examples to verify: certain SegFormer weights (NVIDIA licence), LayoutLMv3 (CC BY-NC-SA) and some video model checkpoints. Others carry **use-restricted licences**, such as some LLM community licences.
- Licence text and attribution are embedded in every export bundle and model card.

### 12.5 Accuracy, fairness and testing requirements for face
- **Accuracy varies** with lighting, camera quality and angle, occlusion (masks, helmets, turbans, spectacles), image resolution, age (including ageing since enrolment) and skin tone. Published benchmarks do not guarantee performance on a customer's population or cameras.
- **Before deploying a face gallery, the customer must:**
  1. evaluate on **representative data** from their own cameras and conditions (a minimum of 50 identities or all enrolled identities, whichever is smaller, with genuine and impostor pairs)
  2. review the slice metrics (lighting, camera, and demographic tags where they lawfully have them)
  3. choose a threshold with the FAR/TAR trade-off shown
  4. confirm that a **human fallback** exists for rejections and errors (e.g., a manual attendance override)
  5. complete the deployment checklist, which is recorded in the audit log
- **Periodic re-check:** the dashboard prompts re-evaluation every 6 months, or when the camera hardware changes.
- **No 1:N identification against public or scraped images.** Galleries contain only consented, enrolled people.

---

## 13. Pricing tiers

Prices are **indicative** and to be validated with GPU cost data from the beta (§2.3 cost targets). INR pricing is shown for India; USD pricing applies elsewhere.

| | **Free** | **Pro** | **Team** | **Enterprise** |
|---|---|---|---|---|
| Indicative price | ₹0 | ~₹1,499/month (~$19) | ~₹7,999/month (~$99) + ₹999/seat beyond 5 | Custom (annual) |
| Users | 1 | 1 | 5 included, more seats extra | Unlimited, SSO/SCIM |
| GPU minutes / month | 60 (Quick budget only) | 600 + top-ups | 3,000 + top-ups (pooled) | Committed volume, reserved GPUs |
| Budget modes | Quick | Quick, Standard | Quick, Standard, Thorough | All + custom search, Ray Tune multi-GPU |
| Projects | 3 | 25 | Unlimited | Unlimited |
| Storage | 2 GB | 50 GB | 500 GB | Custom |
| Exports | ONNX, TF.js | All formats + quantisation | All | All |
| Hosted inference | 1 deployment, 1k calls/month, scale-to-zero (cold starts) | 3 deployments, 50k calls/month, always warm | 20 deployments, 500k calls/month, autoscaling | Dedicated/GPU pools, SLA 99.95% |
| Face recognition template | — | ✓ (verified business, consent tooling) | ✓ | ✓ + liveness, data residency options |
| MLflow access, API automation | Read-only run summary | ✓ | ✓ | ✓ |
| Data residency | India | India | India / EU | India / EU / custom |
| Support | Community | Email (48 h) | Priority (24 h) | Dedicated, SLA |
| Audit logs, RBAC | Basic | Basic | Full RBAC, audit logs | + export to SIEM |

Overage pricing: GPU minutes per GPU class (e.g., T4/L4 versus A10/A100 classes) and per 1k inference calls. Education discount for verified students and educators (Pro at a reduced price). Startup credits programme in Phase 2.

---

## 14. Roadmap

### 14.1 Phase 1: MVP (~4 months)
**Scope:**
- Category-first project creation, with the plain-English path as secondary
- **Camera capture**
- Image classification
- **Object detection**, with the box tool and AI-assisted box suggestions
- Text classification and sentiment (ready-made and custom, with Indic/Hinglish)
- Tabular classification and regression
- The full pipeline from intake through hosted CPU inference
- Free and Pro tiers

Adding detection and camera capture extends the original 3-month plan by about 3–4 weeks.

| Milestone | Weeks | Deliverables | Exit criteria |
|---|---|---|---|
| M0: Foundations | 1–2 | Repos, CI/CD, Terraform (VPC, RDS, Redis, S3, ECR), FastAPI skeleton, React shell, auth (email + Google), orgs/RBAC, OpenAPI → TS types | Sign in and create a workspace in staging |
| M1: Projects + data | 3–5 | Category cards + templates, LLM task spec (PRJ-2..5), multipart uploads, folder/zip/CSV ingestion, **camera capture with burst mode (DAT-9)**, dataset versioning, splits, dataset browser, classification labeling | 10k-image upload resumes after network loss; CSV mapping works; 50-photo burst capture from a phone uploads reliably on 4G |
| M2: Validation | 5–6 | VAL-1..14 checks, report UI, auto-fixes | Seeded-defect benchmark: > 95% detection |
| M3: Training (classification + tabular) | 6–9 | Model registry + weight mirror, rules-based selection, Optuna search, budgets, Celery GPU workers, checkpoints/retries, cancellation, WebSocket progress, MLflow, quota reservation/settlement | Benchmark suite jobs succeed at > 97%; Quick finishes in < 15 min median |
| M3b: Object detection | 9–12 | COCO/YOLO/VOC import (DAT-8), box tool (LBL-2), AI box suggestions (LBL-4a), detection validation (VAL-16), licence-cleared detector backbones (Apache/MIT), detection training + mAP evaluation, live-camera playground | 200-image safety-gear set: labeled in < 1 h with AI suggestions; mAP@0.5 within 3% of a hand-tuned baseline |
| M4: Evaluation + export | 11–14 | Metrics, confusion matrix, worst mistakes, playground, recommendations, model card, ONNX/TFLite/Core ML/TF.js + quantisation with verification (detection included), snippets | Exports verified on all Phase 1 tasks; snippets run in sample apps |
| M5: Deploy + billing + launch | 14–17 | Hosted CPU inference, API keys, rate limits, usage dashboard, Razorpay/Stripe, GST invoices, notifications, security review, load test, closed beta → public beta | Success metrics instrumented; p95 inference targets met; pen-test critical issues closed |

**Team assumption:** 2 backend/ML engineers, 1 ML engineer (training/export focus), 2 frontend engineers, 1 DevOps (part-time), 1 designer (part-time), product owner.

### 14.2 Benchmark suite (built in M2–M3, used continuously)
- About 15 public and licence-clean datasets across the Phase 1 tasks: small and large, balanced and imbalanced, English, Indic and Hinglish text.
- A hand-tuned baseline for each, to measure the "within 3%" quality target and catch regressions in CI (nightly).

### 14.3 Phase 2 (~months 5–8)
- **Face recognition template** (§6.12, §12) with consent tooling, liveness add-on, legal review completed
- Audio classification and keyword spotting (mic capture)
- NER and aspect-based sentiment
- Time series forecasting and anomaly detection
- AI-assisted pre-labeling for classification/text (LBL-4b); threshold tuning and calibration (EVL-4); model comparison
- Team tier: invites, RBAC, audit logs, pooled quotas; webhooks; batch prediction; endpoint monitoring dashboards
- Kubernetes GPU workers (EKS, spot instances, KEDA); EU region
- **Milestones:** P2-M1 face + audio beta (month 6); P2-M2 NER, aspect sentiment and Team tier (month 7); P2-M3 time series + K8s migration (month 8)

### 14.4 Phase 3 (~months 9–13)
- Segmentation (SAM-assisted labeling), pose/keypoints, OCR/document extraction
- Small LLM fine-tuning with LoRA/QLoRA (permissive-licence base models), with evaluation harnesses
- Video action recognition; sensor/IMU activity recognition
- Active learning loop from production predictions (LBL-5, DEP-9); drift detection
- Triton GPU serving, dedicated deployments; Go inference gateway if the §8.2 thresholds are reached
- Enterprise: SSO/SCIM, SOC 2 Type II audit, custom data residency, SLA 99.95%
- **Milestones:** P3-M1 segmentation + OCR (month 10); P3-M2 LoRA fine-tuning beta (month 11); P3-M3 Enterprise readiness (month 13)

---

## 15. Risks, mitigations and open questions

### 15.1 Risks and mitigations
| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | **GPU costs exceed revenue** (the free tier is abused; Thorough jobs run long) | High | High | Hard GPU-minute ceilings per budget; spot instances for Free/Quick; frozen-backbone-first strategy; caching backbone features for repeat runs; strict free-tier abuse controls (BIL-8); cost-per-job dashboards and alerts; plan pricing set from beta data |
| R2 | **Bad user data** (tiny, mislabelled, leaky) gives poor models and churn | High | High | The validation gate (§6.5); data-size guidance with expected accuracy bands; recommendations engine; label-noise detection; clear messaging that the test set is honest |
| R3 | **Misuse** (surveillance, discriminatory classifiers, deepfake-adjacent uses) | Medium | High | AUP; high-risk intent flagging in the LLM step; verified accounts for face features; abuse reporting; manual review queue; suspension policy |
| R4 | **Model licensing violations** (non-commercial or AGPL weights shipped to paying users) | Medium | High | Approved model registry with legal sign-off; the CI check blocks unregistered weights; licences embedded in exports; quarterly licence review |
| R5 | **Biometric compliance exposure** (BIPA class actions, DPDP/GDPR) | Medium | Very high | Consent-gated enrolment; retention and deletion automation; legal review before the Phase 2 launch; geo-restriction where needed; insurance; DPA templates |
| R6 | **Export fragility** (ops unsupported in TFLite/Core ML, accuracy drift after quantisation) | High | Medium | A maintained export matrix; automatic verification (EXP-3); fall back to formats that verify; architectures chosen for exportability when the target platform is edge |
| R7 | Quality falls short of "within 3% of hand-tuned" for some tasks | Medium | Medium | Benchmark suite in CI; expand search spaces per task; Thorough mode with ensembles; publish honest per-task benchmarks |
| R8 | **LLM task-spec errors** (wrong task type or labels) | Medium | Medium | Schema validation; a human review/edit step is mandatory before training; fallback to templates; an eval set of descriptions, with accuracy tracked |
| R9 | GPU capacity shortages / queue spikes | Medium | Medium | Multi-instance-type pools; priority queues by tier; ETA shown honestly; a second cloud or GPU provider as a fallback in Phase 2 |
| R10 | Data breach / tenant data leakage | Low | Very high | Tenant scoping tests; least-privilege IAM; encryption; pen tests; audit logs; incident response plan with DPDP/GDPR notification timelines |
| R11 | Scope creep across many task types delays MVP | High | Medium | Strict phase gates; the Phase 1 scope is frozen at M0 sign-off; templates added only via the registry and export-matrix process |

### 15.2 Open questions
| # | Question | Owner | Needed by |
|---|---|---|---|
| Q1 | Final product name and domain | Product | M0 |
| Q2 | Cloud choice: AWS (default assumption) versus GCP/Azure, versus a GPU cloud (e.g., Indian GPU providers) for training workers only | Eng + Finance | M0 |
| Q3 | Which LLM provider/model for task-spec generation, and the data-processing terms with that provider | Eng + Legal | M1 |
| Q4 | Launch face recognition in Illinois/Texas/Washington, or geo-restrict until legal review is done? | Legal | Phase 2 start |
| Q5 | Allow non-commercial weights on the Free tier for education ("research mode"), or exclude them entirely? | Product + Legal | M3 |
| Q6 | Which commercially licensed face embedding model to use: a vendor licence (e.g., commercial InsightFace) or a self-trained model on licence-clean data? | Eng + Legal | Phase 2 start |
| Q7 | Do we expose raw MLflow UI access, or only an in-app view? | Product | M3 |
| Q8 | Pricing validation: GPU-minute allowances and price points after beta cost data | Product + Finance | M5 |
| Q9 | Is on-prem/VPC deployment needed for the first Enterprise customers (it is a v1 non-goal)? | Sales | Phase 3 planning |
| Q10 | Minimum data-size thresholds per task (VAL-2) — confirm with the benchmark results | ML | M2 |
| Q11 | Grievance officer and DPDP notice requirements; registration as a significant data fiduciary if thresholds apply | Legal | Before paid launch |

---

## Appendix A: Task spec JSON schema (draft)

Example of an LLM-generated task spec, editable by the user before training:

```json
{
  "spec_version": "1.0",
  "task_type": "image_classification",
  "multi_label": false,
  "labels": ["healthy", "early_blight", "late_blight"],
  "input": { "modality": "image", "min_resolution": 224 },
  "primary_metric": "macro_f1",
  "secondary_metrics": ["accuracy", "per_class_recall"],
  "constraints": {
    "target_platforms": ["android"],
    "export_formats": ["tflite"],
    "max_model_size_mb": 15,
    "max_latency_ms": 50,
    "latency_device": "mid_range_android_cpu",
    "quantization": "int8_full"
  },
  "budget_mode": "standard",
  "split": { "train": 0.7, "val": 0.15, "test": 0.15, "stratify": true, "group_by": null, "seed": 42 },
  "language": null,
  "notes": "Photos taken in the field with phone cameras; varied lighting."
}
```

Schema rules:
- `task_type` must be in the enabled templates for the workspace's plan and phase.
- `labels` are required for classification and detection, with 2 to 500 entries.
- `constraints.max_model_size_mb` and `max_latency_ms` filter the candidate backbones in the registry. If none qualify, the UI says so before training.
- `primary_metric` must be valid for the `task_type`.
- Text tasks add `"language": ["hi", "en", "hi-Latn"]`, where `hi-Latn` means romanised Hindi/Hinglish.
- Tabular tasks add `"target_column"`, `"feature_columns"`, `"id_columns"` and `"group_column"`.

---

## Appendix B: Glossary

| Term | Meaning |
|---|---|
| Backbone | A pretrained neural network (e.g., ConvNeXt, DeBERTa) reused as the starting point for fine-tuning |
| Budget mode | A preset compute allowance (Quick/Standard/Thorough) that bounds the search size and GPU minutes |
| Dataset version | An immutable snapshot of samples, labels and splits; every model points to exactly one |
| Export matrix | The maintained table of which export formats and quantisations are supported and verified per task and architecture |
| Gallery | A set of enrolled identities and their face embeddings for 1:N face identification |
| Leakage | Information from the test set (or the target) present in training data or features, which inflates metrics |
| Model registry | The curated list of approved pretrained weights with licence metadata and pinned hashes |
| TAR / FAR | True Accept Rate / False Accept Rate, the verification trade-off controlled by the similarity threshold |
| Trial | One hyperparameter configuration trained and evaluated during search |
