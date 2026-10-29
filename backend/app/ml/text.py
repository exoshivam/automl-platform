"""Text classification: a small multilingual encoder fine-tuned end to end, tiny Optuna search.

paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0): 50+ languages incl. Hindi and other Indic scripts, copes with
Hinglish; 118M params, fine on a 16 GB laptop (MPS or CPU). Weights come from the local Hugging Face cache only.

Also holds the pieces train_tasks / loader plug in: load_data() (Ctx.data) and TextModel (loader.register).
"""

from __future__ import annotations

import os
import tempfile
import time
from collections import Counter
from typing import Any

from app.ml.common import Cancelled, Ctx, Result, calibrate, classification_metrics, pick_metric, softmax_t

CHECKPOINT = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LICENCE = {"registry_id": f"hf/{CHECKPOINT}", "source": "Hugging Face hub", "licence": "Apache-2.0",
           "commercial_use": True}
MAX_LEN = 128
BATCH = 16
# Text fine-tuning needs a few epochs even on tiny sets, so floors apply on top of the shared BUDGETS.
MIN_TRIALS, MIN_EPOCHS = 2, 3


def load_data(db, v, labels: list[str]) -> dict[str, Any]:
    """Ctx.data for text: items [(sample_id, label, split)] + texts {sample_id: str}."""
    from sqlalchemy import select

    from app import models as m

    org = v.organization_id
    samples = db.scalars(select(m.Sample).where(m.Sample.dataset_version_id == v.id, m.Sample.organization_id == org)).all()
    first: dict = {}
    for sid, cls in db.execute(select(m.Label.sample_id, m.Label.class_name).join(m.Sample, m.Sample.id == m.Label.sample_id)
                               .where(m.Sample.dataset_version_id == v.id, m.Label.organization_id == org)
                               .order_by(m.Label.created_at)).all():
        first.setdefault(sid, cls)
    texts = {s.id: (s.meta or {}).get("text") or "" for s in samples}
    items = [(s.id, first[s.id], s.split) for s in sorted(samples, key=lambda s: str(s.id))
             if s.id in first and s.split and first[s.id] in labels and texts[s.id].strip()]
    return {"items": items, "texts": texts}


def _hf_load(num_labels: int | None = None, model: bool = True):
    """Tokenizer (+ model) from the local cache (no network in the sandbox / offline laptops)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
        m = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=num_labels, local_files_only=True) \
            if model else None
    except OSError as e:  # surfaces as error_code dependency_missing
        raise ImportError(f"{CHECKPOINT} isn't in the Hugging Face cache; pre-download it ({e})") from e
    return tok, m


def train(ctx: Ctx) -> Result:
    import optuna
    import torch
    import transformers

    from app.ml import memory
    from app.ml.device import free_memory

    labels, idx = ctx.labels, {c: i for i, c in enumerate(ctx.labels)}
    texts = ctx.data["texts"]
    by_split = {sp: [(sid, lb) for sid, lb, split in ctx.data["items"] if split == sp] for sp in ("train", "val", "test")}
    if ctx.budget["max_train"]:
        by_split["train"] = by_split["train"][: ctx.budget["max_train"]]
    if not by_split["train"]:
        raise ValueError("no labeled training samples")
    val_set = by_split["val"] or by_split["train"]
    n_trials = max(ctx.budget["trials"], MIN_TRIALS)
    epochs, patience = max(ctx.budget["epochs"], MIN_EPOCHS), ctx.budget.get("patience", 2)
    primary = ctx.spec.get("primary_metric", "macro_f1")
    seed = int((ctx.spec.get("split") or {}).get("seed", 0))
    tok, _ = _hf_load(model=False)  # tokenizer only; fails fast if the checkpoint is missing

    counts = Counter(lb for _, lb in by_split["train"])
    weights = None
    if len(counts) > 1 and max(counts.values()) / max(min(counts.get(c, 0) for c in labels), 1) > 10:  # TRN-13
        total = sum(counts.values())
        weights = torch.tensor([total / (len(labels) * max(counts.get(c, 0), 1)) for c in labels], dtype=torch.float)

    def batches(rows, shuffle=False, seed=0):
        order = list(range(len(rows)))
        if shuffle:
            order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
        for i in range(0, len(order), BATCH):
            chunk = [rows[j] for j in order[i:i + BATCH]]
            enc = tok([texts[s] for s, _ in chunk], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            yield {k: v.to(ctx.device) for k, v in enc.items()}, torch.tensor([idx[lb] for _, lb in chunk], device=ctx.device)

    def logits_of(model, rows):
        model.eval()
        out = []
        with torch.no_grad():
            for x, _ in batches(rows):
                out.append(model(**x).logits.float().cpu())
        return torch.cat(out) if out else torch.empty(0, len(labels))

    best: dict[str, Any] = {"score": -1.0, "state": None, "params": {}, "metrics": {}}
    trials_log: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        wd = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        # Head init + dropout use torch's global RNG: seed per trial so a run doesn't depend on what ran before.
        torch.manual_seed(seed + trial.number)
        _, model = _hf_load(len(labels))
        # The 250k-token multilingual embedding matrix is ~80% of the weights: freezing it saves its gradients and
        # Adam state (~1 GB) and suits small datasets, which can't retrain a vocabulary anyway.
        model.get_input_embeddings().weight.requires_grad_(False)
        model.to(ctx.device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        steps = epochs * max(1, -(-len(by_split["train"]) // BATCH))
        sched = transformers.get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(ctx.device) if weights is not None else None,
                                            label_smoothing=0.1)
        t0, score, metrics = time.monotonic(), -1.0, {}
        trial_best, stale, ran = -1.0, 0, 0
        try:
            for epoch in range(epochs):
                model.train()
                loss_sum, n = 0.0, 0
                for b, (x, y) in enumerate(batches(by_split["train"], shuffle=True, seed=trial.number * 100 + epoch)):
                    if b % 10 == 0:
                        ctx.check(f"trial {trial.number} epoch {epoch}")
                    opt.zero_grad()
                    loss = loss_fn(model(**x).logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    loss_sum, n = loss_sum + float(loss) * len(y), n + len(y)
                probs = logits_of(model, val_set)  # argmax only: same as softmax
                pred = [labels[i] for i in probs.argmax(1).tolist()]
                metrics = classification_metrics([lb for _, lb in val_set], pred, labels)
                mname, score = pick_metric(metrics, primary)
                frac = (trial.number + (epoch + 1) / epochs) / n_trials
                ctx.publish({"phase": "searching", "trial": trial.number + 1, "trials_total": n_trials,
                             "epoch": epoch + 1, "epochs_total": epochs, "loss": round(loss_sum / max(n, 1), 4),
                             "metric_name": mname, "current_metric": round(score, 4),
                             "best_metric": round(max(score, best["score"]), 4), "eta_seconds": ctx.eta(frac),
                             "batch_size": BATCH})
                ran = epoch + 1
                if score > best["score"]:  # keep the best epoch's weights, not the last
                    best.update(score=score, metrics=metrics,
                                state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                                params={"lr": lr, "weight_decay": wd, "epochs": ran})
                if score > trial_best:
                    trial_best, stale = score, 0
                else:
                    stale += 1
                    if stale >= patience:  # early stopping
                        break
        except RuntimeError as e:
            if memory.is_oom(e):
                raise memory.NotEnoughMemory(f"out of memory at batch size {BATCH}") from e
            raise
        params = {"lr": lr, "weight_decay": wd, "epochs": ran, "batch_size": BATCH, "max_length": MAX_LEN,
                  "checkpoint": CHECKPOINT}
        metrics = {**metrics, "seconds": round(time.monotonic() - t0, 1)}
        trials_log.append({"trial_no": trial.number, "params": params, "metrics": metrics, "state": "complete"})
        ctx.log_trial(trial.number, params, metrics, "complete")
        del model, opt, sched  # the scheduler holds the optimizer (and its Adam state) otherwise
        free_memory(ctx.device)
        return trial_best

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    cancelled = False
    try:
        study.optimize(objective, n_trials=n_trials)
    except Cancelled:
        if best["state"] is None:
            raise
        cancelled = True

    ctx.publish({"phase": "finalizing", "message": "Predicting on the test set"})
    tok, model = _hf_load(len(labels))
    model.load_state_dict(best["state"])
    model.config.id2label = dict(enumerate(labels))
    model.config.label2id = idx
    model.to(ctx.device)
    test = by_split["test"]
    # EVL-4: fit a temperature on val logits (kept only if it lowers val ECE), then calibrated test probs.
    temperature, calibration = calibrate(logits_of(model, val_set).numpy(), [idx[lb] for _, lb in val_set])
    probs = torch.from_numpy(softmax_t(logits_of(model, test).numpy(), temperature)) if test \
        else torch.empty(0, len(labels))
    rows = []
    for (sid, lb), p in zip(test, probs.tolist(), strict=True):
        k = max(range(len(labels)), key=lambda i: p[i])
        rows.append({"sample_id": str(sid), "y_true": lb, "y_pred": labels[k], "probs": p, "confidence": p[k]})
    native: dict[str, bytes] = {}
    with tempfile.TemporaryDirectory(prefix="automl-text-") as d:
        model.to("cpu").save_pretrained(d, safe_serialization=True)
        tok.save_pretrained(d)
        for f in sorted(os.listdir(d)):
            with open(os.path.join(d, f), "rb") as fh:
                native[f"native/hf/{f}"] = fh.read()
    del model
    free_memory(ctx.device)
    return Result(
        framework="transformers", architecture=CHECKPOINT, base_model=LICENCE, labels=labels,
        input_spec={"modality": "text", "tokenizer": "native/hf/", "max_length": MAX_LEN, "lowercase": False},
        output_spec={"type": "probabilities", "activation": "softmax", "classes": labels, "temperature": temperature,
                     "calibration": calibration},
        val_metrics=best["metrics"],
        best_params={**best["params"], "batch_size": BATCH, "max_length": MAX_LEN, "checkpoint": CHECKPOINT}, trials=trials_log, native=native, predictions=rows,
        cancelled=cancelled, budget_name=ctx.budget["name"],
        library_versions={"torch": torch.__version__, "transformers": transformers.__version__, "optuna": optuna.__version__},
    )


def _loaded_base():
    from app.ml.loader import LoadedModel

    return LoadedModel


class TextModel(_loaded_base()):  # type: ignore[misc]
    """Inference for loader.load_model (CPU). Downloads native/hf/ once into a temp dir held for the cache entry."""

    def __init__(self, meta, prefix):
        super().__init__(meta, prefix)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from app.services import storage

        self.torch = torch
        self._dir = tempfile.TemporaryDirectory(prefix="automl-text-model-")
        hf = prefix + meta["input"].get("tokenizer", "native/hf/")
        for key in storage.existing_sizes(hf):
            with open(os.path.join(self._dir.name, key[len(hf):]), "wb") as f:
                f.write(storage.get_bytes(key))
        self.tok = AutoTokenizer.from_pretrained(self._dir.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self._dir.name).eval()
        self.max_len = meta["input"].get("max_length", MAX_LEN)

    def predict(self, batch):
        if not all(isinstance(x, str) for x in batch):
            raise TypeError("text models take strings")
        enc = self.tok(list(batch), padding=True, truncation=True, max_length=self.max_len, return_tensors="pt")
        with self.torch.no_grad():
            logits = self.model(**enc).logits.numpy()
        probs = softmax_t(logits, self.temperature).tolist()
        return self._classify(probs)


def _register() -> None:
    from app.ml import loader

    loader.register("transformers", TextModel)


_register()
