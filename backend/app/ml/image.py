"""Image classification: timm MobileNetV3-Small (Quick) fine-tuned from pretrained weights, tiny Optuna search.

Lean by design (16 GB laptop): num_workers=0, small images, batch halving on OOM, memory freed between trials.
"""

from __future__ import annotations

import io
import time

from app.ml.common import Cancelled, Ctx, Result, classification_metrics, pick_metric
from app.settings import get_settings

ARCHS = {  # registry subset: all Apache-2.0 timm weights trained on ImageNet-1k
    "quick": ["mobilenetv3_small_100", "efficientnet_b0"],
    "low_memory": ["mobilenetv3_small_100"],
    "fast_test": ["mobilenetv3_small_100"],
    "standard": ["mobilenetv3_small_100", "mobilenetv3_large_100", "efficientnet_b0"],
    "thorough": ["mobilenetv3_large_100", "efficientnet_b0"],
}
LICENCE = {"source": "timm (Hugging Face hub)", "licence": "Apache-2.0", "commercial_use": True}


def train(ctx: Ctx) -> Result:
    import optuna
    import timm
    import torch
    from PIL import Image
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms as T

    from app.ml import memory
    from app.ml.device import free_memory

    s = get_settings()
    items = ctx.data["items"]  # [(sample_id, label, split)]
    fetch = ctx.data["fetch"]  # sample_id -> image bytes (cached)
    labels = ctx.labels
    idx = {c: i for i, c in enumerate(labels)}
    size = ctx.budget["image_size"]
    by_split = {sp: [(sid, lb) for sid, lb, split in items if split == sp] for sp in ("train", "val", "test")}
    if ctx.budget["max_train"]:
        by_split["train"] = by_split["train"][: ctx.budget["max_train"]]
        by_split["val"] = by_split["val"][: max(ctx.budget["max_train"] // 4, 8)]
    if not by_split["train"]:
        raise ValueError("no labeled training samples")
    val_set = by_split["val"] or by_split["train"]  # tiny datasets: fall back to train for model selection

    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    tf_train = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    tf_eval = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])

    class DS(Dataset):
        def __init__(self, rows, tf):
            self.rows, self.tf = rows, tf

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            sid, lb = self.rows[i]
            img = Image.open(io.BytesIO(fetch(sid))).convert("RGB")
            return self.tf(img), idx[lb]

    counts = torch.bincount(torch.tensor([idx[lb] for _, lb in by_split["train"]]), minlength=len(labels)).float()
    weights = (counts.sum() / (counts.clamp(min=1) * len(labels))) if counts.max() / counts.clamp(min=1).min() > 10 \
        else None  # TRN-13: class weights when VAL-4 would fire

    def logits_of(model, rows, bs):
        model.eval()
        out = []
        with torch.no_grad():
            for x, _ in DataLoader(DS(rows, tf_eval), batch_size=bs, num_workers=s.train_num_workers):
                out.append(model(x.to(ctx.device)).cpu())
        return torch.cat(out) if out else torch.empty(0, len(labels))

    def evaluate(model, rows, bs):
        return torch.softmax(logits_of(model, rows, bs), dim=1)

    archs = ARCHS[ctx.budget["name"]]
    epochs, n_trials = ctx.budget["epochs"], ctx.budget["trials"]
    primary = ctx.spec.get("primary_metric", "macro_f1")
    best = {"score": -1.0, "state": None, "arch": archs[0], "params": {}, "metrics": {}}
    trials_log: list[dict] = []
    batch_size = min(s.train_batch_size, ctx.budget.get("max_batch", s.train_batch_size))
    patience = ctx.budget.get("patience", 2)

    def objective(trial: optuna.Trial) -> float:
        nonlocal batch_size
        arch = trial.suggest_categorical("architecture", archs)
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        frozen_epochs = trial.suggest_int("frozen_epochs", 0, min(1, epochs - 1))
        model = timm.create_model(arch, pretrained=s.train_pretrained, num_classes=len(labels)).to(ctx.device)
        head = {id(p) for p in model.get_classifier().parameters()}
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.CrossEntropyLoss(weight=weights.to(ctx.device) if weights is not None else None,
                                      label_smoothing=0.1)  # curbs 100% confidence on look-alike burst frames
        score, t0 = -1.0, time.monotonic()
        trial_best, trial_state, trial_metrics, stale = -1.0, None, {}, 0
        try:
            for epoch in range(epochs):
                for p in model.parameters():
                    p.requires_grad = epoch >= frozen_epochs or id(p) in head
                model.train()
                loss_sum, n = 0.0, 0
                done = False
                while not done:  # retry the epoch with a smaller batch on OOM
                    try:
                        loader = DataLoader(DS(by_split["train"], tf_train), batch_size=batch_size, shuffle=True,
                                            num_workers=s.train_num_workers, drop_last=len(by_split["train"]) > batch_size)
                        for b, (x, y) in enumerate(loader):
                            if b % 10 == 0:
                                ctx.check(f"trial {trial.number} epoch {epoch}")
                            opt.zero_grad()
                            loss = loss_fn(model(x.to(ctx.device)), y.to(ctx.device))
                            loss.backward()
                            opt.step()
                            loss_sum, n = loss_sum + float(loss) * len(y), n + len(y)
                        done = True
                    except RuntimeError as e:
                        if not memory.is_oom(e) or batch_size <= 2:
                            raise memory.NotEnoughMemory(f"out of memory at batch size {batch_size}") from e
                        batch_size //= 2
                        free_memory(ctx.device)
                probs = evaluate(model, val_set, batch_size)
                pred = [labels[i] for i in probs.argmax(1).tolist()]
                metrics = classification_metrics([lb for _, lb in val_set], pred, labels)
                name, score = pick_metric(metrics, primary)
                frac = (trial.number + (epoch + 1) / epochs) / n_trials
                ctx.publish({"phase": "searching", "trial": trial.number + 1, "trials_total": n_trials,
                             "epoch": epoch + 1, "epochs_total": epochs, "loss": round(loss_sum / max(n, 1), 4),
                             "metric_name": name, "current_metric": round(score, 4),
                             "best_metric": round(max(score, best["score"]), 4), "eta_seconds": ctx.eta(frac),
                             "batch_size": batch_size})
                ctx.check(f"trial {trial.number} end of epoch {epoch}")
                if score >= trial_best:  # keep this trial's best epoch; on a tie the LATER (better-calibrated) wins
                    trial_best, trial_metrics, stale = score, metrics, 0
                    if score >= best["score"]:
                        trial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                trial.report(score, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if stale >= patience:
                    break
            score, metrics = trial_best, trial_metrics
            params = {"architecture": arch, "lr": lr, "weight_decay": wd, "frozen_epochs": frozen_epochs,
                      "epochs_run": epoch + 1, "image_size": size, "batch_size": batch_size}
            if score > best["score"] and trial_state is not None:
                best.update(score=score, arch=arch, params=params, metrics=metrics, state=trial_state)
            trials_log.append({"trial_no": trial.number, "params": params, "metrics": metrics, "state": "complete",
                               "seconds": round(time.monotonic() - t0, 1)})
            ctx.log_trial(trial.number, params, {**metrics, "seconds": time.monotonic() - t0}, "complete")
            return score
        except optuna.TrialPruned:
            ctx.log_trial(trial.number, {"architecture": arch, "lr": lr}, {"val_metric": score}, "pruned")
            raise
        finally:
            del model, opt
            free_memory(ctx.device)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=ctx.spec["split"]["seed"]),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=1))
    cancelled = False
    try:
        study.optimize(objective, n_trials=n_trials, catch=())
    except Cancelled:
        if best["state"] is None:
            raise
        cancelled = True  # TRN-8: keep the best finished trial

    ctx.publish({"phase": "finalizing", "message": "Predicting on the test set"})
    done = [t for t in trials_log if t["state"] == "complete"]
    if done:  # seconds per training sample-epoch on this device/arch, for the next job's estimate
        sps = sum(t["seconds"] for t in done) / max(sum(t["params"]["epochs_run"] for t in done) * len(by_split["train"]), 1)
        ctx.data.setdefault("speed", {})[f"{best['arch']}:{ctx.device}"] = sps
    model = timm.create_model(best["arch"], pretrained=False, num_classes=len(labels))
    model.load_state_dict(best["state"])
    model.to(ctx.device)
    from app.ml.common import calibrate, softmax_t

    # EVL-4: probabilities that mean what they say
    temperature, calibration = calibrate(logits_of(model, val_set, batch_size).numpy(), [idx[lb] for _, lb in val_set])
    test = by_split["test"]
    probs = torch.from_numpy(softmax_t(logits_of(model, test, batch_size).numpy(), temperature)).float() if test \
        else torch.empty(0, len(labels))
    rows = []
    for (sid, lb), p in zip(test, probs.tolist(), strict=True):
        k = max(range(len(labels)), key=lambda i: p[i])
        rows.append({"sample_id": str(sid), "y_true": lb, "y_pred": labels[k], "probs": p, "confidence": p[k]})
    buf = io.BytesIO()
    torch.save(best["state"], buf)
    del model
    free_memory(ctx.device)
    return Result(
        framework="timm", architecture=best["arch"],
        base_model={"registry_id": f"timm/{best['arch']}", **LICENCE},
        labels=labels,
        input_spec={"modality": "image", "size": [size, size], "channels": 3, "color": "RGB", "resize": "bilinear",
                    "normalize": {"mean": list(mean), "std": list(std)}, "layout": "NCHW", "dtype": "float32",
                    "scale": 255},
        output_spec={"type": "probabilities", "activation": "softmax", "classes": labels,
                     "temperature": temperature, "calibration": calibration},
        val_metrics={**best["metrics"], **calibration}, best_params=best["params"], trials=trials_log,
        native={"native/model.pt": buf.getvalue()}, predictions=rows, cancelled=cancelled, budget_name=ctx.budget["name"],
        library_versions={"torch": torch.__version__, "timm": timm.__version__, "optuna": optuna.__version__},
    )
