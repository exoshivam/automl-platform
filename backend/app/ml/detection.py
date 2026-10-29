"""Object detection: torchvision SSDlite320-MobileNetV3 (BSD-3), COCO-pretrained detector, new class head.

Native layout for the loader / export: app.ml.detection.build(len(labels) + 1, pretrained=False)
(= COCO SSDlite structure with the classification head for N+1 classes; class 0 = background, labels[i] = i+1)
state_dict saved at native/model.pt. Inputs: RGB float [0, 1], any size (the model resizes to 320). Boxes are
predicted as xyxy pixels and stored as {x, y, w, h} pixels (artifacts contract).
"""

from __future__ import annotations

import io
import json
import time

from app.ml.common import Cancelled, Ctx, Result
from app.settings import get_settings

ARCH = "ssdlite320_mobilenet_v3_large"
LICENCE = {"registry_id": "torchvision/ssdlite320_mobilenet_v3_large.coco_v1", "source": "torchvision",
           "licence": "BSD-3-Clause (weights); trained on COCO (CC BY 4.0)", "commercial_use": True}
SCORE_KEEP = 0.05  # predictions kept for evaluation (mAP needs the low-score tail)
MIN_EPOCHS = 8  # mAP@0.5 saturates early on small sets while scores are still ~uniform: don't stop before this
SCORE_FLOOR = 0.5  # shown boxes must be more likely an object class than background + the other classes
NMS_IOU, TOP_K = 0.5, 20


def tidy(det, labels: list[str], floor: float = SCORE_FLOOR, top_k: int = TOP_K) -> list[dict]:
    """Shared predict-path postprocess (playground + box suggestions): class-wise NMS (a person on a chair keeps
    both boxes; same-class duplicates collapse), a confidence floor, top-k. torchvision's SSDlite defaults
    (score_thresh 0.001, 300 per image) would otherwise flood the image."""
    import torchvision

    keep = torchvision.ops.batched_nms(det["boxes"], det["scores"], det["labels"], NMS_IOU)
    out = []
    for i in keep.tolist():
        sc, lb = float(det["scores"][i]), int(det["labels"][i])
        if sc < floor or not 0 < lb <= len(labels):
            continue
        b = det["boxes"][i].tolist()
        out.append({"class_name": labels[lb - 1], "confidence": sc,
                    "bbox": {"x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1]}})
        if len(out) >= top_k:
            break
    return out


def configure_postprocess(model) -> None:
    model.score_thresh, model.nms_thresh, model.detections_per_img = SCORE_KEEP, 0.45, 100


def build(num_classes: int, pretrained: bool):
    """ONE fixed architecture for training, finalize, loader and export.

    torchvision's builder changes the MobileNetV3 tail width with the weights argument (reduced_tail =
    weights_backbone is None), so we always build the COCO SSDlite structure (weights=None, weights_backbone=None,
    91 classes), load the COCO detector weights when pretrained, then replace ONLY the classification head."""
    import functools

    from torch import nn
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    from torchvision.models.detection import _utils as det_utils
    from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    if pretrained:
        model.load_state_dict(SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.get_state_dict(progress=False))
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = functools.partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
    configure_postprocess(model)
    return model


def build_hybrid(labels: list[str], coco_ids: dict[str, int], pretrained: bool):
    """Hybrid detector = ONE model for chosen COCO classes + the user's custom classes. Starts from the COCO SSDlite
    (pretrained=True), keeps the COCO classification rows of the kept classes (and background), initialises the
    custom rows fresh, and freezes the backbone body so the shared features don't drift (anti-forgetting)."""
    model = build(len(labels) + 1, False)
    if not pretrained:
        return model
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

    coco = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.get_state_dict(progress=False)
    own = model.state_dict()
    rows = [0] + [coco_ids.get(c, -1) for c in labels]  # new class k <- COCO row rows[k] (-1 = new, keep fresh init)
    k_new, k_old = len(labels) + 1, 91
    for name, w in coco.items():
        if name not in own:
            continue
        if ".classification_head." in name and own[name].shape != w.shape and own[name].shape[0] % k_new == 0:
            anchors = own[name].shape[0] // k_new  # output channel = anchor * K + class
            new = own[name].clone()
            for a in range(anchors):
                for k, r in enumerate(rows):
                    if r >= 0:
                        new[a * k_new + k] = w[a * k_old + r]
            own[name] = new
        elif own[name].shape == w.shape:
            own[name] = w  # backbone, extras, box regression, depthwise parts of the class head
    model.load_state_dict(own)
    # Anti-forgetting, measured: freezing only the body still lost 0.11 COCO mAP@0.5 on replay. So the WHOLE network
    # is frozen except the custom class rows: backbone + extras + box head have no grad, BatchNorm stats stay
    # fixed (freeze_bn after every model.train()), and the COCO/background class rows are restored after each
    # optimizer step (AdamW's decoupled weight decay would otherwise shrink them). Custom objects are learned as new
    # classifier rows on the pretrained COCO features.
    import re

    final = re.compile(r"^head\.classification_head\.module_list\.\d+\.1\.(weight|bias)$")  # the 1x1 class conv
    for name, prm in model.named_parameters():
        prm.requires_grad = bool(final.match(name))
    custom_rows = [k for k, r in enumerate(rows) if r < 0]
    keep = {name: (own[name].clone(), [a * k_new + k for a in range(own[name].shape[0] // k_new)
                                       for k in range(k_new) if k not in custom_rows])
            for name, prm in model.named_parameters() if prm.requires_grad}

    def restore():
        import torch

        with torch.no_grad():
            for name, prm in model.named_parameters():
                if name in keep:
                    saved, idx = keep[name]
                    prm[idx] = saved[idx].to(prm.device)

    model.restore_frozen = restore
    return model


def batch_end(start: int, size: int, total: int) -> int:
    """End index of the batch starting at `start`. A final batch of ONE is merged into the previous one: BatchNorm
    in train mode raises on a 1-sample batch at the 1x1 extras feature map (job 01a169ec: 34 photos + replay)."""
    end = min(start + size, total)
    return total if total - end == 1 else end


def freeze_bn(model) -> None:
    """Hybrid: BatchNorm running stats must not drift towards the user's few photos."""
    from torch import nn

    for mod in model.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.eval()


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0.0, min(ax2, bx2) - max(a[0], b[0])), max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def map50(gts: dict, preds: dict, labels: list[str]) -> dict[str, float]:
    """VOC-style all-point AP at IoU 0.5 per class. gts/preds: {image_id: [(cls, (x,y,w,h), score?)]}."""
    aps = {}
    for cls in labels:
        gt = {i: [b for c, b in boxes if c == cls] for i, boxes in gts.items()}
        n_gt = sum(len(v) for v in gt.values())
        if n_gt == 0:
            continue
        dets = sorted(((s, i, b) for i, boxes in preds.items() for c, b, s in boxes if c == cls), reverse=True)
        used = {i: [False] * len(v) for i, v in gt.items()}
        tp, fp = [], []
        for _, i, b in dets:
            cands = gt.get(i, [])
            best, j = 0.0, -1
            for k, g in enumerate(cands):
                o = _iou(b, g)
                if o > best:
                    best, j = o, k
            if best >= 0.5 and not used[i][j]:
                used[i][j] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)
        ap, ctp, cfp, prev_r = 0.0, 0, 0, 0.0
        precisions, recalls = [], []
        for t, f in zip(tp, fp, strict=True):
            ctp, cfp = ctp + t, cfp + f
            precisions.append(ctp / (ctp + cfp))
            recalls.append(ctp / n_gt)
        for k in range(len(precisions) - 2, -1, -1):  # monotone precision envelope
            precisions[k] = max(precisions[k], precisions[k + 1])
        for p, r in zip(precisions, recalls, strict=True):
            ap += p * (r - prev_r)
            prev_r = r
        aps[cls] = ap
    return {"map_50": sum(aps.values()) / len(aps) if aps else 0.0, **{f"ap_50/{c}": v for c, v in aps.items()}}


def train(ctx: Ctx) -> Result:
    import optuna
    import torch
    import torchvision
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    from app.ml import memory
    from app.ml.device import free_memory

    s = get_settings()
    items = ctx.data["items"]  # [(sample_id, [(cls, {x,y,w,h})], split, no_objects)]
    fetch = ctx.data["fetch"]
    labels = ctx.labels
    cid = {c: i + 1 for i, c in enumerate(labels)}  # 0 = background
    usable = [it for it in items if it[1] or it[3]]
    by_split = {sp: [it for it in usable if it[2] == sp] for sp in ("train", "val", "test")}
    if ctx.budget["max_train"]:
        by_split["train"] = by_split["train"][: ctx.budget["max_train"]]
        by_split["val"] = by_split["val"][: max(ctx.budget["max_train"] // 4, 4)]
    if not by_split["train"]:
        raise ValueError("no labeled training images")
    val_set = by_split["val"] or by_split["train"]
    hybrid = ctx.data.get("hybrid")  # {"coco_ids": {label: coco_id}, "custom": [...], "replay": {...}} or None
    if hybrid:
        by_split, replay_eval, fetch, hybrid_info = _hybrid_prepare(ctx, by_split, fetch, hybrid)
        val_set = by_split["val"] or by_split["train"]

    def sample(it, train_mode: bool):
        img = Image.open(io.BytesIO(fetch(it[0]))).convert("RGB")
        x = to_tensor(img)
        boxes = [[b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]] for c, b in it[1] if c in cid and b["w"] > 0
                 and b["h"] > 0]
        cls = [cid[c] for c, b in it[1] if c in cid and b["w"] > 0 and b["h"] > 0]
        if train_mode and torch.rand(1).item() < 0.5:  # horizontal flip
            w = x.shape[-1]
            x = x.flip(-1)
            boxes = [[w - x2, y1, w - x1, y2] for x1, y1, x2, y2 in boxes]
        target = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.tensor(cls, dtype=torch.int64)}
        return x, target

    def predict(model, rows, bs):
        model.eval()
        out = {}
        with torch.no_grad():
            for k in range(0, len(rows), bs):
                chunk = rows[k:k + bs]
                xs = [to_tensor(Image.open(io.BytesIO(fetch(it[0]))).convert("RGB")).to(ctx.device) for it in chunk]
                for it, det in zip(chunk, model(xs), strict=True):
                    keep = det["scores"] >= SCORE_KEEP
                    out[it[0]] = [(labels[int(lb) - 1], (float(b[0]), float(b[1]), float(b[2] - b[0]),
                                                         float(b[3] - b[1])), float(sc))
                                  for b, lb, sc in zip(det["boxes"][keep].cpu(), det["labels"][keep].cpu(),
                                                       det["scores"][keep].cpu(), strict=True) if 0 < int(lb) <= len(labels)]
        return out

    def score(model, rows, bs):
        preds = predict(model, rows, bs)
        gts = {it[0]: [(c, (b["x"], b["y"], b["w"], b["h"])) for c, b in it[1]] for it in rows}
        return map50(gts, preds, labels)

    epochs, n_trials, patience = ctx.budget["epochs"], ctx.budget["trials"], ctx.budget.get("patience", 2)
    batch_size = min(s.train_batch_size, 4 if s.train_low_memory else 8)
    best = {"score": -1.0, "state": None, "params": {}, "metrics": {}}
    trials_log: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal batch_size
        # a fresh class head needs lr >= ~1e-3 to separate object vs background within a few epochs (measured on the
        # demo set: max score 0.42 -> 0.92 over 8 epochs at 1e-3; ~uniform 0.33 at 3.6e-4)
        # hybrid: the COCO rows start trained and the body is frozen -> a lower LR keeps them from drifting
        lr = trial.suggest_float("lr", 5e-4, 2e-3, log=True) if hybrid else trial.suggest_float("lr", 1e-3, 4e-3,
                                                                                                  log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        model = (build_hybrid(labels, hybrid["coco_ids"], s.train_pretrained) if hybrid
                 else build(len(labels) + 1, s.train_pretrained)).to(ctx.device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
        t0, trial_best, trial_state, trial_metrics, stale = time.monotonic(), -1.0, None, {}, 0
        try:
            for epoch in range(epochs):
                model.train()
                if hybrid or len(by_split["train"]) == 1:  # hybrid: stats stay COCO's; a lone image can't feed train-mode BN
                    freeze_bn(model)
                order = torch.randperm(len(by_split["train"])).tolist()
                loss_sum, n, k = 0.0, 0, 0
                while k < len(order):
                    end = batch_end(k, batch_size, len(order))
                    batch = [by_split["train"][i] for i in order[k:end]]
                    try:
                        if (k // batch_size) % 10 == 0:
                            ctx.check(f"trial {trial.number} epoch {epoch}")
                        xs, ts = zip(*(sample(it, True) for it in batch), strict=True)
                        losses = model([x.to(ctx.device) for x in xs],
                                       [{kk: v.to(ctx.device) for kk, v in t.items()} for t in ts])
                        loss = sum(losses.values())
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        if hasattr(model, "restore_frozen"):
                            model.restore_frozen()
                        loss_sum, n, k = loss_sum + float(loss) * len(batch), n + len(batch), end
                    except RuntimeError as e:
                        if not memory.is_oom(e) or batch_size <= 1:
                            raise memory.NotEnoughMemory(f"out of memory at batch size {batch_size}") from e
                        batch_size = max(batch_size // 2, 1)
                        free_memory(ctx.device)
                metrics = score(model, val_set, batch_size)
                val = metrics["map_50"]
                # ">=": mAP@0.5 saturates early on easy sets; on a tie the LATER epoch wins (better-separated scores)
                if val >= trial_best:
                    trial_best, trial_metrics, stale = val, metrics, 0
                    if val >= best["score"]:
                        trial_state = {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()}
                else:
                    stale += 1
                frac = (trial.number + (epoch + 1) / epochs) / n_trials
                ctx.publish({"phase": "searching", "trial": trial.number + 1, "trials_total": n_trials,
                             "epoch": epoch + 1, "epochs_total": epochs, "loss": round(loss_sum / max(n, 1), 4),
                             "metric_name": "map_50", "current_metric": round(val, 4),
                             "best_metric": round(max(val, best["score"]), 4), "eta_seconds": ctx.eta(frac),
                             "batch_size": batch_size})
                ctx.check(f"trial {trial.number} end of epoch {epoch}")
                trial.report(val, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                if stale >= patience and epoch + 1 >= min(MIN_EPOCHS, epochs):
                    break
            params = {"architecture": ARCH, "lr": lr, "weight_decay": wd, "epochs_run": epoch + 1,
                      "batch_size": batch_size}
            if trial_best >= best["score"] and trial_state is not None:
                best.update(score=trial_best, state=trial_state, params=params, metrics=trial_metrics)
            seconds = time.monotonic() - t0
            trials_log.append({"trial_no": trial.number, "params": params, "metrics": trial_metrics,
                               "state": "complete", "seconds": round(seconds, 1)})
            ctx.log_trial(trial.number, params, {**trial_metrics, "seconds": seconds}, "complete")
            return trial_best
        finally:
            del model, opt
            free_memory(ctx.device)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=ctx.spec["split"]["seed"]),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=1))
    cancelled = False
    try:
        study.optimize(objective, n_trials=n_trials)
    except Cancelled:
        if best["state"] is None:
            raise
        cancelled = True

    ctx.publish({"phase": "finalizing", "message": "Predicting on the test set"})
    model = build(len(labels) + 1, False)
    model.load_state_dict(best["state"])
    model.to(ctx.device)
    test = by_split["test"]
    preds = predict(model, test, batch_size) if test else {}
    rows = []
    for it in test:
        p = preds.get(it[0], [])
        rows.append({"sample_id": str(it[0]),
                     "y_true": json.dumps([{"class_name": c, "bbox": b} for c, b in it[1]]),
                     "y_pred": json.dumps([{"class_name": c, "bbox": {"x": b[0], "y": b[1], "w": b[2], "h": b[3]},
                                            "confidence": sc} for c, b, sc in p]),
                     "probs": None, "confidence": (sum(x[2] for x in p) / len(p)) if p else None})
    if hybrid:
        best["metrics"] = {**best["metrics"], **_hybrid_eval(ctx, model, predict, replay_eval, test, preds, hybrid,
                                                             hybrid_info, batch_size)}
    done = [t for t in trials_log if t["state"] == "complete"]
    if done:
        sps = sum(t["seconds"] for t in done) / max(sum(t["params"]["epochs_run"] for t in done)
                                                    * len(by_split["train"]), 1)
        ctx.data.setdefault("speed", {})[f"{ARCH}:{ctx.device}"] = sps
    buf = io.BytesIO()
    torch.save(best["state"], buf)
    del model
    free_memory(ctx.device)
    return Result(
        framework="torchvision", architecture=ARCH, base_model=LICENCE, labels=labels,
        input_spec={"modality": "image", "channels": 3, "color": "RGB", "layout": "list[CHW]", "dtype": "float32",
                    "scale": 255, "normalize": None, "size": [320, 320],
                    "note": "pass RGB images scaled to [0,1]; the model resizes to 320 internally"},
        output_spec={"type": "boxes", "format": "xywh_pixels", "score_threshold": SCORE_FLOOR, "classes": labels,
                     "background_class_index": 0, "nms_iou": NMS_IOU, "top_k": TOP_K,
                     **({"hybrid": {"ready_made": list(hybrid["coco_ids"]), "custom": hybrid["custom"],
                                    **{k.split("/", 1)[1]: v for k, v in best["metrics"].items()
                                       if k.startswith("hybrid/")}}} if hybrid else {})},
        val_metrics=best["metrics"], best_params=best["params"], trials=trials_log,
        native={"native/model.pt": buf.getvalue()}, predictions=rows, cancelled=cancelled,
        budget_name=ctx.budget["name"],
        library_versions={"torch": torch.__version__, "torchvision": torchvision.__version__,
                          "optuna": optuna.__version__},
    )


# ---- hybrid (ready-made COCO classes + custom classes in ONE model)

def keep_pseudo(found: list[dict], user_boxes: list[dict]) -> list[dict]:
    """Auto (ready-made) boxes that DON'T sit on something the user already boxed, whatever its class: a charger the
    user boxed must not also become an auto 'cell phone' (one object, two labels). Dropped when IoU >= 0.3 or >= 70%
    of the auto box lies inside a user box. Shared by training and the Label-page preview."""
    def xywh(b):
        return b["x"], b["y"], b["w"], b["h"]

    out = []
    for f in found:
        a = xywh(f["bbox"])
        hit = False
        for u in user_boxes:
            b = xywh(u)
            iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
            if _iou(a, b) >= 0.3 or (a[2] * a[3] > 0 and iw * ih / (a[2] * a[3]) >= 0.7):
                hit = True
                break
        if not hit:
            out.append(f)
    return out


DROP_WARN = 0.05  # ready-made mAP@0.5 may drop at most 5 points on the replay split before we warn
PSEUDO_SCORE = 0.5


def _hybrid_prepare(ctx: Ctx, by_split: dict, fetch, hybrid: dict):
    """1) pseudo-label the chosen COCO classes in every user photo with the stock COCO model (the user only boxed
    the custom objects; unboxed people/cups would otherwise be learned as background); 2) add COCO replay images."""
    from PIL import Image

    from app.ml import pretrained

    coco = list(hybrid["coco_ids"])
    added = 0
    for sp, rows in by_split.items():
        out = []
        for sid, boxes, split, no_obj in rows:
            found, _ = pretrained.predict(Image.open(io.BytesIO(fetch(sid))), coco, PSEUDO_SCORE)
            mine = [(c, b) for c, b in boxes]
            for f in keep_pseudo(found, [b for _, b in boxes]):
                mine.append((f["class_name"], f["bbox"]))
                added += 1
            out.append((sid, mine, split, no_obj))
        by_split[sp] = out
        ctx.check(f"pseudo-labels {sp}")
    pretrained.unload()  # free the stock model before training
    replay = hybrid.get("replay") or {}
    train_r, eval_r = replay.get("train", []), replay.get("eval", [])
    by_split["train"] = by_split["train"] + train_r

    def fetch2(sid):
        return replay["fetch"](sid) if isinstance(sid, str) and sid.startswith("replay:") else fetch(sid)

    info = {"pseudo_boxes": added, "replay_train": len(train_r), "replay_eval": len(eval_r),
            "pseudo_score": PSEUDO_SCORE, "stock_model": pretrained.weights_identity(),
            "replay_index": replay.get("index_key"), "replay_index_sha256": replay.get("index_sha256")}
    ctx.publish({"phase": "preparing", "message": f"Auto-labelled {added} ready-made objects; "
                 f"{len(train_r)} COCO replay images"})
    return by_split, eval_r, fetch2, info


def _hybrid_eval(ctx: Ctx, model, predict, replay_eval, test, preds, hybrid, info, bs) -> dict:
    """Custom mAP on the user's test split; ready-made mAP before (stock COCO) vs after on held-out replay images."""
    from PIL import Image

    from app.ml import pretrained

    custom, coco = hybrid["custom"], list(hybrid["coco_ids"])
    out = {f"hybrid/{k}": v for k, v in info.items()}
    gts = {it[0]: [(c, (b["x"], b["y"], b["w"], b["h"])) for c, b in it[1]] for it in test}
    out["hybrid/custom_map50"] = round(map50(gts, preds, custom)["map_50"], 4) if test and custom else None
    # Trust the fine-tuned model for the ready-made classes ONLY when they were seen during training (pseudo-labels
    # or replay) AND a held-out check shows they didn't get worse. Otherwise predict keeps the stock COCO model for
    # them (a second pass): ready-made objects must never silently vanish after adding a custom one.
    out["hybrid/ready_made_from"] = "stock"
    if not replay_eval:
        seen = info["pseudo_boxes"] + info["replay_train"]
        out["hybrid/warning"] = ("No COCO replay images" + ("" if seen else " and no ready-made objects in your photos")
                                 + ": the original ready-made detector keeps handling the ready-made classes")
        return out
    gts = {it[0]: [(c, (b["x"], b["y"], b["w"], b["h"])) for c, b in it[1]] for it in replay_eval}
    after = map50(gts, predict(model, replay_eval, bs), coco)["map_50"]
    stock = {}
    for it in replay_eval:  # the stock COCO model on the same images, low floor so mAP sees the whole PR curve
        found, _ = pretrained.predict(Image.open(io.BytesIO(hybrid["replay"]["fetch"](it[0]))), coco, SCORE_KEEP)
        stock[it[0]] = [(f["class_name"], (f["bbox"]["x"], f["bbox"]["y"], f["bbox"]["w"], f["bbox"]["h"]),
                         f["confidence"]) for f in found]
    pretrained.unload()
    before = map50(gts, stock, coco)["map_50"]
    out.update({"hybrid/coco_map50_before": round(before, 4), "hybrid/coco_map50_after": round(after, 4),
                "hybrid/coco_drop": round(before - after, 4)})
    if before - after > DROP_WARN:
        out["hybrid/warning"] = (f"Ready-made objects got worse after training: mAP@0.5 {before:.2f} -> {after:.2f} "
                                 "on COCO replay images; the original ready-made detector keeps handling them")
    else:
        out["hybrid/ready_made_from"] = "fine-tuned"  # one model, one pass
    return out
