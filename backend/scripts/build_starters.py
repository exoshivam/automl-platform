"""DAT-13: builds the starter dataset library from automl-platform/.starter-cache into S3 starters/ + catalog.json.

    uv run python scripts/build_starters.py            # every dataset whose raw files are present
    uv run python scripts/build_starters.py beans iris # only these (the rest of the catalog is kept)

Subsamples for laptop training: <= 1,000 images (stratified), <= 3,000 text rows, <= 5,000 table rows. Photos are
re-encoded as JPEG with the long side <= 640 px (training runs at 112-224 px). Licences are recorded, informational.
"""

import io
import json
import random
import re
import sys
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services import starters, storage

CACHE = Path(__file__).resolve().parents[2] / ".starter-cache"
MAX_IMAGES, MAX_TEXT, MAX_ROWS, MAX_SIDE = 1000, 3000, 5000, 640
SEED = 13


def jpeg(data: bytes) -> tuple[bytes, float]:
    """Returns (jpeg bytes, scale applied) so box coordinates can follow."""
    img = Image.open(io.BytesIO(data)).convert("RGB")
    scale = min(1.0, MAX_SIDE / max(img.size))
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue(), scale


def stratified(items: list, key, n: int) -> list:
    """Round-robin over classes so every class keeps (nearly) the same share; deterministic."""
    rnd, groups = random.Random(SEED), defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    for g in groups.values():
        rnd.shuffle(g)
    if sum(map(len, groups.values())) <= n:
        return [it for g in groups.values() for it in g]
    share = {k: round(n * len(g) / len(items)) for k, g in groups.items()}  # proportional, at least 1
    return [it for k, g in sorted(groups.items()) for it in g[:max(1, share[k])]][:n]


def balanced(items: list, key, n: int) -> list:
    """Equal per-class counts (image sets: the classes are roughly balanced anyway)."""
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(it)
    rnd = random.Random(SEED)
    for g in groups.values():
        rnd.shuffle(g)
    per = n // len(groups)
    return [it for _, g in sorted(groups.items()) for it in g[:per]]


def image_entry(meta: dict, items: list[tuple[str, bytes]]) -> dict:
    """items: [(label, raw bytes)] -> folder layout <label>/<n>.jpg."""
    picked = balanced(items, lambda x: x[0], MAX_IMAGES)
    files = [(f"{lb}/{i:04d}.jpg", jpeg(data)[0], lb) for i, (lb, data) in enumerate(picked)]
    return starters.publish_files({**meta, "task_type": "image_classification",
                                   "classes": sorted({lb for lb, _ in picked})}, files)


# ---------------------------------------------------------------- image classification

def beans():
    names = ["angular_leaf_spot", "bean_rust", "healthy"]
    df = pd.concat(pd.read_parquet(p) for p in sorted((CACHE / "beans" / "data").glob("*.parquet")))
    items = [(names[int(r["labels"])], r["image"]["bytes"]) for _, r in df.iterrows()]
    return image_entry({"id": "beans", "name": "Bean leaf diseases",
                        "description": "Photos of bean plant leaves from Uganda: healthy, angular leaf spot or bean "
                                       "rust. Field photos taken with phones.",
                        "source": "Makerere AI Lab (Hugging Face)", "licence": "MIT",
                        "source_url": "https://huggingface.co/datasets/AI-Lab-Makerere/beans"}, items)


def tf_flowers():
    items = []
    with tarfile.open(CACHE / "tf_flowers" / "flower_photos.tgz") as tar:
        for mem in tar:
            parts = mem.name.split("/")
            if mem.isfile() and len(parts) == 3 and parts[2].lower().endswith(".jpg"):
                items.append((parts[1], tar.extractfile(mem).read()))
    return image_entry({"id": "tf_flowers", "name": "Flowers (5 kinds)",
                        "description": "Daisies, dandelions, roses, sunflowers and tulips from Flickr.",
                        "source": "TensorFlow datasets (flower_photos)", "licence": "CC BY 2.0 (per photo)",
                        "source_url": "https://www.tensorflow.org/datasets/catalog/tf_flowers"}, items)


def oxford_pets():
    d = CACHE / "oxford_pets"
    species = {}
    if (d / "annotations.tar.gz").exists():
        with tarfile.open(d / "annotations.tar.gz") as tar:
            lst = tar.extractfile("annotations/list.txt").read().decode()
        for line in lst.splitlines():
            if line and not line.startswith("#"):
                name, _cls, sp, _breed = line.split()
                species[name] = "cat" if sp == "1" else "dog"
    items = []
    with tarfile.open(d / "images.tar.gz") as tar:
        for mem in tar:
            stem = Path(mem.name).stem
            if mem.isfile() and mem.name.endswith(".jpg"):
                lb = species.get(stem) or ("cat" if stem[0].isupper() else "dog")
                items.append((lb, tar.extractfile(mem).read()))
    return image_entry({"id": "oxford_pets", "name": "Cats vs dogs (Oxford Pets)",
                        "description": "Cats and dogs from 37 breeds, photographed at home and outdoors.",
                        "source": "Oxford-IIIT Pet Dataset (VGG, Oxford)", "licence": "CC BY-SA 4.0",
                        "source_url": "https://www.robots.ox.ac.uk/~vgg/data/pets/"}, items)


# ---------------------------------------------------------------- object detection

def det_entry(meta: dict, images: list[tuple[str, bytes, list[tuple[str, list[float]]]]]) -> dict:
    """images: [(name, raw bytes, [(class, [x, y, w, h] px)])] -> images/<name>.jpg + COCO json."""
    classes = sorted({c for _, _, bs in images for c, _ in bs})
    cat = {c: i + 1 for i, c in enumerate(classes)}
    files, coco = [], {"images": [], "annotations": [], "categories": [{"id": i, "name": c} for c, i in cat.items()]}
    for i, (name, raw, boxes) in enumerate(images[:MAX_IMAGES]):
        data, s = jpeg(raw)
        w, h = Image.open(io.BytesIO(data)).size
        fname = f"{Path(name).stem}.jpg"
        files.append((f"images/{fname}", data, None))
        coco["images"].append({"id": i + 1, "file_name": fname, "width": w, "height": h})
        for c, (x, y, bw, bh) in boxes:
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": i + 1, "category_id": cat[c],
                                        "bbox": [round(x * s, 1), round(y * s, 1), round(bw * s, 1),
                                                 round(bh * s, 1)]})
    return starters.publish_files({**meta, "task_type": "object_detection", "classes": classes}, files,
                                  json.dumps(coco).encode())


def penn_fudan():
    box = re.compile(r'Bounding box for object \d+ "(\w+)" \(Xmin, Ymin\) - \(Xmax, Ymax\) : '
                     r"\((\d+), (\d+)\) - \((\d+), (\d+)\)")
    images = []
    with zipfile.ZipFile(CACHE / "penn_fudan" / "PennFudanPed.zip") as z:
        for n in sorted(z.namelist()):
            if "/Annotation/" in n and n.endswith(".txt"):
                txt = z.read(n).decode("latin-1")
                png = n.replace("/Annotation/", "/PNGImages/").replace(".txt", ".png")
                boxes = [("person", [float(a), float(b), float(c) - float(a), float(d) - float(b)])
                         for _, a, b, c, d in box.findall(txt)]
                images.append((Path(png).name, z.read(png), boxes))
    return det_entry({"id": "penn_fudan", "name": "Pedestrians (Penn-Fudan)",
                      "description": "Street photos from Penn and Fudan universities with a box around every "
                                     "pedestrian.",
                      "source": "Penn-Fudan Database for Pedestrian Detection", "licence": "Research use (see source)",
                      "source_url": "https://www.cis.upenn.edu/~jshi/ped_html/"}, images)


def coco_subset():
    d = CACHE / "coco_val2017_subset"
    with zipfile.ZipFile(d / "annotations_trainval2017.zip") as z:
        doc = json.loads(z.read("annotations/instances_val2017.json"))
    want = {c["id"]: c["name"] for c in doc["categories"] if c["name"] in ("person", "car", "dog")}
    files = {p.name for p in (d / "val2017").glob("*.jpg")}
    by_img = defaultdict(list)
    for a in doc["annotations"]:
        if a["category_id"] in want and not a.get("iscrowd"):
            by_img[a["image_id"]].append((want[a["category_id"]], a["bbox"]))
    images = [(im["file_name"], (d / "val2017" / im["file_name"]).read_bytes(), by_img[im["id"]])
              for im in sorted(doc["images"], key=lambda x: x["id"]) if im["file_name"] in files and by_img[im["id"]]]
    return det_entry({"id": "coco_person_car_dog", "name": "People, cars and dogs (COCO subset)",
                      "description": "Everyday scenes from COCO val2017 with boxes for people, cars and dogs.",
                      "source": "COCO 2017 (val)", "licence": "Annotations CC BY 4.0; images Flickr terms",
                      "source_url": "https://cocodataset.org/"}, images)


# ---------------------------------------------------------------- text

def text_entry(meta: dict, df: pd.DataFrame) -> dict:
    df = df.dropna()
    df = df[df["text"].str.strip().astype(bool)]
    rows = stratified(list(df.itertuples(index=False)), lambda r: r.label, MAX_TEXT)
    out = pd.DataFrame(rows, columns=["text", "label"])
    cols = [{"name": "text", "role": "text", "type": "text"}, {"name": "label", "role": "target", "type": "categorical"}]
    return starters.publish_table({**meta, "task_type": "text_classification",
                                   "classes": sorted(out["label"].unique().tolist()), "target": "label"},
                                  out.to_csv(index=False).encode(), cols, len(out))


def imdb():
    df = pd.read_parquet(CACHE / "imdb" / "plain_text" / "train-00000-of-00001.parquet")
    df["label"] = df["label"].map({0: "negative", 1: "positive"})
    return text_entry({"id": "imdb", "name": "Movie review sentiment (IMDB)",
                       "description": "English movie reviews labelled positive or negative.",
                       "source": "Stanford Large Movie Review Dataset (HF stanfordnlp/imdb)",
                       "licence": "Research use (see source)",
                       "source_url": "https://huggingface.co/datasets/stanfordnlp/imdb"}, df[["text", "label"]])


def sms_spam():
    with zipfile.ZipFile(CACHE / "sms_spam" / "sms_spam_collection.zip") as z:
        raw = z.read("SMSSpamCollection").decode("utf-8", "replace")
    rows = [line.split("\t", 1)[::-1] for line in raw.splitlines() if "\t" in line]
    return text_entry({"id": "sms_spam", "name": "SMS spam",
                       "description": "Text messages labelled spam or ham (not spam).",
                       "source": "UCI SMS Spam Collection", "licence": "CC BY 4.0",
                       "source_url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"},
                      pd.DataFrame(rows, columns=["text", "label"]))


def sentimix_hinglish():
    d = CACHE / "sentimix_hinglish"
    rows, words, label = [], [], None
    for f in ("train_14k_split_conll.txt", "dev_3k_split_conll.txt"):
        for line in (d / f).read_text(encoding="utf-8", errors="replace").splitlines() + [""]:
            parts = line.split("\t")
            if parts[0] == "meta":
                if words and label:
                    rows.append((" ".join(words), label))
                words, label = [], parts[2].strip() if len(parts) > 2 else None
            elif line.strip():
                words.append(parts[0])
        if words and label:
            rows.append((" ".join(words), label))
        words, label = [], None
    df = pd.DataFrame(rows, columns=["text", "label"])
    df["text"] = df["text"].str.replace(r"\s+([@#])\s+", r" \1", regex=True)  # "@ user" -> "@user"
    return text_entry({"id": "sentimix_hinglish", "name": "Hinglish tweet sentiment (SentiMix)",
                       "description": "Code-mixed Hindi-English tweets labelled positive, neutral or negative.",
                       "source": "SemEval-2020 Task 9 SentiMix (HF RTT1/SentiMix)",
                       "licence": "Research use (see source)",
                       "source_url": "https://huggingface.co/datasets/RTT1/SentiMix"}, df)


# ---------------------------------------------------------------- tabular

def table_entry(meta: dict, df: pd.DataFrame, target: str, task: str, ids: tuple[str, ...] = ()) -> dict:
    if len(df) > MAX_ROWS:
        df = (pd.DataFrame(stratified(list(df.itertuples(index=False)), lambda r: getattr(r, target), MAX_ROWS),
                           columns=df.columns) if task == "tabular_classification"
              else df.sample(MAX_ROWS, random_state=SEED))
    cols = [{"name": target, "role": "target", "type": "categorical" if task == "tabular_classification"
             else "numeric"}] + [{"name": c, "role": "id", "type": None} for c in ids]
    classes = sorted(df[target].astype(str).unique().tolist()) if task == "tabular_classification" else []
    return starters.publish_table({**meta, "task_type": task, "classes": classes, "target": target},
                                  df.to_csv(index=False).encode(), cols, len(df))


def iris():
    with zipfile.ZipFile(CACHE / "iris" / "iris.zip") as z:
        raw = z.read("iris.data").decode()
    df = pd.read_csv(io.StringIO(raw), header=None,
                     names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]).dropna()
    df["species"] = df["species"].str.replace("Iris-", "")
    return table_entry({"id": "iris", "name": "Iris flowers",
                        "description": "Petal and sepal measurements of 150 irises; predict the species.",
                        "source": "UCI Machine Learning Repository", "licence": "CC BY 4.0",
                        "source_url": "https://archive.ics.uci.edu/dataset/53/iris"},
                       df, "species", "tabular_classification")


def titanic():
    df = pd.read_csv(CACHE / "titanic" / "titanic3.csv")
    df = df.drop(columns=[c for c in ("boat", "body", "home.dest", "name", "ticket") if c in df])  # boat/body leak
    df["survived"] = df["survived"].map({0: "no", 1: "yes"})
    return table_entry({"id": "titanic", "name": "Titanic survival",
                        "description": "Passenger class, age, sex, fare and family aboard; predict who survived.",
                        "source": "Vanderbilt Biostatistics (titanic3)", "licence": "Public (see source)",
                        "source_url": "https://hbiostat.org/data/"}, df, "survived", "tabular_classification")


def telco_churn():
    df = pd.read_csv(CACHE / "telco_churn" / "Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return table_entry({"id": "telco_churn", "name": "Telecom customer churn",
                        "description": "Contracts, services and charges of telecom customers; predict who leaves.",
                        "source": "IBM sample data", "licence": "Apache-2.0",
                        "source_url": "https://github.com/IBM/telco-customer-churn-on-icp4d"},
                       df, "Churn", "tabular_classification", ids=("customerID",))


def adult_income():
    names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation",
             "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country",
             "income"]
    with zipfile.ZipFile(CACHE / "adult_income" / "adult.zip") as z:
        raw = z.read("adult.data").decode()
    df = pd.read_csv(io.StringIO(raw), header=None, names=names, skipinitialspace=True, na_values="?").dropna(
        subset=["income"])
    return table_entry({"id": "adult_income", "name": "Adult income (census)",
                        "description": "US census records; predict whether income is above $50K a year.",
                        "source": "UCI Machine Learning Repository", "licence": "CC BY 4.0",
                        "source_url": "https://archive.ics.uci.edu/dataset/2/adult"},
                       df, "income", "tabular_classification")


def california_housing():
    df = pd.read_csv(CACHE / "california_housing" / "housing.csv")
    return table_entry({"id": "california_housing", "name": "California house prices",
                        "description": "1990 census block groups: location, rooms, population and income; predict "
                                       "the median house value.",
                        "source": "StatLib via ageron/handson-ml2", "licence": "Apache-2.0 (repo); public data",
                        "source_url": "https://github.com/ageron/handson-ml2/tree/master/datasets/housing"},
                       df, "median_house_value", "tabular_regression")


BUILDERS = {"beans": beans, "tf_flowers": tf_flowers, "oxford_pets": oxford_pets, "penn_fudan": penn_fudan,
            "coco_person_car_dog": coco_subset, "imdb": imdb, "sms_spam": sms_spam,
            "sentimix_hinglish": sentimix_hinglish, "iris": iris, "titanic": titanic, "telco_churn": telco_churn,
            "adult_income": adult_income, "california_housing": california_housing}


def main(ids: list[str]) -> None:
    storage.ensure_bucket()
    entries = {e["id"]: e for e in starters.catalog()}
    for sid in ids or list(BUILDERS):
        try:
            for old in storage.existing_sizes(f"{starters.PREFIX}{sid}/"):  # rebuild from scratch
                storage.delete(old)
            entries[sid] = BUILDERS[sid]()
            e = entries[sid]
            print(f"ok    {sid:22} {e['task_type']:24} {e['sample_count']:6} samples  {e['size_bytes'] / 1e6:7.1f} MB  "
                  f"{e['classes'][:6]}")
        except (FileNotFoundError, KeyError) as ex:
            print(f"skip  {sid:22} raw data missing: {ex}")
        starters.write_catalog([entries[k] for k in BUILDERS if k in entries])  # after each, so progress is usable


if __name__ == "__main__":
    main(sys.argv[1:])
