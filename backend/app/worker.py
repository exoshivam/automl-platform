"""Celery app. Dev: uv run celery -A app.worker worker -P threads -c 4 -Q default --loglevel INFO
Tests set CELERY_EAGER=true so tasks run inline."""

import importlib.util

from celery import Celery

from app.settings import get_settings

s = get_settings()
# ML task modules join when they exist (e.g. app.ml.eval_tasks). They import torch etc. lazily, inside tasks.
TASK_MODULES = ["app.tasks"] + [mod for mod in ("app.ml.train_tasks", "app.ml.eval_tasks", "app.ml.autolabel")
                                if importlib.util.find_spec(mod) is not None]
celery = Celery("automl", broker=s.redis_url, backend=s.redis_url, include=TASK_MODULES)
celery.conf.update(
    task_always_eager=s.celery_eager,
    task_eager_propagates=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_default_queue="default",
    broker_connection_retry_on_startup=True,
    # One torch-loaded process on this 16 GB machine: training, evaluation and export all run on "train" (-c 1).
    task_routes={"train_job": {"queue": "train"}, "ml.*": {"queue": "train"}},
)
