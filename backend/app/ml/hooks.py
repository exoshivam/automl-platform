"""Called by training after a model version is marked ready and committed. Enqueue only."""


def on_model_ready(model_version_id) -> None:
    from app.ml import eval_tasks

    eval_tasks.evaluate.delay(str(model_version_id))
