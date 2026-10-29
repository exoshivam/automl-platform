"""Child-process entry point for one training job: python -m app.ml.run_job <job_id>."""

import sys

from app.ml.train_tasks import run

if __name__ == "__main__":
    run(sys.argv[1])
