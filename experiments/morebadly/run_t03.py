"""morebadly T03 compatibility CLI; delegates to the team's shared entry."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_pipeline import main as pipeline_main  # noqa: E402
from run_pipeline import metrics as metrics  # noqa: E402
from run_pipeline import run_from_config  # noqa: E402
from run_pipeline import validation_threshold as validation_threshold  # noqa: E402


def run(input_path, output_dir, config_path=ROOT / 'configs/experiment.json', *, synthetic=False):
    """Keep the original personal runner API while using the shared pipeline."""
    return run_from_config(input_path, output_dir, config_path, synthetic=synthetic,
                           entry_script='experiments/morebadly/run_t03.py')


if __name__ == '__main__':
    pipeline_main(entry_script='experiments/morebadly/run_t03.py')
