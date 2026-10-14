"""Application entry point for running ML training and prediction workflows.

Usage:
    python run.py --mode train
    python run.py --mode predict
    python run.py --help
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from production_app.config import Config
from production_app.logging_config import get_logger, setup_logging
from production_app.main import predict_workflow, train_workflow

logger = get_logger("run")


def parse_arguments() -> argparse.Namespace:
    """Parse CLI options for the application."""
    parser = argparse.ArgumentParser(
        description="Production ML Pipeline - Breast Cancer Wisconsin Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        default="train",
        help="Pipeline mode: 'train' to fit and save, 'predict' to run inference.",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional override for model artifact path.",
    )

    parser.add_argument(
        "--data-source",
        type=str,
        default=None,
        help="Optional override for data source: 'sklearn', 'url', or path to CSV.",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Optional override for logging level.",
    )

    return parser.parse_args()


def main() -> int:
    """Main execution function."""
    args = parse_arguments()

    # Build custom configuration overrides if specified
    config_overrides: dict = {}
    if args.model_path:
        config_overrides["model_path"] = Path(args.model_path)
    if args.data_source:
        config_overrides["data_source"] = args.data_source
    if args.log_level:
        config_overrides["log_level"] = args.log_level

    config = Config(**config_overrides) if config_overrides else Config()
    setup_logging(config.log_level)

    try:
        if args.mode == "train":
            train_workflow(config=config)
        elif args.mode == "predict":
            predict_workflow(config=config)
        return 0

    except FileNotFoundError as fnf_err:
        logger.error("Configuration/File Error: %s", fnf_err)
        return 1
    except ValueError as val_err:
        logger.error("Validation Error: %s", val_err)
        return 1
    except Exception as exc:
        logger.exception("Unexpected system failure occurred: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
