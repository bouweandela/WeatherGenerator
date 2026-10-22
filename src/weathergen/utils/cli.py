import argparse
import enum
from pathlib import Path

import pandas as pd


class Stage(enum.StrEnum):
    train = enum.auto()
    train_continue = enum.auto()
    inference = enum.auto()


def get_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="stage")

    train_parser = subparsers.add_parser(
        Stage.train,
        help="Train a WeatherGenerator configuration from the ground up.",
    )
    _add_train_args(train_parser)
    continue_parser = subparsers.add_parser(
        Stage.train_continue,
        help="Resume training from a pretrained WeatherGenerator configuration.",
    )
    _add_continue_args(continue_parser)
    inference_parser = subparsers.add_parser(
        Stage.inference,
        help="Run infernce on a trained WeatherGenerator configuration",
    )
    _add_inference_args(inference_parser)

    return parser


def get_train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_train_args(parser)

    return parser


def get_continue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_continue_args(parser)

    return parser


def get_inference_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    _add_inference_args(parser)

    return parser


def _add_train_args(parser: argparse.ArgumentParser):
    _add_general_arguments(parser)


def _add_continue_args(parser: argparse.ArgumentParser):
    _add_general_arguments(parser)
    _add_model_loading_params(parser)

    parser.add_argument(
        "--finetune-forecast",
        action="store_true",
        help=(
            "Fine tune for forecasting. It overwrites some of the Config settings. "
            "Overwrites specified with --config take precedence."
        ),
    )


def _add_inference_args(parser: argparse.ArgumentParser):
    _add_model_loading_params(parser)
    _add_general_arguments(parser)


def _format_date(date: str) -> str:
    try:
        parsed = pd.to_datetime(date, errors="raise")
    except (pd.errors.ParserError, ValueError) as e:
        msg = f"Can not parse a valid date from input: {date}, with type {type(date)}."
        raise ValueError(msg) from e

    return parsed.strftime("%Y%m%d%H%M")


def _add_general_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=[],
        help="Optional experiment specfic configuration files in ascending order of precedence.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help=(
            "The run id for this run."
            " All artifacts (models, metrics, ...) will be stored under this run_id."
            " If not provided, a new run_id will be created"
        ),
    )
    parser.add_argument(
        "--options",
        nargs="+",
        default=[],
        help=(
            "Overwrite individual config options."
            " This takes precedence over overwrites passed via --config or --finetune_forecast."
            " Individual items should be of the form: parent_obj.nested_obj=value"
        ),
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        nargs="?",
        help=(
            "Path to the base configuration file."
            "If not provided, ./config/default_config.yml is used."
        ),
        default=None,
    )


def _add_model_loading_params(parser: argparse.ArgumentParser):
    _add_pretrained_model_args(parser, required=True)
    parser.add_argument(
        "--reuse-run-id",
        action="store_true",
        help="Use the id given via --from-run-id also for the current run. "
        "The storage location for artifacts will be reused as well. "
        "This might overwrite artifacts from previous runs.",
    )


def _add_pretrained_model_args(parser: argparse.ArgumentParser, required: bool):
    parser.add_argument(
        "-id",
        "--from-run-id",
        required=required,
        help=(
            "Start inference or continue training from the WeatherGenerator"
            " model with the given run id."
        ),
    )
    parser.add_argument(
        "-e",
        "--mini-epoch",
        type=int,
        default=-1,
        help=(
            "Mini-epoch of pretrained WeatherGenerator model used"
            " (Default -1 corresponds to the last checkpoint)."
        ),
    )


def get_download_parser() -> argparse.ArgumentParser:
    """Parser of `download_datasets`: the datasets of a run as for `inference` or `train`."""
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Download the remote datasets of a run to a local directory, restricted to "
        "the dates of the run's stages, for compute nodes without internet access.",
    )
    _add_pretrained_model_args(parser, required=False)
    _add_general_arguments(parser)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Directory for the local copies (default: first local directory in data_paths).",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=["training", "validation", "test"],
        help="Stages whose dates to cover (default: test with --from-run-id, else training "
        "and validation).",
    )
    parser.add_argument("--start", help="Override the start date of all stages.")
    parser.add_argument("--end", help="Override the end date of all stages.")
    parser.add_argument("--all", action="store_true", help="Copy the datasets completely.")
    parser.add_argument("--transfers", type=int, default=16, help="Parallel downloads.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download existing chunks.")
    parser.add_argument("--dry-run", action="store_true", help="Only report what would be copied.")
    return parser
