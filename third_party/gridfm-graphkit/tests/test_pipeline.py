import sys
from argparse import ArgumentParser
from unittest import mock

import pytest

from gridfm_graphkit.cli import main_cli
from gridfm_graphkit.__main__ import main


# -------------------------------------------------
# Test configurations
# -------------------------------------------------
CONFIGS = [
    "tests/config/datamodule_test_base_config.yaml",
    "tests/config/datamodule_test_base_config2.yaml",
    "tests/config/datamodule_test_base_config3.yaml",
]

DATA_PATH = "tests/data"
LOG_DIR = "tests/mlruns"
MODEL_PATH = "tests/models/dummy_model.pt"
EXP_NAME = "pytest_exp"
RUN_NAME = "pytest_run"


# -------------------------------------------------
# Fixtures
# -------------------------------------------------
@pytest.fixture
def parser():
    """
    Argument parser matching the CLI.
    """
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for cmd in ["train", "finetune", "evaluate"]:
        sp = subparsers.add_parser(cmd)
        sp.add_argument("--config")
        sp.add_argument("--data_path")
        sp.add_argument("--log_dir")
        sp.add_argument("--model_path")
        sp.add_argument("--exp_name")
        sp.add_argument("--run_name")

    return parser


# -------------------------------------------------
# CLI command tests
# -------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("command", ["train", "finetune", "evaluate"])
def test_cli_commands(parser, config, command):
    """
    Test main_cli() directly for all commands and configs.
    """
    args_list = [
        command,
        "--config",
        config,
        "--data_path",
        DATA_PATH,
        "--log_dir",
        LOG_DIR,
        "--exp_name",
        EXP_NAME,
        "--run_name",
        RUN_NAME,
    ]

    if command in ["finetune", "evaluate"]:
        args_list += ["--model_path", MODEL_PATH]

    args = parser.parse_args(args_list)

    # Should run without raising
    main_cli(args)


# -------------------------------------------------
# Entrypoint (__main__) test
# -------------------------------------------------
@pytest.mark.parametrize("config", CONFIGS)
def test_entrypoint_train(config):
    """
    Test the console entrypoint: python -m gridfm_graphkit train ...
    """
    test_argv = [
        "gridfm_graphkit",
        "train",
        "--config",
        config,
        "--data_path",
        DATA_PATH,
        "--log_dir",
        LOG_DIR,
        "--exp_name",
        EXP_NAME,
        "--run_name",
        RUN_NAME,
    ]

    with mock.patch.object(sys, "argv", test_argv):
        main()
