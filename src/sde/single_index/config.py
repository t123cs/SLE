#!/usr/bin/env python3
"""Validate and render the released BEIR-5 single-index SDE configuration."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import sys
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the released configuration is incomplete or invalid."""


def require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object")
    return value


def require_value(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"{path}.{key} is required")
    return mapping[key]


def require_string(mapping: dict[str, Any], key: str, path: str) -> str:
    value = require_value(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path}.{key} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise ConfigError(f"{path}.{key} must not contain control characters")
    return value


def require_choice(
    mapping: dict[str, Any],
    key: str,
    path: str,
    choices: set[str],
) -> str:
    value = require_string(mapping, key, path)
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ConfigError(f"{path}.{key} must be one of: {expected}")
    return value


def require_integer(
    mapping: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: int | None = None,
) -> int:
    value = require_value(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path}.{key} must be >= {minimum}")
    return value


def require_number(
    mapping: dict[str, Any],
    key: str,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = require_value(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ConfigError(f"{path}.{key} must be finite")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{path}.{key} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{path}.{key} must be <= {maximum}")
    return number


def require_boolean(
    mapping: dict[str, Any],
    key: str,
    path: str,
) -> bool:
    value = require_value(mapping, key, path)
    if not isinstance(value, bool):
        raise ConfigError(f"{path}.{key} must be true or false")
    return value


def number_text(value: float) -> str:
    return format(value, ".15g")


def load_plan(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open(encoding="utf-8") as source:
            root = json.load(source)
    except FileNotFoundError as error:
        raise ConfigError(f"config file does not exist: {config_path}") from error
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"invalid JSON at {config_path}:{error.lineno}:{error.colno}: "
            f"{error.msg}"
        ) from error

    root = require_mapping(root, "config")
    method = require_string(root, "method", "config")
    if method != "SDE, single index":
        raise ConfigError("config.method must be 'SDE, single index'")

    selection_protocol = require_mapping(
        require_value(root, "selection_protocol", "config"),
        "config.selection_protocol",
    )
    selection_metric = require_string(
        selection_protocol,
        "metric",
        "config.selection_protocol",
    )
    expected_metric = "train delta_nDCG@10 - 0.5 * bootstrap_SE"
    if selection_metric != expected_metric:
        raise ConfigError(
            f"config.selection_protocol.metric must be {expected_metric!r}"
        )
    if require_boolean(
        selection_protocol,
        "test_tuning",
        "config.selection_protocol",
    ):
        raise ConfigError("config.selection_protocol.test_tuning must be false")
    transfer = require_mapping(
        require_value(
            selection_protocol,
            "transfer",
            "config.selection_protocol",
        ),
        "config.selection_protocol.transfer",
    )
    expected_transfer = {
        "scidocs": "fiqa-2018",
        "arguana": "nfcorpus",
    }
    if transfer != expected_transfer:
        raise ConfigError(
            "config.selection_protocol.transfer must map scidocs to fiqa-2018 "
            "and arguana to nfcorpus"
        )

    trace_config = require_mapping(
        require_value(root, "trace", "config"),
        "config.trace",
    )
    retrieval_config = require_mapping(
        require_value(root, "retrieval", "config"),
        "config.retrieval",
    )
    dataset_configs = require_mapping(
        require_value(root, "datasets", "config"),
        "config.datasets",
    )
    if not dataset_configs:
        raise ConfigError("config.datasets must contain at least one dataset")

    templates = require_integer(
        trace_config,
        "templates",
        "config.trace",
        minimum=1,
    )
    samples_per_template = require_integer(
        trace_config,
        "samples_per_template",
        "config.trace",
        minimum=1,
    )
    sample_idx_max = require_integer(
        trace_config,
        "sample_idx_max",
        "config.trace",
        minimum=0,
    )
    if sample_idx_max >= samples_per_template:
        raise ConfigError(
            "config.trace.sample_idx_max must be smaller than "
            "config.trace.samples_per_template"
        )
    trace_args = [
        "--expected_templates",
        str(templates),
        "--sample_idx_max",
        str(sample_idx_max),
        "--candidate_topk",
        str(
            require_integer(
                trace_config,
                "candidate_topk",
                "config.trace",
                minimum=2,
            )
        ),
        "--event_min_probability",
        number_text(
            require_number(
                trace_config,
                "event_min_probability",
                "config.trace",
                minimum=0,
                maximum=1,
            )
        ),
        "--candidate_min_len",
        str(
            require_integer(
                trace_config,
                "candidate_min_length",
                "config.trace",
                minimum=1,
            )
        ),
        "--boundary_min_len",
        str(
            require_integer(
                trace_config,
                "boundary_min_length",
                "config.trace",
                minimum=1,
            )
        ),
    ]

    retrieval_args = [
        "--min_aggregate_probability",
        number_text(
            require_number(
                retrieval_config,
                "min_aggregate_probability",
                "config.retrieval",
                minimum=0,
                maximum=1,
            )
        ),
        "--candidate_max_df_ratio",
        number_text(
            require_number(
                retrieval_config,
                "candidate_max_df_ratio",
                "config.retrieval",
                minimum=0,
                maximum=1,
            )
        ),
        "--k1",
        number_text(
            require_number(
                retrieval_config,
                "k1",
                "config.retrieval",
                minimum=0,
            )
        ),
        "--b",
        number_text(
            require_number(
                retrieval_config,
                "b",
                "config.retrieval",
                minimum=0,
                maximum=1,
            )
        ),
        "--retrieval_topn",
        str(
            require_integer(
                retrieval_config,
                "retrieval_topn",
                "config.retrieval",
                minimum=1,
            )
        ),
    ]

    datasets: list[dict[str, Any]] = []
    for dataset, raw_config in dataset_configs.items():
        if not isinstance(dataset, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*",
            dataset,
        ):
            raise ConfigError(
                "config.datasets keys may contain letters, digits, dot, "
                "underscore, and hyphen"
            )
        path = f"config.datasets.{dataset}"
        dataset_config = require_mapping(raw_config, path)
        datasets.append(
            {
                "name": dataset,
                "selection_source": require_string(
                    dataset_config,
                    "selection_source",
                    path,
                ),
                "position_mode": require_choice(
                    dataset_config,
                    "position_mode",
                    path,
                    {"all", "boundary"},
                ),
                "max_soft_terms": require_integer(
                    dataset_config,
                    "max_soft_terms",
                    path,
                    minimum=0,
                ),
                "exclude_original_terms": require_boolean(
                    dataset_config,
                    "exclude_original_terms",
                    path,
                ),
            }
        )

    return {
        "trace_args": trace_args,
        "retrieval_args": retrieval_args,
        "datasets": datasets,
    }


def shell_array(name: str, values: list[Any]) -> str:
    rendered = " ".join(shlex.quote(str(value)) for value in values)
    return f"{name}=({rendered})"


def render_shell(plan: dict[str, Any]) -> str:
    datasets = plan["datasets"]
    lines = [
        shell_array("CONFIG_DATASETS", [row["name"] for row in datasets]),
        shell_array(
            "CONFIG_POSITION_MODES",
            [row["position_mode"] for row in datasets],
        ),
        shell_array(
            "CONFIG_MAX_SOFT_TERMS",
            [row["max_soft_terms"] for row in datasets],
        ),
        shell_array(
            "CONFIG_EXCLUDE_ORIGINAL",
            [int(row["exclude_original_terms"]) for row in datasets],
        ),
        shell_array("CONFIG_TRACE_ARGS", plan["trace_args"]),
        shell_array("CONFIG_RETRIEVAL_ARGS", plan["retrieval_args"]),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--format", choices=("shell",), default="shell")
    args = parser.parse_args()
    try:
        plan = load_plan(args.config)
    except (ConfigError, OSError) as error:
        print(f"Invalid single-index SDE config: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(render_shell(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
