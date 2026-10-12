"""Consolidated comparison report generation across experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def generate_consolidated_comparison(
    topology_results: pd.DataFrame | None,
    dimension_results: pd.DataFrame | None,
    output_dir: Path,
    is_quick_mode: bool = False,
) -> list[Path]:
    """Generate consolidated comparison tables and reports.

    Args:
        topology_results: DataFrame from topology selection.
        dimension_results: DataFrame from dimension comparison.
        output_dir: Output directory.
        is_quick_mode: Whether to add quick-mode warnings.

    Returns:
        List of generated file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    if topology_results is not None and not topology_results.empty:
        path = output_dir / "consolidated_topology_comparison.csv"
        topology_results.to_csv(path, index=False)
        generated.append(path)
        logger.info("Consolidated topology comparison saved: %s", path)

    if dimension_results is not None and not dimension_results.empty:
        path = output_dir / "consolidated_dimension_comparison.csv"
        dimension_results.to_csv(path, index=False)
        generated.append(path)
        logger.info("Consolidated dimension comparison saved: %s", path)

    return generated
