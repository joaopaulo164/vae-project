"""Orchestrator of the factorial scientific protocol.

Flow executed by :func:`run_factorial_protocol`:

    validate -> new run directory (or resume) -> shared files
    -> train every topology x latent-dimension combination (unsupervised)
    -> factorial table -> winner selection (validation only)
    -> single test evaluation of the winner -> supervised extension
    -> test evaluation of the extension -> comparison -> reports/manifest/state

Everything is delegated to shared Python services
(:func:`~vae_project.experiments.single_experiment.execute_single_experiment`,
:func:`~vae_project.experiments.heldout_evaluation.evaluate_on_test`); no CLI
command is invoked and nothing is duplicated. The state is saved after every
combination and phase, so an interrupted run can be resumed.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from vae_project.config.loader import save_resolved_config
from vae_project.config.models import ExperimentConfig, TopologyConfig
from vae_project.config.validation import validate_config
from vae_project.data.splits import DataSplits
from vae_project.experiments.experiment_result import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    EvidenceProfile,
    SingleExperimentResult,
)
from vae_project.experiments.heldout_evaluation import (
    RESULTS_FILENAME,
    HeldOutEvaluation,
    evaluate_on_test,
)
from vae_project.experiments.protocol_models import (
    FINAL_EVALUATION_DIRECTORY,
    GENERATED_CONFIGS_DIRECTORY,
    MANIFEST_FILENAME,
    SELECTION_DIRECTORY,
    SHARED_DIRECTORY,
    SUMMARY_FILENAME,
    SUPERVISED_DIRECTORY,
    CombinationRecord,
    FactorialCombination,
    ProtocolError,
    build_factorial_grid,
    failed_record,
    is_reusable,
    missing_run_files,
    read_combination_record,
    read_run_record,
)
from vae_project.experiments.protocol_selection import (
    SelectionOutcome,
    select_winner,
    write_factorial_table,
    write_selected_configuration,
    write_selection_summary,
)
from vae_project.experiments.protocol_shared import SharedSamples, write_shared_files
from vae_project.experiments.protocol_state import (
    PHASE_DONE,
    PHASE_FINAL_EVALUATION,
    PHASE_GRID,
    PHASE_REPORTING,
    PHASE_SELECTION,
    PHASE_SUPERVISED_EVALUATION,
    PHASE_SUPERVISED_TRAINING,
    STATUS_AWAITING_MANUAL,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_INTERRUPTED,
    STATUS_RUNNING,
    STATUS_SKIPPED,
    ProtocolState,
    ProtocolStateError,
    compute_config_hash,
    load_state,
    now_timestamp,
    save_state,
    validate_resume,
)
from vae_project.experiments.protocol_state import (
    STATUS_FAILED as PROTOCOL_FAILED,
)
from vae_project.experiments.single_experiment import (
    ExperimentData,
    SingleExperimentOptions,
    derive_run_config,
    execute_single_experiment,
    prepare_experiment_data,
)
from vae_project.reporting.protocol_report import (
    ProtocolReportContext,
    write_evidence_index,
    write_protocol_manifest,
    write_protocol_summary,
)
from vae_project.reporting.supervised_comparison import ModeArtifacts, write_supervised_comparison
from vae_project.utils.json_io import read_json
from vae_project.utils.logging import setup_logging
from vae_project.utils.paths import (
    build_comparison_run_dir,
    build_run_timestamp,
    ensure_dir,
    ensure_run_dir_is_free,
    ensure_smoke_test_dir,
    sanitize_path_component,
)

logger = logging.getLogger(__name__)

PROTOCOL_BASE_FOLDER = "factorial_protocol"
UNSUPERVISED_FOLDER = "unsupervised"
PROTOCOL_CONFIG_FILENAME = "protocol_config_resolved.yaml"
SUPERVISED_CONFIG_FILENAME = "supervised_selected_configuration.yaml"
PROTOCOL_LOG_PATH = Path("logs") / "protocol.log"
RENAME_ATTEMPTS = 5
RENAME_RETRY_SECONDS = 1.0
ENVIRONMENT_REFERENCE = "shared/environment.json"
SUPERVISED_TRAINING_DIRECTORY = "training"
SUPERVISED_EVALUATION_DIRECTORY = "final_evaluation"
SYNTHETIC_ORIGIN = "synthetic"
REAL_ORIGIN = "real"
REUSABLE_EXECUTION_STATUSES = (STATUS_COMPLETED, STATUS_PARTIAL)
MANUAL_COMMAND_TEMPLATE = "python -m vae_project.cli run-pipeline --config {config_path}"


@dataclass
class ProtocolResult:
    """Outcome of one protocol execution.

    Attributes:
        protocol_directory: Run directory holding every artifact.
        status: Final protocol status (see ``protocol_state``).
        state: Final persisted state.
        records: One record per grid combination.
        selection: Selection outcome (None if the run stopped earlier).
        final_evaluation: Test evaluation of the winner executed in this call.
        supervised_training: Supervised training executed in this call.
        supervised_evaluation: Supervised test evaluation executed in this call.
        manual_command: Command for the manual supervised execution.
        report_paths: Summary, evidence index and manifest paths.
    """

    protocol_directory: Path
    status: str
    state: ProtocolState
    records: list[CombinationRecord] = field(default_factory=list)
    selection: SelectionOutcome | None = None
    final_evaluation: HeldOutEvaluation | None = None
    supervised_training: SingleExperimentResult | None = None
    supervised_evaluation: HeldOutEvaluation | None = None
    manual_command: str | None = None
    report_paths: dict[str, Path] = field(default_factory=dict)


def default_protocol_base_dir(config: ExperimentConfig) -> Path:
    """Return ``<artifacts>/<dataset>/unsupervised/factorial_protocol``."""
    return (
        config.paths.artifacts_dir
        / sanitize_path_component(config.dataset.name.value)
        / UNSUPERVISED_FOLDER
        / PROTOCOL_BASE_FOLDER
    )


def _origin(splits: DataSplits | None) -> str:
    """Return the data origin label of a run."""
    return SYNTHETIC_ORIGIN if splits is not None else REAL_ORIGIN


def _phase_status(execution_status: str) -> str:
    """Map an execution status to the vocabulary of the protocol state."""
    if execution_status == STATUS_COMPLETED:
        return STATUS_COMPLETED
    if execution_status == STATUS_PARTIAL:
        return STATUS_COMPLETED_WITH_WARNINGS
    return PROTOCOL_FAILED


def _rename_with_retries(source: Path, target: Path) -> None:
    """Rename a directory, retrying while Windows holds a transient file lock.

    Antivirus scanners and indexers can keep a handle on freshly written files for
    a short time, which makes ``Path.rename`` fail with ``PermissionError``.

    Raises:
        ProtocolError: If the directory is still locked after every attempt.
    """
    for attempt in range(1, RENAME_ATTEMPTS + 1):
        try:
            source.rename(target)
            return
        except PermissionError as error:
            if attempt == RENAME_ATTEMPTS:
                raise ProtocolError(
                    f"Cannot preserve the incomplete directory {source}: it is locked by "
                    f"another process ({error}). Close programs using it and resume again."
                ) from error
            logger.warning(
                "Directory %s is locked (attempt %d/%d); retrying", source, attempt, RENAME_ATTEMPTS
            )
            time.sleep(RENAME_RETRY_SECONDS)


def _move_aside(directory: Path) -> Path | None:
    """Rename a stale, incomplete directory so its partial artifacts are kept.

    Returns:
        The new path, or None when ``directory`` does not exist.
    """
    if not directory.exists():
        return None
    stamp = build_run_timestamp()
    target = directory.with_name(f"{directory.name}_incomplete_{stamp}")
    counter = 1
    while target.exists():
        target = directory.with_name(f"{directory.name}_incomplete_{stamp}_{counter}")
        counter += 1
    _rename_with_retries(directory, target)
    logger.warning("Incomplete artifacts preserved at %s", target)
    return target


class FactorialProtocolRunner:
    """Executes one protocol run (new or resumed) phase by phase."""

    def __init__(
        self,
        config: ExperimentConfig,
        protocol_dir: Path,
        state: ProtocolState,
        data: ExperimentData,
    ) -> None:
        self.config = config
        self.protocol_dir = protocol_dir
        self.state = state
        self.data = data
        self.shared: SharedSamples | None = None
        self.grid = build_factorial_grid(config)
        self.records: list[CombinationRecord] = []
        self.selection: SelectionOutcome | None = None
        self.final_evaluation: HeldOutEvaluation | None = None
        self.supervised_training: SingleExperimentResult | None = None
        self.supervised_evaluation: HeldOutEvaluation | None = None
        self.manual_command: str | None = None
        self.combination_failures: dict[str, str] = {}

    # ------------------------------------------------------------------ state

    def _save(self) -> None:
        """Persist the state."""
        save_state(self.state, self.protocol_dir)

    def _set_phase(self, phase: str) -> None:
        """Enter a phase and persist the state."""
        self.state.phase = phase
        self._save()

    def _options(self, run_id: str) -> SingleExperimentOptions:
        """Build the options that share indices and seed across executions."""
        if self.shared is None:
            raise ProtocolError("shared files must be written before any execution")
        return SingleExperimentOptions(
            run_id=run_id,
            reconstruction_indices=self.shared.reconstruction_indices,
            generation_seed=self.shared.generation_seed,
            num_generated_samples=self.shared.num_generated_samples,
            include_latent_traversal=False,
            write_environment=False,
            environment_reference=ENVIRONMENT_REFERENCE,
        )

    # ------------------------------------------------------------------- grid

    def run_grid(self) -> None:
        """Train every combination that is not already complete."""
        self._set_phase(PHASE_GRID)
        profile = (
            EvidenceProfile.FULL
            if self.config.factorial_comparison.generate_full_evidence
            else EvidenceProfile.TRAINING_ONLY
        )
        for position, combination in enumerate(self.grid, start=1):
            existing = read_combination_record(self.protocol_dir, combination)
            if is_reusable(self.protocol_dir, combination, existing, self.config):
                logger.info("Skipping %s: already complete (resume).", combination.key)
                self.state.mark_combination(combination.key, True)
                continue
            logger.info(
                "Combination %d/%d: %s (topology=%s, latent_dimension=%d)",
                position,
                len(self.grid),
                combination.key,
                combination.topology_name,
                combination.latent_dimension,
            )
            self._train_combination(combination, profile)
            self._save()

    def _train_combination(
        self,
        combination: FactorialCombination,
        profile: EvidenceProfile,
    ) -> None:
        """Train one combination from scratch and record its outcome."""
        run_dir = self.protocol_dir / combination.relative_directory
        _move_aside(run_dir)
        self.state.current_combination = combination.key
        self._save()
        try:
            result = execute_single_experiment(
                config=self.config,
                topology=combination.topology,
                latent_dimension=combination.latent_dimension,
                supervised=False,
                evidence_profile=profile,
                output_directory=run_dir,
                data=self.data,
                options=self._options(
                    f"{combination.topology_name}_latent_dim_{combination.latent_dimension}"
                ),
            )
        except Exception as error:
            logger.error("Combination %s failed: %s", combination.key, error, exc_info=True)
            message = f"{type(error).__name__}: {error}"
            self.combination_failures[combination.key] = message
            self.state.add_error(f"{combination.key}: {message}")
            self.state.mark_combination(combination.key, False)
            return
        keep = result.status == STATUS_COMPLETED or (
            result.status == STATUS_PARTIAL
            and self.config.protocol.continue_on_noncritical_evidence_error
        )
        if result.status == STATUS_PARTIAL:
            self.state.add_warning(
                f"{combination.key}: evidence incomplete ({result.error_summary()})"
            )
        if result.status == STATUS_FAILED:
            self.state.add_error(f"{combination.key}: {result.error_summary()}")
        self.state.mark_combination(combination.key, keep)

    def _collect_records(self) -> list[CombinationRecord]:
        """Rebuild one record per combination from the files on disk."""
        records: list[CombinationRecord] = []
        for combination in self.grid:
            record = read_combination_record(self.protocol_dir, combination)
            if record is None:
                message = self.combination_failures.get(
                    combination.key, "no metrics.json was written for this combination"
                )
                record = failed_record(self.config, combination, message)
            records.append(record)
        return records

    # -------------------------------------------------------------- selection

    def run_selection(self) -> bool:
        """Consolidate the grid and choose the winner.

        Returns:
            True when a valid winner exists.
        """
        self._set_phase(PHASE_SELECTION)
        self.records = self._collect_records()
        rule = self.config.factorial_comparison.selection
        selection_dir = ensure_dir(self.protocol_dir / SELECTION_DIRECTORY)
        write_factorial_table(self.records, selection_dir)
        outcome = select_winner(
            self.records,
            rule,
            self.protocol_dir,
            allow_partial=self.config.protocol.continue_on_noncritical_evidence_error,
        )
        previous = self.state.selected_configuration
        if previous and outcome.winner is not None:
            if previous.get("key") != outcome.winner.key:
                raise ProtocolError(
                    "The stored selection "
                    f"({previous.get('key')}) differs from the recomputed one "
                    f"({outcome.winner.key}); results of different runs must not be mixed."
                )
            outcome.selected_at = str(previous.get("selected_at", outcome.selected_at))
        self.selection = outcome
        write_selection_summary(outcome, rule, selection_dir, self.data.synthetic)
        if outcome.winner is None:
            message = (
                "No valid combination was found: the test evaluation and the supervised "
                "extension were not executed. See selection/selection_summary.md."
            )
            logger.error(message)
            self.state.add_error(message)
            return False
        write_selected_configuration(outcome, rule, self.config, selection_dir)
        winner = outcome.winner
        self.state.selected_configuration = {
            "key": winner.key,
            "topology_name": winner.topology_name,
            "hidden_dimensions": winner.hidden_dimensions,
            "latent_dimension": winner.latent_dimension,
            "validation_total_loss": winner.validation_total_loss,
            "checkpoint_path": winner.checkpoint_path,
            "selected_at": outcome.selected_at,
        }
        self._save()
        logger.info("Selected combination: %s", winner.key)
        return True

    def _winner(self) -> CombinationRecord:
        """Return the selected record."""
        if self.selection is None or self.selection.winner is None:
            raise ProtocolError("no selected configuration is available")
        return self.selection.winner

    def _winner_checkpoint_dir(self) -> Path:
        """Return the checkpoint directory of the winner."""
        winner = self._winner()
        if winner.checkpoint_path is None:
            raise ProtocolError("the selected combination has no checkpoint")
        return (self.protocol_dir / winner.checkpoint_path).parent

    def _winner_topology(self) -> TopologyConfig:
        """Return the topology of the winner."""
        winner = self._winner()
        return TopologyConfig(
            name=winner.topology_name, hidden_dimensions=list(winner.hidden_dimensions)
        )

    # ------------------------------------------------------ evaluation phases

    def _evaluation_is_reusable(
        self,
        directory: Path,
        experiment_type: str,
    ) -> bool:
        """Check that a finished test evaluation of the winner is on disk."""
        results_path = directory / RESULTS_FILENAME
        if not results_path.is_file():
            return False
        payload = read_json(results_path)
        winner = self._winner()
        return (
            isinstance(payload, dict)
            and payload.get("execution_status") in REUSABLE_EXECUTION_STATUSES
            and payload.get("experiment_type") == experiment_type
            and payload.get("topology_name") == winner.topology_name
            and payload.get("latent_dimension") == winner.latent_dimension
            and (directory / MANIFEST_FILENAME).is_file()
            and (directory / SUMMARY_FILENAME).is_file()
        )

    def run_final_evaluation(self) -> bool:
        """Evaluate only the winner on the test split.

        Returns:
            False when the evaluation failed (the protocol then stops).
        """
        self._set_phase(PHASE_FINAL_EVALUATION)
        if not self.config.final_evaluation.enabled:
            self.state.final_evaluation_status = STATUS_SKIPPED
            self._save()
            return True
        directory = self.protocol_dir / FINAL_EVALUATION_DIRECTORY
        if self._evaluation_is_reusable(directory, "unsupervised"):
            logger.info("Skipping final evaluation: already complete (resume).")
            self.state.final_evaluation_status = STATUS_COMPLETED
            self._save()
            return True
        _move_aside(directory)
        self.state.final_evaluation_status = STATUS_RUNNING
        self._save()
        winner = self._winner()
        evaluation = evaluate_on_test(
            config=self.config,
            topology=self._winner_topology(),
            latent_dimension=winner.latent_dimension,
            supervised=False,
            checkpoint_dir=self._winner_checkpoint_dir(),
            output_dir=directory,
            data=self.data,
            options=self._options("final_evaluation"),
            selected_configuration=True,
        )
        self.final_evaluation = evaluation
        self.state.final_evaluation_status = _phase_status(evaluation.result.status)
        if evaluation.result.status == STATUS_FAILED:
            self.state.add_error(f"final evaluation failed: {evaluation.result.error_summary()}")
        elif evaluation.result.status == STATUS_PARTIAL:
            self.state.add_warning(
                f"final evaluation evidence incomplete: {evaluation.result.error_summary()}"
            )
        self._save()
        return evaluation.result.status != STATUS_FAILED

    # ---------------------------------------------------- supervised extension

    def _supervised_base_config(self) -> ExperimentConfig:
        """Return the configuration whose supervised block follows the extension."""
        extension = self.config.supervised_extension
        base = copy.deepcopy(self.config)
        base.supervised_component.classification_loss_weight = extension.classification_loss_weight
        base.supervised_component.use_latent_mean = extension.use_latent_mean
        return base

    def write_supervised_configuration(self) -> Path:
        """Write the resolved supervised configuration of the winner."""
        winner = self._winner()
        derived = derive_run_config(
            self._supervised_base_config(),
            self._winner_topology(),
            winner.latent_dimension,
            supervised=True,
        )
        errors = validate_config(derived)
        if errors:
            raise ProtocolError("invalid supervised configuration: " + "; ".join(errors))
        path = ensure_dir(self.protocol_dir / GENERATED_CONFIGS_DIRECTORY) / (
            SUPERVISED_CONFIG_FILENAME
        )
        save_resolved_config(derived, path)
        return path

    def _supervised_training_reusable(self, directory: Path) -> bool:
        """Check that a finished supervised training is on disk."""
        record = read_run_record(self.protocol_dir, directory)
        winner = self._winner()
        if record is None or record.status not in REUSABLE_EXECUTION_STATUSES:
            return False
        full = self.config.factorial_comparison.generate_full_evidence
        return (
            record.experiment_type == "supervised"
            and record.topology_name == winner.topology_name
            and record.latent_dimension == winner.latent_dimension
            and not missing_run_files(directory, record.status, full)
        )

    def run_supervised_extension(self) -> bool:
        """Train and evaluate the supervised extension (or prepare it manually).

        Returns:
            False when the training or its evaluation failed.
        """
        extension = self.config.supervised_extension
        if not extension.enabled:
            self.state.supervised_training_status = STATUS_SKIPPED
            self.state.supervised_evaluation_status = STATUS_SKIPPED
            self._save()
            return True
        self._set_phase(PHASE_SUPERVISED_TRAINING)
        config_path = self.write_supervised_configuration()
        if extension.execution_mode != "automatic":
            return self._prepare_manual_execution(config_path)

        succeeded = self._train_supervised()
        if succeeded:
            succeeded = self._evaluate_supervised()
        self._write_comparison()
        return succeeded

    def _prepare_manual_execution(self, config_path: Path) -> bool:
        """Record the command that runs the extension manually."""
        self.manual_command = MANUAL_COMMAND_TEMPLATE.format(config_path=config_path.as_posix())
        self.state.supervised_training_status = STATUS_AWAITING_MANUAL
        self.state.supervised_evaluation_status = STATUS_AWAITING_MANUAL
        self.state.add_warning("Supervised extension in manual mode. Run: " + self.manual_command)
        self._save()
        logger.info("Supervised extension is manual. Command: %s", self.manual_command)
        return True

    def _train_supervised(self) -> bool:
        """Train the supervised model from scratch, with the winner's structure."""
        directory = self.protocol_dir / SUPERVISED_DIRECTORY / SUPERVISED_TRAINING_DIRECTORY
        if self._supervised_training_reusable(directory):
            logger.info("Skipping supervised training: already complete (resume).")
            self.state.supervised_training_status = STATUS_COMPLETED
            self._save()
            return True
        _move_aside(directory)
        self.state.supervised_training_status = STATUS_RUNNING
        self._save()
        profile = (
            EvidenceProfile.FULL
            if self.config.factorial_comparison.generate_full_evidence
            else EvidenceProfile.TRAINING_ONLY
        )
        winner = self._winner()
        try:
            result = execute_single_experiment(
                config=self._supervised_base_config(),
                topology=self._winner_topology(),
                latent_dimension=winner.latent_dimension,
                supervised=True,
                evidence_profile=profile,
                output_directory=directory,
                data=self.data,
                options=self._options("supervised_training"),
            )
        except Exception as error:
            logger.error("Supervised training failed: %s", error, exc_info=True)
            self.state.supervised_training_status = PROTOCOL_FAILED
            self.state.add_error(f"supervised training failed: {type(error).__name__}: {error}")
            self._save()
            return False
        self.supervised_training = result
        self.state.supervised_training_status = _phase_status(result.status)
        if result.status == STATUS_FAILED:
            self.state.add_error(f"supervised training failed: {result.error_summary()}")
        elif result.status == STATUS_PARTIAL:
            self.state.add_warning(f"supervised evidence incomplete: {result.error_summary()}")
        self._save()
        return result.status != STATUS_FAILED

    def _evaluate_supervised(self) -> bool:
        """Evaluate the best supervised checkpoint once on the test split."""
        self._set_phase(PHASE_SUPERVISED_EVALUATION)
        base = self.protocol_dir / SUPERVISED_DIRECTORY
        directory = base / SUPERVISED_EVALUATION_DIRECTORY
        if self._evaluation_is_reusable(directory, "supervised"):
            logger.info("Skipping supervised evaluation: already complete (resume).")
            self.state.supervised_evaluation_status = STATUS_COMPLETED
            self._save()
            return True
        _move_aside(directory)
        self.state.supervised_evaluation_status = STATUS_RUNNING
        self._save()
        winner = self._winner()
        evaluation = evaluate_on_test(
            config=self._supervised_base_config(),
            topology=self._winner_topology(),
            latent_dimension=winner.latent_dimension,
            supervised=True,
            checkpoint_dir=base / SUPERVISED_TRAINING_DIRECTORY / "checkpoints",
            output_dir=directory,
            data=self.data,
            options=self._options("supervised_final_evaluation"),
            selected_configuration=True,
            phase_label="supervised extension final evaluation",
        )
        self.supervised_evaluation = evaluation
        self.state.supervised_evaluation_status = _phase_status(evaluation.result.status)
        if evaluation.result.status == STATUS_FAILED:
            self.state.add_error(
                f"supervised evaluation failed: {evaluation.result.error_summary()}"
            )
        self._save()
        return evaluation.result.status != STATUS_FAILED

    def _write_comparison(self) -> None:
        """Write the supervised vs unsupervised comparison files."""
        winner = self._winner()
        base = self.protocol_dir / SUPERVISED_DIRECTORY
        unsupervised = ModeArtifacts(
            "unsupervised",
            self.protocol_dir / winner.run_directory,
            self.protocol_dir / FINAL_EVALUATION_DIRECTORY,
        )
        supervised = ModeArtifacts(
            "supervised",
            base / SUPERVISED_TRAINING_DIRECTORY,
            base / SUPERVISED_EVALUATION_DIRECTORY,
        )
        write_supervised_comparison(
            self.protocol_dir, base, unsupervised, supervised, self.data.synthetic
        )

    # ---------------------------------------------------------------- reports

    def final_status(self, failed: bool) -> str:
        """Derive the protocol status from the phase outcomes."""
        if failed:
            return PROTOCOL_FAILED
        if self.manual_command is not None:
            return STATUS_AWAITING_MANUAL
        degraded = (
            any(record.status != STATUS_COMPLETED for record in self.records)
            or bool(self.state.warnings)
            or bool(self.state.errors)
        )
        return STATUS_COMPLETED_WITH_WARNINGS if degraded else STATUS_COMPLETED

    def write_reports(self, failed: bool) -> dict[str, Path]:
        """Finalize the state and write the summary, index and global manifest."""
        self._set_phase(PHASE_REPORTING)
        if self.records:
            self.records = self._collect_records()
        self.state.status = self.final_status(failed)
        self.state.phase = PHASE_DONE
        self._save()
        context = ProtocolReportContext(
            protocol_dir=self.protocol_dir,
            config=self.config,
            state=self.state,
            records=self.records,
            selection=self.selection,
            manual_command=self.manual_command,
        )
        return {
            "protocol_summary": write_protocol_summary(context),
            "scientific_evidence_index": write_evidence_index(context),
            "protocol_manifest": write_protocol_manifest(context),
        }


def _prepare_new_run(
    config: ExperimentConfig,
    base_output_dir: Path | None,
    run_name: str | None,
    splits: DataSplits | None,
) -> tuple[Path, ProtocolState]:
    """Create the exclusive run directory and its initial state."""
    base = Path(base_output_dir) if base_output_dir else default_protocol_base_dir(config)
    protocol_dir = build_comparison_run_dir(base, run_name or config.run_name)
    if splits is not None:
        ensure_smoke_test_dir(protocol_dir)
    ensure_run_dir_is_free(protocol_dir, overwrite=False)
    ensure_dir(protocol_dir)
    grid = build_factorial_grid(config)
    state = ProtocolState(
        run_id=protocol_dir.name,
        config_hash=compute_config_hash(config),
        dataset=config.dataset.name.value,
        protocol_name=config.protocol.name,
        data_origin=_origin(splits),
        status=STATUS_RUNNING,
        pending_combinations=[combination.key for combination in grid],
    )
    save_resolved_config(config, protocol_dir / PROTOCOL_CONFIG_FILENAME)
    save_state(state, protocol_dir)
    return protocol_dir, state


def _prepare_resume(
    config: ExperimentConfig,
    resume_dir: Path,
    splits: DataSplits | None,
) -> tuple[Path, ProtocolState]:
    """Load and validate the state of an interrupted run."""
    protocol_dir = Path(resume_dir)
    if not config.protocol.resume_if_interrupted:
        raise ProtocolStateError("protocol.resume_if_interrupted is false: resume is disabled.")
    state = load_state(protocol_dir)
    validate_resume(state, config)
    if state.data_origin != _origin(splits):
        raise ProtocolStateError(
            f"Cannot resume: the run used {state.data_origin} data but this call uses "
            f"{_origin(splits)} data."
        )
    if splits is not None:
        ensure_smoke_test_dir(protocol_dir)
    grid = build_factorial_grid(config)
    done = set(state.completed_combinations)
    state.pending_combinations = [c.key for c in grid if c.key not in done]
    state.failed_combinations = [key for key in state.failed_combinations if key not in done]
    state.status = STATUS_RUNNING
    state.resumed_at.append(now_timestamp())
    save_state(state, protocol_dir)
    logger.info("Resuming protocol run %s", protocol_dir)
    return protocol_dir, state


def _close_file_handlers() -> None:
    """Close the protocol log so the run directory is not left locked."""
    project_logger = logging.getLogger("vae_project")
    for handler in list(project_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            project_logger.removeHandler(handler)
            handler.close()


def run_factorial_protocol(
    config: ExperimentConfig,
    base_output_dir: Path | None = None,
    run_name: str | None = None,
    resume_dir: Path | None = None,
    splits: DataSplits | None = None,
) -> ProtocolResult:
    """Run (or resume) the complete factorial scientific protocol.

    Args:
        config: Protocol configuration (``factorial_comparison.enabled`` true).
        base_output_dir: Base directory for new runs. A new
            ``run_YYYYMMDD_HHMMSS_<id>`` directory is always created inside it;
            existing runs are never overwritten.
        run_name: Optional identifier appended to the run directory name.
        resume_dir: Existing run directory to continue. The configuration must
            hash to the stored value.
        splits: Pre-built synthetic splits (tests and smoke tests only). The
            run directory must then contain ``smoke_test`` or ``synthetic``.

    Returns:
        ProtocolResult with the final status and the report paths.

    Raises:
        ProtocolError: If the configuration cannot run the protocol or the
            stored selection is inconsistent on resume.
        ProtocolStateError: If the run cannot be resumed.
        ArtifactError: If the run directory is not free or a synthetic run
            targets an unmarked directory.
    """
    errors = validate_config(config)
    if errors:
        raise ProtocolError("Invalid protocol configuration: " + "; ".join(errors))
    if not config.factorial_comparison.enabled:
        raise ProtocolError("factorial_comparison.enabled is false; nothing to run.")

    if resume_dir is not None:
        protocol_dir, state = _prepare_resume(config, resume_dir, splits)
    else:
        protocol_dir, state = _prepare_new_run(config, base_output_dir, run_name, splits)
    setup_logging(config.logging, log_file=protocol_dir / PROTOCOL_LOG_PATH)
    try:
        return _execute(config, protocol_dir, state, splits)
    except KeyboardInterrupt:
        state.status = STATUS_INTERRUPTED
        state.add_warning("The run was interrupted; resume it with --resume.")
        save_state(state, protocol_dir)
        raise
    except Exception as error:
        logger.error("Protocol failed: %s", error, exc_info=True)
        state.status = PROTOCOL_FAILED
        state.add_error(f"{type(error).__name__}: {error}")
        save_state(state, protocol_dir)
        raise
    finally:
        _close_file_handlers()


def _execute(
    config: ExperimentConfig,
    protocol_dir: Path,
    state: ProtocolState,
    splits: DataSplits | None,
) -> ProtocolResult:
    """Run every phase and build the result."""
    data = prepare_experiment_data(config, splits)
    runner = FactorialProtocolRunner(config, protocol_dir, state, data)
    runner.shared = write_shared_files(protocol_dir / SHARED_DIRECTORY, config, data)

    runner.run_grid()
    failed = not runner.run_selection()
    if not failed:
        failed = not runner.run_final_evaluation()
    if not failed:
        failed = not runner.run_supervised_extension()
    report_paths = runner.write_reports(failed)
    logger.info("Protocol finished with status %s: %s", runner.state.status, protocol_dir)
    return ProtocolResult(
        protocol_directory=protocol_dir,
        status=runner.state.status,
        state=runner.state,
        records=runner.records,
        selection=runner.selection,
        final_evaluation=runner.final_evaluation,
        supervised_training=runner.supervised_training,
        supervised_evaluation=runner.supervised_evaluation,
        manual_command=runner.manual_command,
        report_paths=report_paths,
    )


__all__ = [
    "FactorialProtocolRunner",
    "ProtocolResult",
    "default_protocol_base_dir",
    "run_factorial_protocol",
]
