"""Typed configuration dataclasses for the VAE project.

All configuration is expressed through these dataclasses, which are
populated from YAML files. This provides type safety, IDE auto-completion,
and explicit documentation of every configuration option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class DatasetName(str, Enum):
    """Supported dataset names."""

    MNIST = "mnist"
    FASHION_MNIST = "fashion_mnist"


class DevicePolicy(str, Enum):
    """Device selection policy."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ReconstructionLossType(str, Enum):
    """Reconstruction loss function type.

    BCE (Binary Cross-Entropy) is the methodologically preferred default
    for images normalized to [0, 1] when they represent Bernoulli
    probabilities. MSE is provided as an alternative when the image
    values are interpreted as continuous measurements.
    """

    BCE = "bce"
    MSE = "mse"


class ProjectionMethod(str, Enum):
    """Available dimensionality reduction projection methods."""

    PCA = "pca"
    UMAP = "umap"
    TSNE = "tsne"


@dataclass
class ProjectConfig:
    """Top-level project metadata."""

    name: str = "vae_project"
    description: str = "Variational Autoencoder study for MNIST and Fashion-MNIST"
    version: str = "0.1.0"
    quick_mode: bool = False


@dataclass
class PathsConfig:
    """File system paths, all relative to the project root."""

    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    configs_dir: Path = Path("configs")

    def __post_init__(self) -> None:
        """Ensure paths are Path objects."""
        self.data_dir = Path(self.data_dir)
        self.artifacts_dir = Path(self.artifacts_dir)
        self.configs_dir = Path(self.configs_dir)


@dataclass
class DatasetConfig:
    """Dataset selection and loading configuration."""

    name: DatasetName = DatasetName.MNIST
    download: bool = True
    num_workers: int = 0  # Safe default for Windows
    pin_memory: bool = False

    def __post_init__(self) -> None:
        """Normalize dataset name to enum."""
        if isinstance(self.name, str):
            self.name = DatasetName(self.name)


@dataclass
class SplitsConfig:
    """Train/validation/test split configuration."""

    validation_fraction: float = 0.1
    seed: int = 42
    stratified: bool = True
    quick_mode_train_size: int = 500
    quick_mode_val_size: int = 100


@dataclass
class TopologyConfig:
    """VAE encoder/decoder topology expressed as hidden layer dimensions.

    The encoder uses these dimensions from input to latent space.
    The decoder mirrors them symmetrically.
    """

    name: str = "medium"
    hidden_dimensions: list[int] = field(default_factory=lambda: [512, 256])


@dataclass
class ModelConfig:
    """VAE model architecture configuration."""

    latent_dimension: int = 16
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    input_dimension: int = 784  # 28 * 28 for MNIST/Fashion-MNIST
    output_activation: str = "sigmoid"  # Ensures output in [0, 1]

    def __post_init__(self) -> None:
        """Normalize nested config."""
        if isinstance(self.topology, dict):
            self.topology = TopologyConfig(**self.topology)


@dataclass
class LossConfig:
    """VAE loss function configuration.

    reconstruction_loss_type: 'bce' is the default for Bernoulli-interpreted
        pixel values. 'mse' treats pixels as continuous Gaussian outputs.
    beta: KL divergence weight. Default 1.0 is standard VAE. Values > 1.0
        produce Beta-VAE behavior (not the focus of this project). Any
        value != 1.0 is logged as a deviation from standard VAE.
    reduction: 'sum' or 'mean' over pixels. 'sum' followed by mean over
        batch is the canonical VAE formulation.
    """

    reconstruction_loss_type: ReconstructionLossType = ReconstructionLossType.BCE
    beta: float = 1.0
    reduction: str = "sum"

    def __post_init__(self) -> None:
        """Normalize loss type to enum."""
        if isinstance(self.reconstruction_loss_type, str):
            self.reconstruction_loss_type = ReconstructionLossType(self.reconstruction_loss_type)


@dataclass
class SupervisedComponentConfig:
    """Optional supervised classification head on top of the latent space.

    When enabled: a classification head is trained jointly with the VAE.
    The total loss = vae_loss + classification_loss_weight * classification_loss.
    When disabled: behaves exactly as an unsupervised VAE.
    """

    enabled: bool = False
    classification_loss_weight: float = 1.0
    hidden_dimensions: list[int] = field(default_factory=lambda: [128])
    use_latent_mean: bool = True  # Use mu rather than sampled z for classification


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""

    enabled: bool = False
    name: str = "reduce_on_plateau"
    patience: int = 5
    factor: float = 0.5
    min_lr: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""

    enabled: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    monitor: str = "val_total_loss"
    restore_best_weights: bool = True


@dataclass
class TrainingConfig:
    """Training loop configuration."""

    max_epochs: int = 50
    batch_size: int = 128
    shuffle_train: bool = True
    log_interval_batches: int = 50  # Log every N batches


@dataclass
class RuntimeConfig:
    """Runtime device and execution configuration."""

    device: DevicePolicy = DevicePolicy.AUTO
    cuda_fallback_to_cpu: bool = False  # If True, falls back to CPU when CUDA missing
    deterministic: bool = False  # Enables torch.use_deterministic_algorithms
    benchmark: bool = True  # torch.backends.cudnn.benchmark

    def __post_init__(self) -> None:
        """Normalize device policy to enum."""
        if isinstance(self.device, str):
            self.device = DevicePolicy(self.device)


@dataclass
class ReproducibilityConfig:
    """Seed and determinism configuration."""

    global_seed: int = 42
    deterministic_mode: bool = False
    warn_nondeterministic: bool = True


@dataclass
class UMAPConfig:
    """UMAP projection hyperparameters."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42


@dataclass
class TSNEConfig:
    """t-SNE projection hyperparameters.

    Note: t-SNE distances and cluster sizes do not have a simple global
    interpretation. Only local neighborhood structure is preserved.
    """

    perplexity: float = 30.0
    learning_rate: float = 200.0
    max_iter: int = 1000
    init: str = "pca"
    random_state: int = 42


@dataclass
class SamplingConfig:
    """Configuration for subsampling large latent sets for projection."""

    enabled: bool = True
    max_samples: int = 5000
    stratified: bool = True
    seed: int = 42


@dataclass
class ProjectionsConfig:
    """Configuration for dimensionality reduction projections."""

    methods: list[str] = field(default_factory=lambda: ["pca", "umap", "tsne"])
    pca_n_components: int = 2
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    tsne: TSNEConfig = field(default_factory=TSNEConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        """Normalize nested configs."""
        if isinstance(self.umap, dict):
            self.umap = UMAPConfig(**self.umap)
        if isinstance(self.tsne, dict):
            self.tsne = TSNEConfig(**self.tsne)
        if isinstance(self.sampling, dict):
            self.sampling = SamplingConfig(**self.sampling)


@dataclass
class MetricsConfig:
    """Subsidiary quantitative metrics configuration."""

    compute_silhouette: bool = True
    compute_calinski_harabasz: bool = True
    compute_davies_bouldin: bool = True
    compute_trustworthiness: bool = True
    compute_linear_classifier: bool = True
    linear_classifier_name: str = "logistic_regression"
    kmeans_n_clusters: int | None = None  # None = use number of classes
    kmeans_random_state: int = 42
    trustworthiness_n_neighbors: int = 5
    silhouette_sample_size: int | None = 5000


@dataclass
class PlottingConfig:
    """Figure generation configuration."""

    dpi: int = 150
    figure_width: float = 10.0
    figure_height: float = 8.0
    save_png: bool = True
    save_svg: bool = True
    colormap: str = "tab10"
    point_size: float = 5.0
    point_alpha: float = 0.7
    max_display_points: int = 5000
    reconstruction_grid_rows: int = 4
    reconstruction_grid_cols: int = 8
    generated_grid_rows: int = 4
    generated_grid_cols: int = 8


@dataclass
class ArtifactConfig:
    """Artifact generation and management configuration."""

    save_latent_csv: bool = True
    save_latent_npz: bool = True
    save_training_history_csv: bool = True
    save_training_history_json: bool = True
    save_metrics_json: bool = True
    save_environment_json: bool = True
    save_run_summary_md: bool = True
    save_artifact_manifest: bool = True
    save_resolved_config: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    console_level: str = "INFO"
    file_level: str = "DEBUG"
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


@dataclass
class LatentDimensionComparisonConfig:
    """Configuration for comparing multiple latent dimensions.

    Each entry of ``dimensions`` is an independent training run. The list
    lives exclusively here; ``model.latent_dimension`` always holds a single
    integer for a single model.

    Attributes:
        enabled: Whether the comparison experiment may be executed.
        dimensions: Latent dimensions to train, one independent VAE each.
        fixed_topology: Name of the previously selected topology that every
            model in the comparison uses (no topology search is performed).
    """

    enabled: bool = True
    dimensions: list[int] = field(default_factory=lambda: [2, 8, 16, 32])
    fixed_topology: str = "medium"


# Standard topologies of the project: name -> encoder hidden layer dimensions.
STANDARD_TOPOLOGIES: dict[str, list[int]] = {
    "small": [256, 128],
    "medium": [512, 256],
    "deep": [512, 256, 128],
}

# Metrics available (from validation only) to rank factorial combinations.
SELECTION_LOSS_METRICS = (
    "validation_total_loss",
    "validation_reconstruction_loss",
    "validation_kl_divergence",
)
SELECTION_NUMERIC_METRICS = (*SELECTION_LOSS_METRICS, "parameter_count", "latent_dimension")
SELECTION_TEXT_METRICS = ("topology_name",)
SELECTION_NUMERIC_DIRECTIONS = ("minimize", "maximize")
SELECTION_TEXT_DIRECTIONS = ("ascending", "descending")
SUPERVISED_EXECUTION_MODES = ("automatic", "manual")
FINAL_EVALUATION_SPLIT = "test"


@dataclass
class ProtocolConfig:
    """Behaviour of the factorial protocol run itself.

    Attributes:
        name: Human-readable protocol name stored in reports and manifests.
        resume_if_interrupted: Allow ``--resume`` on an interrupted run.
        prevent_overwrite: Never write into an existing protocol run directory.
        continue_on_noncritical_evidence_error: Keep a combination whose
            training succeeded even if a projection or figure failed (it is
            then marked ``partial``). When False such a combination is treated
            as failed and excluded from the selection.
    """

    name: str = "factorial_protocol"
    resume_if_interrupted: bool = True
    prevent_overwrite: bool = True
    continue_on_noncritical_evidence_error: bool = True


@dataclass
class TieBreakerConfig:
    """One tie-breaking criterion applied in the order it is listed."""

    metric: str = "validation_reconstruction_loss"
    direction: str = "minimize"


def _default_tie_breakers() -> list[TieBreakerConfig]:
    """Return the documented tie-breaking order."""
    return [
        TieBreakerConfig("validation_reconstruction_loss", "minimize"),
        TieBreakerConfig("parameter_count", "minimize"),
        TieBreakerConfig("latent_dimension", "minimize"),
        TieBreakerConfig("topology_name", "ascending"),
    ]


@dataclass
class SelectionConfig:
    """Rule used to choose the winning combination from validation results.

    Attributes:
        primary_metric: Validation metric ranked first.
        direction: ``minimize`` or ``maximize``.
        tolerance: Absolute difference under which two primary values count
            as equal, so the tie-breakers decide.
        tie_breakers: Ordered criteria applied among tied combinations.
    """

    primary_metric: str = "validation_total_loss"
    direction: str = "minimize"
    tolerance: float = 1.0e-6
    tie_breakers: list[TieBreakerConfig] = field(default_factory=_default_tie_breakers)

    def __post_init__(self) -> None:
        """Normalize tie-breakers given as dictionaries."""
        self.tie_breakers = [
            TieBreakerConfig(**item) if isinstance(item, dict) else item
            for item in self.tie_breakers
        ]


@dataclass
class FactorialComparisonConfig:
    """Full factorial grid: topologies x latent dimensions.

    Attributes:
        enabled: Whether the factorial protocol may run.
        topologies: Standard topology names (see ``STANDARD_TOPOLOGIES``).
        latent_dimensions: Latent dimensions, each trained for every topology.
        generate_full_evidence: Generate every evidence category per combination.
        selection: Winner selection rule (validation only).
    """

    enabled: bool = False
    topologies: list[str] = field(default_factory=lambda: list(STANDARD_TOPOLOGIES))
    latent_dimensions: list[int] = field(default_factory=lambda: [2, 4, 8, 16, 32])
    generate_full_evidence: bool = True
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    def __post_init__(self) -> None:
        """Normalize the nested selection block."""
        if isinstance(self.selection, dict):
            self.selection = SelectionConfig(**self.selection)


@dataclass
class FinalEvaluationConfig:
    """Single evaluation of the selected configuration on the test split."""

    enabled: bool = True
    split: str = FINAL_EVALUATION_SPLIT
    use_selected_configuration: bool = True
    use_best_checkpoint: bool = True
    retrain_selected_configuration: bool = False


@dataclass
class SupervisedExtensionConfig:
    """Supervised extension trained with the selected configuration.

    Attributes:
        enabled: Run (or prepare) the extension.
        execution_mode: ``automatic`` trains it inside the protocol;
            ``manual`` only writes the resolved configuration.
        use_selected_configuration: Reuse the selected topology and dimension.
        classification_loss_weight: Weight of the classification loss.
        use_latent_mean: Classify from ``latent_mean`` instead of a sample.
    """

    enabled: bool = False
    execution_mode: str = "automatic"
    use_selected_configuration: bool = True
    classification_loss_weight: float = 1.0
    use_latent_mean: bool = True


@dataclass
class ExperimentConfig:
    """Root configuration object combining all sub-configurations."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    splits: SplitsConfig = field(default_factory=SplitsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    supervised_component: SupervisedComponentConfig = field(
        default_factory=SupervisedComponentConfig
    )
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    reproducibility: ReproducibilityConfig = field(default_factory=ReproducibilityConfig)
    projections: ProjectionsConfig = field(default_factory=ProjectionsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    plotting: PlottingConfig = field(default_factory=PlottingConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    latent_dimension_comparison: LatentDimensionComparisonConfig = field(
        default_factory=LatentDimensionComparisonConfig
    )
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    factorial_comparison: FactorialComparisonConfig = field(
        default_factory=FactorialComparisonConfig
    )
    final_evaluation: FinalEvaluationConfig = field(default_factory=FinalEvaluationConfig)
    supervised_extension: SupervisedExtensionConfig = field(
        default_factory=SupervisedExtensionConfig
    )
    run_name: str | None = None

    @property
    def experiment_type(self) -> str:
        """Return 'supervised' or 'unsupervised' based on the supervised head."""
        return "supervised" if self.supervised_component.enabled else "unsupervised"

    def __post_init__(self) -> None:
        """Normalize all nested config objects from dicts."""
        if isinstance(self.project, dict):
            self.project = ProjectConfig(**self.project)
        if isinstance(self.paths, dict):
            raw = self.paths
            self.paths = PathsConfig(**raw)
        if isinstance(self.dataset, dict):
            self.dataset = DatasetConfig(**self.dataset)
        if isinstance(self.splits, dict):
            self.splits = SplitsConfig(**self.splits)
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.loss, dict):
            self.loss = LossConfig(**self.loss)
        if isinstance(self.supervised_component, dict):
            self.supervised_component = SupervisedComponentConfig(**self.supervised_component)
        if isinstance(self.optimizer, dict):
            self.optimizer = OptimizerConfig(**self.optimizer)
        if isinstance(self.scheduler, dict):
            self.scheduler = SchedulerConfig(**self.scheduler)
        if isinstance(self.training, dict):
            self.training = TrainingConfig(**self.training)
        if isinstance(self.early_stopping, dict):
            self.early_stopping = EarlyStoppingConfig(**self.early_stopping)
        if isinstance(self.runtime, dict):
            self.runtime = RuntimeConfig(**self.runtime)
        if isinstance(self.reproducibility, dict):
            self.reproducibility = ReproducibilityConfig(**self.reproducibility)
        if isinstance(self.projections, dict):
            self.projections = ProjectionsConfig(**self.projections)
        if isinstance(self.metrics, dict):
            self.metrics = MetricsConfig(**self.metrics)
        if isinstance(self.plotting, dict):
            self.plotting = PlottingConfig(**self.plotting)
        if isinstance(self.artifacts, dict):
            self.artifacts = ArtifactConfig(**self.artifacts)
        if isinstance(self.logging, dict):
            self.logging = LoggingConfig(**self.logging)
        if isinstance(self.latent_dimension_comparison, dict):
            self.latent_dimension_comparison = LatentDimensionComparisonConfig(
                **self.latent_dimension_comparison
            )
        if isinstance(self.protocol, dict):
            self.protocol = ProtocolConfig(**self.protocol)
        if isinstance(self.factorial_comparison, dict):
            self.factorial_comparison = FactorialComparisonConfig(**self.factorial_comparison)
        if isinstance(self.final_evaluation, dict):
            self.final_evaluation = FinalEvaluationConfig(**self.final_evaluation)
        if isinstance(self.supervised_extension, dict):
            self.supervised_extension = SupervisedExtensionConfig(**self.supervised_extension)
