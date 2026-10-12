# Evaluation

This document describes how the latent space and the model are evaluated, what
each output means and what it cannot tell you. For how to read the results see
[results_interpretation.md](results_interpretation.md).

## 1. What the standard `train` pipeline evaluates

`ExperimentRunner` evaluates on the **validation split** after reloading the
best checkpoint:

1. reconstruction and generated-sample figures,
2. latent extraction (`latent_data/`),
3. projections (direct 2D, or PCA/UMAP/t-SNE),
4. label-based clustering metrics in the original latent space
   (`tables/projection_metrics.csv`).

The linear classifier, reconstruction MSE/BCE numbers, K-Means ARI/NMI and
trustworthiness are implemented but are **not** part of that pipeline:
`VAEEvaluator` (linear probe, reconstruction metrics) and
`run_final_evaluation` (test reconstruction metrics and clustering metrics)
must be called from Python.

## 2. Latent extraction

[`extract_latent_representations`](../src/vae_project/evaluation/latent_extraction.py)
puts the model in `eval()` mode, runs the loader in order and stores, for
each image:

* `latent_mean` (deterministic encoding; used for everything below),
* `latent_log_variance`,
* `latent_sample`: an explicit draw `mu + eps * exp(0.5 * log_var)`. Because
  `eval()` returns the mean from `reparameterize`, the extraction code
  samples separately; these draws are random and **not seeded** by the
  function, so they change between calls unless the global RNG state is
  identical.

Saved to `latent_data/latent_representations.csv`, with columns
`sample_index`, `split`, `label`, `class_name`, `latent_mean_0..L-1`,
`latent_log_var_0..L-1`, `latent_sample_0..L-1` (and `classifier_prediction`
when a head is supplied), and to `.npz` when `artifacts.save_latent_npz` is
true (arrays: `sample_indices`, `labels`, `latent_means`,
`latent_log_variances`, `latent_samples`). `sample_index` is the position in
the loader order of the split, not the original dataset index.

Labels are attached for colouring and metrics; they are not inputs to the
unsupervised model.

## 3. Projections

All projections use `latent_mean` of a class-balanced subsample of at most
`projections.sampling.max_samples` (default 5,000) validation points
(`stratified_subsample`; the chosen positions are saved in
`latent_data/projection_sample_indices.csv`). With the default 6,000 validation
images and 10 classes, 500 points per class are drawn when a class has enough
samples, which gives 5,000 points. The subsample is class-balanced rather than
proportional, so the plotted class frequencies may differ from the true ones.

If `latent_dimension == 2`, the 2D latent means are plotted directly (up to
`plotting.max_display_points`) and no PCA/UMAP/t-SNE is run.

### PCA and explained variance

PCA is linear. `pca_explained_variance.csv` stores, for the first two
components (the runner fits two components), `explained_variance_ratio` and
the cumulative value. This is the fraction of the **total linear variance of
the latent means** captured. It is not a class-separation measure: a latent
space can have a large first component unrelated to the classes and a small
one that separates them. Because only two components are fitted, the ratio
does not show how the variance is spread over the remaining dimensions. The
figure `pca_explained_variance_<L>d` shows the bars and the annotation "Top-2
components: X% of linear variance".

### UMAP and t-SNE limitations

* Neither gives explained variance; never report one.
* Both preserve **local** neighbourhoods; global layout, inter-cluster
  distances, cluster sizes and densities are not reliable.
* Results depend on hyperparameters (`n_neighbors`, `min_dist`; `perplexity`,
  `init`, iterations) and random seed; the same data can look more or less
  clustered with other settings.
* They can show apparent clusters that are not separable in the original
  latent space, and can fragment a continuous distribution.
* t-SNE cost and memory grow quickly with N; UMAP needs `umap-learn`.
* Remedy: report parameters and n, compare against PCA, and vary parameters
  as a sensitivity check.

The figure titles include method, dataset, latent dimension, split and n, and
UMAP/t-SNE figures carry an on-figure note about the limitations.

## 4. Clustering metrics (subsidiary)

[`compute_clustering_metrics`](../src/vae_project/evaluation/clustering_metrics.py)
uses the **ground-truth labels as cluster assignments** on `latent_mean`
(full latent space, not the 2D projection):

| Column in `projection_metrics.csv` | Meaning |
|---|---|
| `space_name` | `val_latent_original` in the standard pipeline. |
| `silhouette_score` | [-1, 1]; higher = tighter, better separated labelled groups. Computed on at most `silhouette_sample_size` random points. |
| `calinski_harabasz_index` | Higher = larger between-group vs within-group dispersion. Uses all samples. |
| `davies_bouldin_index` | Lower = better separated groups. Uses all samples. |
| `trustworthiness` | Empty in the standard pipeline (needs a reference high-dimensional space). |
| `kmeans_ari`, `kmeans_nmi` | Empty in the standard pipeline (K-Means is off by default). |
| `n_samples`, `n_samples_total`, `was_subsampled` | Sample accounting for the silhouette. |

These numbers describe agreement between labels and geometry. They do not
establish that natural clusters exist, are sensitive to the latent
dimension, and are not comparable across different spaces without care. If
ARI/NMI are enabled, treat them as exploratory (K-Means makes spherical-cluster
assumptions and its result depends on initialisation). They never replace
the visual analysis.

## 5. Linear classifier (probe)

`train_linear_classifier` fits a multinomial logistic regression on
standardised **training** latent means (scaler fitted on train only), and
reports train and validation accuracy, plus test accuracy only if test arrays
are supplied; it can also return a confusion matrix and a classification
report. It measures **linear separability** of the labels, which is neither
necessary nor sufficient for cluster structure: classes can be linearly
separable without forming compact clusters, and clustered classes can be
non-linearly arranged. It is invoked by `VAEEvaluator` when
`metrics.compute_linear_classifier` is true.

## 6. Reconstruction quality

* **Training/validation losses**: per-epoch reconstruction and KL terms in
  `training_history.csv`. They depend on loss type, `reduction` and beta.
* **Visual**: `reconstruction_comparison` (8 examples) compares original and
  reconstruction from the posterior mean (eval mode), and `generated_samples`
  shows decodings of prior samples.
* **Numeric**: `compute_reconstruction_metrics` returns `mean_mse` and
  `mean_bce`, each averaged over batches of per-element means (so on the
  per-pixel scale, not the summed loss used in training). The value is
  computed with the posterior mean as decoder input. It is reported in the
  final evaluation JSON.

Good reconstruction does not imply an informative latent space for classes,
and vice versa.

## 7. Validation vs test evaluation

Validation evaluation is safe during model selection. The test evaluation
(`run_final_evaluation`) is reserved for the end; its output is labelled
`evaluation_type: final_test`. The final evaluation function reads the model
configuration from the YAML you pass; use the configuration that produced the
checkpoint, otherwise loading fails or silently compares the wrong
architecture.

## 8. Evaluation checklist

* Which split was each number computed on?
* Subsampled or not (`was_subsampled`, `n`)?
* Which latent representation (mean vs sample)?
* Which projection parameters?
* Is the metric label-based?
* Was any decision made using test data?
