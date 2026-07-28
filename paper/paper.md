---
title: 'N$^2$: A unified Python package and test bench for nearest neighbor matrix completion'
tags:
  - Python
  - matrix completion
  - nearest neighbors
  - causal inference
  - panel data
  - missing data
authors:
  - name: Caleb Chin
    equal-contrib: true
    affiliation: 1
  - name: Aashish Khubchandani
    orcid: 0009-0003-2638-9472
    equal-contrib: true
    affiliation: 1
  - name: Harshvardhan Maskara
    affiliation: 1
  - name: Kyuseong Choi
    orcid: 0000-0002-3380-2849
    affiliation: 1
  - name: Jacob Feitelberg
    orcid: 0000-0002-4551-0245
    corresponding: true
    affiliation: 2
  - name: Albert Gong
    orcid: 0009-0005-1687-0240
    affiliation: 1
  - name: Manit Paul
    orcid: 0000-0002-1735-3289
    affiliation: 3
  - name: Tathagata Sadhukhan
    orcid: 0009-0001-1549-1469
    affiliation: 1
  - name: Anish Agarwal
    affiliation: 2
  - name: Raaz Dwivedi
    orcid: 0000-0002-9993-8554
    affiliation: 1
affiliations:
  - name: Cornell University, USA
    index: 1
    ror: 05bnh6r87
  - name: Columbia University, USA
    index: 2
    ror: 00hj8s172
  - name: University of Pennsylvania, USA
    index: 3
    ror: 00b30xv10
date: 27 July 2026
bibliography: paper.bib
---

# Summary

Many scientific datasets can be arranged as a matrix whose rows are units (patients,
users, states) and whose columns are conditions (time points, movies, prompts), where
only some of the entries are actually observed. *Matrix completion* is the task of
filling in the missing entries, and it underpins recommendation systems, counterfactual
inference in panel data, and the evaluation of large language models.

`N`$^2$ is a Python package that unifies the fast-growing family of *nearest neighbor*
(NN) methods for matrix completion behind a single interface. It implements row-wise and
column-wise NN [@li2019nearest; @dwivedi2022counterfactual], two-sided NN
[@sadhukhan2024adaptivity], doubly robust NN [@dwivedi2022doubly], adaptively-weighted NN
[@sadhukhan2025adaptively], and kernel- and Wasserstein-based *distributional* NN
[@choi2024learning; @feitelberg2024distributional], together with cross-validation
routines for their tuning parameters and with two classical non-NN baselines,
universal singular value thresholding [@chatterjee2015matrix] and SoftImpute
[@hastie2015matrix].

Beyond scalar matrices, `N`$^2$ supports *distributional* matrix completion, in which
each entry is an empirical distribution (for example, the full distribution of a
patient's step counts in an hour) rather than a single number. Distributional entries
are treated as first-class citizens, so the same estimator code works for both settings.

`N`$^2$ also ships **N$^2$-Bench**, a benchmark harness with automatic data loaders for
four real-world datasets spanning mobile health [@klasnja2019efficacy], recommender
systems [@movielens], causal panel data [@abadie2010synthetic], and large language model
evaluation [@polo2024efficient], plus a configurable synthetic data generator. Each
loader returns a data matrix, a missingness mask, and a held-out ground truth, so a new
method can be evaluated across all four domains without writing dataset-specific code.

# Statement of need

Nearest neighbor methods have re-emerged as a competitive and theoretically well-understood
approach to matrix completion. Unlike spectral and nuclear-norm methods, which typically
require missingness to be completely at random, NN methods impute one entry at a time by
matching on observed rows or columns, which makes them robust when missingness is driven
by the entries themselves or by unobserved confounders [@agarwal2023causal;
@abadie2024doubly]. Recent work has established entry-wise error bounds, valid confidence
intervals, and minimax optimality for several variants.

That progress has come at a cost: each new variant has arrived as a standalone research
implementation, with its own data conventions, its own hyperparameter interface, and its
own evaluation script. Practitioners who want to know which variant suits their data must
reimplement or reconcile several codebases, and researchers proposing a new variant have
no shared baseline against which to compare. There is also no common testbed — NN methods
for matrix completion are usually validated on synthetic low-rank matrices with uniform
missingness, precisely the regime in which their advantages over classical methods are
least visible.

`N`$^2$ addresses both gaps. Switching between NN variants is a one-line change, so an
applied researcher can compare methods on their own data without rewriting anything, and
a methodologist can add a new variant by implementing a single class and inherit the full
benchmark suite, cross-validation machinery, and baselines for free. The package targets
statisticians and machine learning researchers working on matrix completion and
counterfactual inference, and applied researchers in health, economics, and
recommendation who need to impute structured missing data.

# State of the field

The closest widely used tool is `scikit-learn`'s `KNNImputer` [@scikit-learn]. It is
designed for the feature-matrix setting, so neighbors are defined only across samples
(row-wise), the target is always a scalar, and there is no facility for the two-sided,
doubly robust, or adaptively-weighted estimators that dominate the recent literature.
`fancyimpute` provides SoftImpute and related low-rank solvers but no NN variants beyond
a basic k-NN imputer. Domain-specific packages exist for individual estimators — for
example, `syntheticNN` implements synthetic nearest neighbors for causal matrix
completion — but each covers one method and one problem framing.

No existing package covers distributional matrix completion, where entries are
probability distributions and the imputation target is a barycenter under a Wasserstein
or maximum mean discrepancy geometry. `N`$^2$ is, to our knowledge, the first package to
provide these estimators, and the first to place scalar and distributional estimators
under one abstraction so that both can be benchmarked on the same footing. We chose to
build rather than contribute upstream because the required abstraction — decoupling *what
kind of object lives in a cell* from *how neighbors are combined* — is incompatible with
`scikit-learn`'s transformer API, which assumes a numeric feature matrix and row-wise
neighbor structure throughout.

# Software design

The design separates two concerns that prior implementations entangle.

A `DataType` implements the geometry of the entries: a `distance` between two entries and
an `average` of a set of entries. `Scalar` uses squared distance and the arithmetic mean;
`DistributionKernelMMD` uses a U-statistic estimate of the squared maximum mean
discrepancy and the corresponding kernel barycenter; `DistributionWassersteinSamples` and
`DistributionWassersteinQuantile` use the 2-Wasserstein distance and its barycenter.

An `EstimationMethod` implements the estimator: which neighbors to use, how to weight
them, and how to compose partial estimates. `RowRowEstimator`, `ColColEstimator`,
`TSEstimator`, `DREstimator`, `AWNNEstimator`, and `AutoEstimator` are all expressed
against the abstract `DataType` interface and therefore work with any entry geometry.

A user composes the two into a `NearestNeighborImputer` — an instance of the *Composite*
pattern [@gamma1993design] — and calls `impute(row, column, data_array, mask_array)`. The
practical consequence is that the two axes multiply rather than add: adding one new
`DataType` (say, text embeddings or images) immediately yields every implemented
estimator for that data type, and adding one new estimator immediately yields it for
scalars and all distributional geometries. The one deliberate exception is doubly robust
estimation, which requires a well-defined subtraction and so is restricted to entry
spaces that are vector spaces.

Hyperparameter selection is factored out into a parallel `FitMethod` hierarchy
(`LeaveBlockOutValidation`, `DualThresholdLeaveBlockOutValidation`, and estimator-specific
subclasses) that holds out a block of observed entries and searches distance thresholds
with `hyperopt` [@bergstra2013hyperopt]. Because tuning is separate from estimation,
cross-validation strategies and estimators can be mixed freely. Benchmark datasets are
registered through a decorator-based factory (`@register_dataset`), so contributing a new
dataset means writing one loader class rather than modifying the harness. All numerical
work is built on NumPy [@harris2020array].

# Research impact

`N`$^2$ is the reference implementation for a line of methodological work on nearest
neighbor matrix completion: the two-sided [@sadhukhan2024adaptivity], adaptively-weighted
[@sadhukhan2025adaptively], kernel distributional [@choi2024learning], and Wasserstein
distributional [@feitelberg2024distributional] estimators are all implemented here by
their original authors, and the accompanying benchmark study [@chin2025nsquared] uses the
package for all reported experiments.

The benchmark itself has already produced a methodological finding: on N$^2$-Bench's
real-world tasks, classical spectral and optimization-based methods that dominate on
synthetic low-rank matrices are frequently outperformed by NN variants, and distributional
NN methods outperform every scalar method on the HeartSteps task — evidence that
conclusions drawn from idealized synthetic experiments can be misleading. The package is
released on PyPI as `nsquared` under the MIT license, has been developed publicly since
December 2024 across more than 450 commits from seven contributors, and ships the shell
scripts and data loaders needed to reproduce every experiment in
[@chin2025nsquared].

# AI usage disclosure

Generative AI coding assistants were used, under author direction, to help draft portions
of this paper and of the package's user-facing documentation. All AI-assisted text was
read, edited, and verified for technical accuracy by the authors, and every factual claim
about the software was checked against the source code and test suite. The statistical
methods, their implementations, and the benchmark experiments were designed and written
by the authors.

<!-- ACTION REQUIRED BEFORE SUBMISSION: all co-authors must confirm this disclosure is
complete. If generative AI was used elsewhere (e.g. to write or refactor library source,
tests, or experiment scripts), amend the paragraph above to say where it was used and how
the output was verified. If it was used nowhere else, the paragraph is already accurate. -->

# Acknowledgements

Jacob Feitelberg and Anish Agarwal were supported by the Columbia Center for AI and
Responsible Financial Innovation in collaboration with Capital One. Albert Gong's work was
partially supported by funding from NewYork-Presbyterian for the NYP-Cornell
Cardiovascular AI Collaboration.

# References
