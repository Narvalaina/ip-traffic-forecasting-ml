# Limitations

## Dataset redistribution

UGR'16 is not bundled with this repository. Full end-to-end reproduction
requires obtaining the dataset separately under the terms of its original
source.

The absence of dataset payloads is intentional, not an incomplete upload.

## No production-system claim

This repository documents an academic/research forecasting workflow. It is not
presented as a production forecasting service and does not include deployment,
monitoring, online inference or production-SLA infrastructure.

## No universal model winner

The final TFM does not support a context-free ranking in which one family wins
every horizon and metric. Results depend on horizon, metric, temporal regime,
training history, stability and computational cost.

The competitiveness of ARIMA and VAR in the principal external evaluation
should not be generalized into universal superiority.

## Runtime heterogeneity

The historical workflow combined WSL2 CPU execution and Google Colab GPU
execution. Raw timing values from different hardware/runtime regimes should
not be interpreted as a universally fair speed ranking.

The public CPython 3.12 environment is a portability baseline, not a byte-for-
byte recreation of every historical runtime.

## PyTorch / GPU portability

Historical Colab evidence includes PyTorch 2.11.0+cu128, CUDA 12.8 and cuDNN
91900. The public requirements intentionally use portable `torch==2.11.0`.
GPU installation details therefore depend on the target platform.

## Colab-specific tooling

The cleared notebooks may contain Colab-specific `google` tooling. It is
treated as runtime tooling rather than a core scientific dependency of the
portable WSL/CPython environment.

## Validation scope

GitHub reconstruction validated syntax, notebook structure, imports, paths,
selected artifact formats and dependency availability without re-running the
scientific campaigns.

No model was retrained, retuned or reselected to make the public repository
look better.

## Rights

The repository currently has no open-source license. Dataset rights, academic
document rights, related-paper rights and third-party dependency licenses must
be considered separately. See [`../RIGHTS.md`](../RIGHTS.md).
