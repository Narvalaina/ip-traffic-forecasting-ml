# Data Provenance and Artifact Boundary

## Source dataset

The principal traffic dataset is **UGR'16**. Dataset payloads are not
redistributed by this repository.

## Capture roles

The final methodological roles are:

```text
March Week #3
→ development / screening

April Week #3
→ configuration and representative selection

June Week #3
→ principal external / blind evaluation

June Week #4
→ complementary training-history sensitivity (Phase G)
```

These roles are part of the final scientific governance. In particular, June
Week #4 must not be reinterpreted as a new global-selection stage.

## Public artifacts

The curated repository keeps selected:

- June-blind metric summaries;
- residual/inference summaries;
- cross-family analysis outputs;
- stability/cost/provenance information;
- selected manifests;
- selected public figures;
- Phase G summary artifacts;
- final research runners and two cleared notebooks.

## Intentionally omitted artifacts

The public repository does not contain:

- raw UGR'16 archives;
- `data/raw/`, `data/interim/` or `data/processed/` payloads;
- aligned prediction Parquet files;
- `results/predictions/` payloads;
- model checkpoints;
- full Colab recovery bundles;
- internal governance/handoff archives;
- historical virtual environments;
- academic manuscript packages;
- paper-specific ICEET assets.

This boundary reduces data leakage, avoids accidental redistribution and keeps
the repository focused on the final engineering/scientific workflow rather
than on academic-project archaeology.

## Sanitized provenance

Historical result files may preserve placeholders such as `<PROJECT_ROOT>` or
other sanitized provenance markers. These are intentional: they retain
traceability while removing private machine paths.

Scientific values, model identifiers, metrics, seeds and final model-selection
decisions were not changed for GitHub reconstruction.
