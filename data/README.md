# Data

## Availability

The **UGR'16** dataset used by the Master's Thesis is **not redistributed** in
this repository. No raw, interim or processed dataset payload is committed,
and aligned prediction payloads are also excluded.

Obtain UGR'16 from its original distribution source under the applicable
access and usage terms. This repository intentionally does not mirror the
dataset or assert redistribution rights.

## Temporal captures used

| Capture | Role in the final methodology |
| --- | --- |
| March Week #3 | Development and screening |
| April Week #3 | Configuration and representative selection |
| June Week #3 | Principal external / blind evaluation |
| June Week #4 | Complementary training-history sensitivity experiment |

June Week #4 belongs to the complementary Phase G analysis and does not reopen
the main model-selection process.

## Expected local data layout

The research scripts refer to local working stages such as:

```text
data/
├── raw/
├── interim/
└── processed/
```

These directories are deliberately ignored by Git and are expected to be
created locally as needed. Their payloads must not be committed to the public
repository.

Some evaluation scripts also refer to local prediction products under
`results/predictions/`. Those payloads are intentionally not published.

## Provenance

The curated manifests and result files retain sanitized provenance references
and scientific hashes where useful. Personal machine paths were removed during
repository reconstruction.

See [`../docs/data_provenance.md`](../docs/data_provenance.md) for the role of
each capture and the public/private artifact boundary.
