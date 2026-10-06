# Methodology

## Objective

The project studies multihorizon forecasting of IP-network traffic and
compares heterogeneous model families under a temporally separated
experimental protocol.

The goal is not to declare one universally superior model. Evaluation is
conditioned on forecast horizon, metric, temporal regime, training history,
stability and computational cost.

## Temporal protocol

The final workflow uses distinct UGR'16 captures with distinct methodological
roles:

1. **March Week #3 — development / screening.** Candidate approaches and
   configurations are explored without treating this capture as the final
   external test.
2. **April Week #3 — selection.** Configurations and family representatives
   are selected for later evaluation.
3. **June Week #3 — principal external / blind evaluation.** The selected
   representatives are evaluated on a later capture.
4. **June Week #4 — complementary training-history sensitivity.** Phase G
   studies sensitivity to the amount/history of training data. It is not a
   second global selection stage.

This separation reduces the risk of rewriting the final scientific narrative
around the external evaluation capture.

## Model families

The final project covers:

- persistence and baseline forecasting;
- AR / MA / ARMA;
- ARIMA;
- VAR;
- Ridge;
- tree-based ensembles and boosting;
- RNN;
- LSTM;
- GRU;
- a lightweight Transformer.

SARIMA was explored as part of the broader statistical work but is not used to
claim a universal reference solution.

## Evaluation

The repository preserves selected multihorizon metrics, residual analyses,
paired/sensitivity analyses, cross-family comparisons, stability/cost
information and provenance manifests.

The final interpretation is deliberately conditional rather than a single
leaderboard claim. The June Week #3 evidence particularly highlights the
competitiveness of statistical approaches such as ARIMA and VAR, while model
performance remains dependent on the evaluation context.

## Phase G

Phase G is a complementary experiment on training-history/data-scaling
sensitivity. Its purpose is to study how representative models respond to the
amount/history of training information.

Phase G does **not** reopen the main selection process and is not used to
declare a different global winner.

## Public-repository boundary

The public GitHub derivative preserves the final scientific story. It does not
retrain, retune, reselect or search for a new winner merely to optimize the
repository presentation.
