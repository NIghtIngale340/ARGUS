# ARGUS Phase 3 Model Card

## Model

ARGUS Phase 3 is a binary attack-vs-normal classifier built on the ARGUS-BERT architecture. The model lineage is:

1. **ARGUS-BERT MLM pretraining** on 58 days of tokenized LANL authentication log sessions.
2. **Supervised binary fine-tuning** on labeled attack/normal sessions.

Architecture: BERT encoder (6 layers, 256 hidden, 8 heads) → [CLS] pooling → dropout → linear classifier (2 classes).

## Intended Use

Local cybersecurity log-intelligence demos and research:

- Score ARGUS sessions as `normal` or `attack`.
- Generate calibrated attack probabilities.
- Feed composite alert severity and Redis UEBA risk updates.
- Support Kafka replay/streaming demonstrations.

## Not Intended Use

- Do not use as a production SOC decision system without additional validation.
- Do not present outputs as real MITRE multi-technique classification.
- Do not use fallback technique metadata as analyst-grade ATT&CK attribution.

## Training Data

- **Source**: LANL Comprehensive Multi-Source Cyber-Security Events dataset (authentication logs).
- **Train split**: days 1–40.
- **Validation split**: days 41–50.
- **Test split**: days 51–58.
- **Token format**: `event_id_auth_type_logon_type` (1233 vocabulary).
- **Sequence length**: 16 tokens per session.

## Evaluation Metrics

Current bundle metrics (same-day eval on test split):

| Metric | Value |
|---|---|
| ROC-AUC | 0.998133 |
| PR-AUC | 0.964293 |
| Operating threshold | 0.999 |
| Precision @ threshold | 0.910798 |
| Recall @ threshold | 0.915094 |
| FPR @ threshold | 0.0019 |
| F1 @ threshold | 0.912941 |
| Best F1 (any threshold) | 0.926829 |

## Bundle Artifact

```text
data/argus_phase3_model_bundle/
  best_classifier.pt
  calibrated_thresholds.json
  vocab.json
  model_metadata.json
```

Bundle zip SHA256: `E25F1A5EBC99337069A98AB6F539F3809B229E673D082027514E5C249F87D44C`

## Caveats

- These metrics are strong same-day eval evidence, not broad held-out production proof.
- The attack set is small and likely training-adjacent. The model has not been validated against novel attack patterns.
- Real production validation would require leakage-resistant held-out data, broader attack coverage, and endpoint-log adapters.
- The model operates on tokenized LANL authentication sessions only. It has not been tested on other log formats.

## Ethical and Security Notes

The model can be wrong. Use it as a detection research artifact and demo platform, not as an autonomous enforcement system. False positives and false negatives are expected in any real deployment scenario.
