# SERE Course Project Reproduction

This repository contains the reproducible course-project implementation used by Group 9 for the SERE event-causality-identification experiment.

## Group members

- Wang Lei (王磊)
- Ao Wenjie (敖文杰)
- Liu Jinyu (刘劲宇)

## Scope

The repository provides a resource-constrained reproduction of the SERE retrieval workflow on the 1,000-example EventStoryLine sample. It includes:

- document-level 60/20/20 splitting with seed 42;
- E0 archived GPT-4 zero-shot baseline;
- E1 local structural retrieval with `k=2`;
- E2 single-factor comparison with `k=4`;
- class-balanced retrieval, validation-only parameter selection, metrics and per-example traces.

The local E1/E2 pipeline uses transparent deterministic proxies. It is not presented as a numerical reproduction of the paper's complete GPT-4o-mini, Neo4j/ConceptNet, Contriever and spaCy configuration.

## Run

Python 3.9 or later is sufficient for the course reproduction script.

```powershell
powershell -ExecutionPolicy Bypass -File .\run_experiment.ps1
```

Alternatively:

```powershell
python .\course_reproduction\run_experiment.py --root .
```

Generated metrics and traces are written to `results/`. The checked-in results are the outputs reported in the course report.

## Main results

| Experiment | Precision (%) | Recall (%) | F1 (%) |
|---|---:|---:|---:|
| E0 | 25.81 | 93.02 | 40.40 |
| E1 | 25.50 | 88.37 | 39.58 |
| E2 | 25.85 | 88.37 | 40.00 |

## Sources and attribution

- SERE paper and official implementation: <https://github.com/DMIRLAB-Group/SERE>
- Archived ESC sample and GPT-4 zero-shot responses: <https://github.com/ArrogantL/ChatGPT4CausalReasoning>
- The included upstream sample and response archive retain the Apache License 2.0 notice under `THIRD_PARTY_LICENSES/`.

No API keys, virtual environments, model weights or full external knowledge bases are included.
