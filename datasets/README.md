# Datasets

`synthetic/` is a deterministic, inspectable export of the existing fictional
record fixtures: 18 training records and 60 held-out records, balanced across
contact, inventory and meeting families. Training and test source texts are
disjoint. The test partition uses offset 16; it is not the earlier offset-6 cohort.

Each JSONL row contains `case_id`, `kind`, `source`, `expected`, `phase`, and
`intervention`. `source` is the fictional input. `expected` is a verifier-only
answer key: never append it to model prompts, selection state or peer feedback.
The intervention names describe the existing six controlled draft fault types.
They are not external attack prompts.

The published runtime continues to construct its tasks from the built-in
definitions; it does not load arbitrary JSONL inputs. The exported files are
reviewable fixtures, not a new configurable attack-data interface.

Verify the export locally, without a model:

```bash
python release_tools/export_synthetic_data.py --check
```

`manifest.json` records file/source SHA-256 values and the original runtime
partition digest. Unit tests reconcile every exported row with that runtime.
Generated answers and real model outputs are not included. An offline unit test
does not reproduce the accuracy measured by a particular LLM.

Historical raw inputs and selected benchmark panels are kept outside the public
repository. They are not replaced by this synthetic dataset or covered by an
assumed redistribution permission. Legacy tools require explicitly supplied,
authorized private inputs. `datasets/private/` and `datasets/raw/` are excluded
from source releases.
