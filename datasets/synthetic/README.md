# Fixed fictional-record dataset

**Dataset ID** `co-agentfuzz-fictional-repair-v1`

[Dataset catalog](../README.md) · [Train JSONL](train.jsonl) · [Test JSONL](test.jsonl) · [Row schema](schema.json) · [Manifest](manifest.json)

This is a deterministic export of the existing controlled record fixtures.
Each row is one fixed fictional task with a reference answer and a controlled
draft intervention. These are input fixtures, not generated model outputs.

## Splits and statistical unit

| Split file | `phase` | Contact | Inventory | Meeting | Total |
|:--|:--|--:|--:|--:|--:|
| `train.jsonl` | `train` | 6 | 6 | 6 | 18 |
| `test.jsonl` | `eval` | 20 | 20 | 20 | 60 |

Training and test source texts are disjoint. All 78 case identifiers are unique.
The evaluation offset is 16. The test identifiers run from 16 to 35 within each
record family.

The retained seeds are 101, 137 and 173. A test record is the independent unit.
Seed outcomes are aggregated within that record. Three runs over the same
60 records do not create 180 independent tasks.

## Field definitions

| Field | Type | Meaning |
|:--|:--|:--|
| `case_id` | string | Stable identifier containing the phase, family and index |
| `kind` | string | `contact`, `inventory` or `meeting` |
| `source` | string | Fictional source text available to the role |
| `expected` | object of strings | Reference fields used only by the external verifier |
| `phase` | string | `train` for training, `eval` for held-out evaluation |
| `intervention` | string | Controlled transformation of the generated draft before revision |

The reference keys are `name` and `city` for contact, `item` and `color` for
inventory, and `day` and `location` for meeting records. The
[JSON Schema](schema.json) specifies the allowed fields, types and identifiers.

## Draft interventions

| Label | Meaning | Training rows | Test rows |
|:--|:--|--:|--:|
| `clean` | Leave the generated draft unchanged | 3 | 12 |
| `missing` | Remove the last key in sorted order | 3 | 12 |
| `extra` | Add the key `unused` with value `fictional` | 3 | 9 |
| `wrong_value` | Set the selected field to `fictional_mismatch` | 3 | 9 |
| `wrong_type` | Set the selected field to integer `0` | 3 | 9 |
| `syntax` | Remove the final `}` when present, otherwise append `{` | 3 | 9 |

Field edits use the last key in sorted order for a nonempty parsed object.
They leave an unparseable or empty object unchanged. The `clean` label adds no
fault and does not guarantee a correct model draft.

The intervention acts on a generated draft. It does not change the source text
or the verifier reference. The label is not a model outcome or an attack prompt.

## Inspect the files

A training-row example follows. The reference is shown for auditing.

```json
{"case_id":"train-contact-00","kind":"contact","source":"name: ava. city: lima.","expected":{"city":"lima","name":"ava"},"phase":"train","intervention":"clean"}
```

Read the files after cloning the repository.

```python
import json
from pathlib import Path

directory = Path("datasets/synthetic")
train = [json.loads(line) for line in (directory / "train.jsonl").read_text(encoding="utf-8").splitlines()]
test = [json.loads(line) for line in (directory / "test.jsonl").read_text(encoding="utf-8").splitlines()]
assert len(train) == 18 and len(test) == 60
assert not {row["source"] for row in train} & {row["source"] for row in test}
```

Direct file links are available for [training](https://raw.githubusercontent.com/Charles-Tian077/co-agentfuzz/main/datasets/synthetic/train.jsonl)
and [testing](https://raw.githubusercontent.com/Charles-Tian077/co-agentfuzz/main/datasets/synthetic/test.jsonl).
No external data download is needed for these fixtures.

## Integrity and provenance

| File | Bytes | SHA-256 |
|:--|--:|:--|
| `train.jsonl` | 3,380 | `8e7a5c1864e8982a584234258f916708f818cbd05c472d861d5913a2bbc7b730` |
| `test.jsonl` | 11,308 | `91796823a2abbccb3e7206f05da392022602174f7c4309279613928ca84178a6` |

The original partition digest is
`df70ae7b0c9867b914e5a22678cda376f6fdb839b6077e4f095ae6cd4a1d1f90`.
The [manifest](manifest.json) also records source hashes for the task builder,
study harness and feedback implementation.

From the repository root, verify every exported record against those definitions.

```bash
python release_tools/export_synthetic_data.py --check
```

The runtime continues to construct tasks from
[`coagent_repro/replicate_records.py`](../../src/coagent_repro/replicate_records.py).
It does not load arbitrary JSONL inputs. The export supports inspection and
integrity checks. A new model run uses the
[documented experiment entry points](../../README.md#experiments).

## Interpretation and availability

Exact-match accuracy on these fictional records measures controlled generation,
repair and selection. It is distinct from historical hijacking and extraction
success. Earlier data offsets, rewards and inference presets retain separate
result boundaries.

`expected` stays outside model prompts and controller state. Test-time evaluation
freezes learned action values and peer priors. Actual model responses and token
ledgers are generated by a new run and are not included here.

See the [historical-search guide](../benchmarks/README.md),
[reproducibility notes](../../docs/REPRODUCIBILITY.md) and
[data-availability statement](../../docs/DATA_AVAILABILITY.md) for the other
experimental collections.
