# Co-AgentFuzz datasets

[Browse the records](synthetic/README.md) · [Training split](synthetic/train.jsonl) · [Test split](synthetic/test.jsonl) · [Schema](synthetic/schema.json) · [Integrity manifest](synthetic/manifest.json)

## Choose the relevant data

| Collection | Purpose | Public contents |
|:--|:--|:--|
| [Fixed fictional records](synthetic/README.md) | Inspect and reproduce the controlled generation, repair and selection study | 18 training records, 60 held-out records, field schema and SHA-256 manifest |
| [Historical search panels](benchmarks/README.md) | Understand the input boundary of the Co-AgentFuzz–PromptFuzz search comparison | Protocol and availability guide. Historical raw panels are not included in this release |

The fictional records measure the feedback mechanism. Their exact-match accuracy
is distinct from prompt-injection or jailbreak success. The historical search
panels are a separate collection.

## Directory guide

```text
datasets/
├── README.md                  # This catalog and entry point
├── synthetic/
│   ├── README.md              # Data card, split counts and usage
│   ├── train.jsonl            # 18 records
│   ├── test.jsonl             # 60 held-out records, phase = eval
│   ├── schema.json            # JSON Schema for each JSONL row
│   └── manifest.json          # File hashes, source hashes and partition digest
└── benchmarks/
    └── README.md              # Historical search protocol and availability
```

The existing `synthetic/train.jsonl`, `synthetic/test.jsonl` and
`synthetic/manifest.json` paths remain stable. The records and their original
checksums are unchanged by this organization update.

## Start here

1. Read the [fixed-record data card](synthetic/README.md).
2. Inspect or download the [training](https://raw.githubusercontent.com/Charles-Tian077/co-agentfuzz/main/datasets/synthetic/train.jsonl) and [test](https://raw.githubusercontent.com/Charles-Tian077/co-agentfuzz/main/datasets/synthetic/test.jsonl) files.
3. Run this check from the repository root.

```bash
python release_tools/export_synthetic_data.py --check
```

The check compares the exported bytes against the built-in task definitions.
It performs no inference and requires no model download.

For execution, see [model deployment](../README.md#model-deployment),
[experiments](../README.md#experiments) and
[protocol/version boundaries](../docs/REPRODUCIBILITY.md).

## Evaluation labels and reuse

`expected` is the verifier-only answer key. It stays outside model prompts,
selection inputs and peer messages. The runtime constructs its tasks from the
source definitions rather than loading arbitrary JSONL files.

The test split uses evaluation offset 16. Earlier offset-6 cohorts retain their
own data and result boundaries. Model outputs, historical run ledgers, model
weights and private benchmark inputs are outside this dataset release.

The repository [license](../LICENSE.txt) and [source-provenance notice](../THIRD_PARTY.md)
describe the public source release. External benchmarks and checkpoints retain
their own terms.
