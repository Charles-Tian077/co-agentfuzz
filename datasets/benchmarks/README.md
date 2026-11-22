# Historical prompt-search panels

[Dataset catalog](../README.md) · [Public fictional records](../synthetic/README.md) · [Experiment guide](../../README.md#experiments)

This guide describes the separate hijacking and message-extraction collection.
The historical raw search panels and original run ledgers are not included in
the public source release.

## Reported comparison

The shared Qwen2.5-7B comparison used 80 archived initial seeds and 20 held-out
defenses per task. Each method received 400 focus target queries per task and
run. The target-output cap was 128 tokens. The follow-up used three runs.

Two-stage PromptFuzz additionally used preparation defenses after overlap
filtering. The preparation panels contained 48 hijacking defenses and
50 extraction defenses. Preparation calls and tokens are accounted for
separately from the matched focus budget.

The paper reports the full task results and resource boundary. This guide does
not supply a substitute panel or claim a new reproduction of those results.

## Using historical inputs

The retained legacy implementation requires explicitly supplied, authorized
inputs and the matching original protocol. See
[reproducibility and version boundaries](../../docs/REPRODUCIBILITY.md) and
[data and artifact availability](../../docs/DATA_AVAILABILITY.md).

Public fictional records are not a replacement for a missing historical panel.
They measure the controlled feedback mechanism. External raw benchmarks and
model weights retain their own terms.

## What is publicly available here

- The [fixed fictional-record fixtures](../synthetic/README.md).
- The [historical search controllers](../../src/co_agent_fuzz/).
- The [retained PromptFuzz compatibility implementation](../../src/PromptFuzz/).
- The [model and experiment documentation](../../README.md).

The [source-provenance notice](../../THIRD_PARTY.md) records inherited components.
