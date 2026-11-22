# Reproducibility and version boundaries

## Separate implementations

`src/co_agent_fuzz/` is the retained historical prompt-search controller. Its tests
use mock target clients. `src/benign_record_feedback/` implements the independently
audited fictional-record repair mechanism. `src/benign_llm_orca/` and `src/benign_orca/`
retain earlier transport/record and arithmetic diagnostics. These are not
interchangeable experiments, and compatibility names must not be used to imply
that every archived run used the same rewards or inputs.

The cleanup leaves the core generator/mutator/critic controllers, Q-learning,
record reward rules, peer-prior updates, and deterministic task definitions
unchanged. Relocated wrappers have new source hashes. Use an archived original
source snapshot to audit an old run that pins its original hashes. Do not rewrite
those hashes to match a later release.

## Repair-space replication

- Three roles (generator, editor and selector) work with an external deterministic record verifier.
- Three record families: contact, inventory, meeting.
- 18 training records, six balanced draft intervention types.
- 60 held-out records, beginning at the fixed evaluation offset 16.
- Smoke: the same training protocol with six held-out records.
- Seeds: 101, 137, 173. Each seed/condition starts from its own controller state.
- Full / No learning / No communication / Neither are independently logged.
- Evaluation disables exploration and freezes both action values and peer history.
- LLM weights are never trained by these scripts.

Preserve parse/schema/exactness, dense verifier scores, observed repair opportunities,
candidate disagreement, role rewards, nonzero Q changes, feedback receipts, and the
next decision's consumed state. Real tokens come from final endpoint usage frames.
Failed or incomplete calls are not zero-cost successful calls.

## Model runtime presets

The base wrapper retains its existing 350/1000 call fuse for smoke/replication,
100000/250000 token fuse, and 360/900 second phase envelope. The 32B overlay retains
a 400000-token envelope and 900/1800-second runtime limits. QwQ uses 512 output
tokens per role and model-default thinking, while Qwen3 uses the retained
non-thinking 64/16 limits. These are different compute presets, not an equal-cost
cross-model comparison.

The V3.2 overlay retains the offline Q4_K_M weight manifest and allowed configurations
(2 GPUs, 36 offloaded layers), (4 GPUs, 40 layers), and (4 GPUs, 20 layers).
It records 16 CPU threads, CPU offload, and actual residency. Its bounded phase
envelopes include the historical long cold start. The last retained V3.2 attempt
did not yield a complete three-seed replication. Availability of an adapter is
not evidence of a successful efficacy result.

Keep model digest, quantization, chat template, endpoint version, GPU allocation,
offload preset, context/output limits, and seed constant within a comparison.
Do not silently change precision, reward weights, prompts, or selection schema.

## Matched communication-text probe

The portable probe takes an explicit verified input directory:

```bash
srun --ntasks=1 python -m benign_record_feedback.token_probe \
  --port 11434 --prior /your/private/verified-smoke \
  --output /your/private/new-token-probe
```

It retains the same role inputs/actions and compares appended versus omitted
verified training-history text. It requires the same model digest as its input
run. This is a small matched prompt-text cost probe, not total hardware savings,
accuracy noninferiority, or evidence of attack efficacy.

## Statistics and reruns

For a complete three-seed cohort, aggregate seed observations within each fixed
record before computing paired effects. Resample records for 95% bootstrap
intervals. Do not resample 180 seed observations as independent tasks. Stable
success means correctness on all three seeds. Apply exact paired tests and
model-wise Holm correction only to complete, prespecified contrasts. Retain
unfavorable outcomes and missing-run boundaries.

Archived datasets, source snapshots, and outputs are private inputs, not included
in this source-and-synthetic-fixture repository. The split builders and measurement/audit functions
are retained. They cannot reconstruct a missing historical input selection or
model checkpoint. A new run gets a new output directory and a new source manifest.
