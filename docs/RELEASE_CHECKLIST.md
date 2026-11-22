# Local pre-publication checklist

- Install the pinned dependencies, using an authorized wheelhouse for offline setup.
- Run `python -m pytest -q` (no model or network calls).
- Run `python release_tools/export_synthetic_data.py --check`.
- Run `python release_tools/check_release.py` and manually inspect its scope.
- Build and inspect a wheel/source distribution. Keep build artifacts private.
- Commit only the public `co-agentfuzz` root, not a parent folder, private archive or backup.
- Check staged files for keys, private inputs/outputs, absolute host paths and weights.
- Preserve the existing MIT notices and baseline attribution in `THIRD_PARTY.md`.
- Obtain contributor approval and verify upstream provenance. The original upstream
  commit is not recoverable from the retained checkout alone.
- Assign a release tag and archive the exact source revision used for each claim.
- Record model/runtime digests, quantization, hardware, cost boundaries and missing runs.
- Do not imply every named model adapter has produced a successful full cohort.

This organization task does not create a remote repository, upload content, assign
a DOI, resolve third-party license permissions or guarantee acceptance by a journal.
