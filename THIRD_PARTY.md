# Source provenance and licensing

The existing MIT notice is retained byte-for-byte in `LICENSE.txt` and in
`src/PromptFuzz/Fuzzer/LICENSE`. Do not remove or replace the upstream copyright notice.

`src/PromptFuzz/` contains the inherited PromptFuzz implementation, including its
GPTFuzzer components. It is retained as a compatibility baseline, not presented
as newly authored Co-AgentFuzz code. The local checkout has no Git history, so an
exact upstream commit cannot be certified from this checkout alone. Future releases
should record the source URL and commit when that provenance is available.

Co-AgentFuzz controllers and experimental additions are in `src/co_agent_fuzz/`,
`src/benign_record_feedback/`, and the supporting packages listed in the README.
Before public release, the authors must confirm permission to publish all additions
under the repository license; no contributor identity or authorization is inferred here.

Historical raw datasets, private outputs, and model weights are not redistributed.
Obtain external benchmark data and weights from authorized sources and respect
their separate licenses. Built-in fictional records require no external data download.
