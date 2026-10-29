# Production file-owner evidence

The tests use each port's production FileSessionStore and native identity/probe
functions. A separate owned holder lives past TTL, then exits through a nonce
barrier without release. Recovery replaces that exact nonce. Foreign, foreign
released and both-unknown identity fixtures stay held with zero PID probes.
The both-unknown row injects unavailable identity through the test framework;
no container is created or claimed as measured.

Run node tools/file-owner-native/gate.mjs ts (or py). Windows runs ordinary,
nested and inherited-supervisor owned Job contexts without leaving the ambient
chain. The no-Job attempt must prove absent membership and pass, or retain the
exact approved unmeasured condition; it never claims recovery in that case.

windows-native.cs and windows-context-contract.mjs are vendored from accepted
decision0022 review tooling at831d714. vendor.json pins their contents with
documented newline normalization. They are verified before contract import or
C# compilation. They are test tooling, excluded from published runtime artifacts.
Identity production modules contain no Job/silo API. Eight report controls and
six producer-verdict controls reject missing work and fabricated evidence.
