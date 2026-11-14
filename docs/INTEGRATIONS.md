# Proposed Φ integrations (not yet implemented)

The JSON printed by `python -m phineubio` is the initial stable boundary for external consumers.

- **PhiStream:** orchestrate isolated experiment jobs; parse `schema_version`, `configuration`, `metrics`, and `limitations`. Never interpret simulator metrics as biological observations.
- **PhiMemory9:** store a versioned experiment record, including input parameters, code commit SHA, environment, metrics, and provenance. Commit SHA and environment are not emitted automatically in Phase 1 and must be attached by the caller.
- **PhiDiscovery:** compare controlled runs, compute confidence intervals, and suggest follow-ups; do not silently change hypotheses after seeing test outcomes.
- **CompAware:** track operational and ethical questions separately from scientific evidence. A simulated spike or benchmark success cannot imply subjective experience.
- **SuperPhiVessel / SOMA:** potential consumers of readout or experiment APIs after explicit interface design; no coupling in Phase 1.

An approved future biological adapter should use an explicit capability boundary with operator authorization, safety-review references, and hard-coded no-stimulation defaults. No such interface exists in this PR.