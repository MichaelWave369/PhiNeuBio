# Architecture and boundaries

Phase 1 implements a deliberately small digital model:

    Binary input stream
           |
    Seeded LIF reservoir (leak, thresholds, sparse recurrent edges)
           |
    Potential + spike feature vectors
           |
    Trainable logistic readout
           |
    Held-out accuracy + baseline + JSON report

The reservoir updates synchronously, consumes the previous step's spikes on recurrent edges, and resets between independent sequences. It learns no recurrent weights: **only the readout is trained**. This distinction is essential when interpreting results.

Everything is pure Python with no network access, biospecimens, remote services, or device drivers. Paths intended for future adapters are conceptual, not implemented.

Potential future components: time-varying feedback, multiple readout baselines, energy accounting, plastic synapses, robust seed sweeps, PhiStream task routing, PhiMemory9 experiment ledgers, optional approved hardware providers.

No runtime dependency on another Φ project is introduced.