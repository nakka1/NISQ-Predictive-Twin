# Physics

## Causal chain

```
theta(t) [environmental perturbation]
    -> Delta_phi_c(t) [optical phase drift]
    -> interference_penalty_dB = -10*log10(cos(Delta_phi_c)^2)
    -> optical_power_dbm = TX_POWER - loss_dB - interference_penalty_dB
    -> OSNR_dB = optical_power_dbm - NOISE_FLOOR_dBm
    -> BER_optical = 0.5*erfc(sqrt(OSNR_linear))   [Proakis, AWGN/BPSK]
theta(t) -> T1(t)/T2(t) [shared environmental coupling: T_eff = T_base * exp(-KAPPA*theta^2)]
depol_effective = depol_base + KAPPA_DEPOL_FROM_BER * BER_optical
    -> QuantumChannel.transmit() [real Qiskit Aer density-matrix simulation]
    -> F(t)
```

Every approximation above carries an equation, a validity range, and a
named limitation in `dataset_v3.py`'s module docstring — see that file
for the complete, precise statement of each. In short: (1) the
interference-penalty formula assumes small-angle-adjacent phase noise;
(2) the BER formula assumes AWGN/BPSK, not the project's actual
modulation scheme; (3) depolarization-from-BER is a linear coupling
coefficient, not derived from a first-principles noise budget; (4) the
T1/T2-environmental-coupling exponential is a phenomenological fit, not
a first-principles decoherence derivation.

## Quantum channel: two validated engines

- **`ReferenceEngine`** (`quantum_channel_v3.QuantumChannel`): builds a
  real noisy circuit and runs it through `AerSimulator(method="density_matrix")`.
  The most physically faithful engine in this project.
- **`AnalyticalEngine`** (`quantum_channel.QuantumNoiseChannel`, formerly named `FastEngine` -- kept as a backward-compatible alias): closed-form
  Kraus-operator algebra (depolarizing x amplitude-damping x
  phase-damping), no circuit execution.

Both agree to floating-point precision on every regime tested
(`quantum_twin/quantum/physics_engine.py`'s benchmark). Speed is
regime-dependent: `AnalyticalEngine` wins ~6x when a fresh engine object must
be constructed per call, shows no advantage when an engine object is
reused (as the actual dataset generator does).

## Purification (BBPSSW)

Bennett et al. (1996) protocol. Two forms, cross-validated to <1e-8:

```
p_success(F) = F^2 + (2/3)*F*(1-F) + (5/9)*(1-F)^2
F_after(F)   = [F^2 + (1/9)*(1-F)^2] / p_success(F)
```

`purification.DensityMatrixBBPSSW` implements the SAME physics via a
real bilateral-CNOT circuit on two Werner-state density matrices
(qubit-indexing convention documented explicitly in that module, after a
real big-endian/little-endian bug was found and fixed during
development).

## Entanglement swapping

Werner-state formula, cross-validated against a real BSM density-matrix
simulation:

```
F_swap = F1*F2 + (1-F1)*(1-F2)/3
```

## Quantum memory (WAIT decoherence)

`quantum_memory.QuantumMemory` stores a Werner-state density matrix and
applies real T1/T2 decoherence over an arbitrary elapsed time.
`environment.py`'s `begin_wait_hold()`/`wait_tick_and_reobserve()` cycle
uses this for genuine multi-round WAIT decisions — verified
monotonically decreasing fidelity from any valid (F>0) starting pair.

## Physics regression suite

`tests/test_physics_regression.py` locks in exact golden numeric values
(not just formula agreement) for channel, memory, purification,
swapping, and multi-hop, each with an explicit absolute or relative
tolerance — designed to catch silent physics drift from future
refactoring. See that file's module docstring for the full list and the
exact generating call for every golden value.
