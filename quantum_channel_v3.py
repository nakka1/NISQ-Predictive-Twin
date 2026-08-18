"""
quantum_channel_v3.py
========================

CAUSAL rewrite of the physics core, per the roadmap. The central rule this
module exists to satisfy:

    NOT:  parameters -> artificial formula -> F(t)
    YES:  quantum state -> noise/loss channels -> degraded state -> F(t)

Every quantity in the resulting telemetry (Loss_dB, Transmission_Efficiency,
Photon_Rate, BER, F_t) is *derived* from the same small set of physical
inputs (Distance_km, T1, T2, depolarizing probability, exposure time),
instead of being sampled independently and glued together afterward.

Key differences from the old (v2) `quantum_channel.py`:
    - F(t) comes from an ACTUAL Qiskit Aer density-matrix simulation of a
      noisy Bell-pair circuit (qiskit_aer.noise.depolarizing_error +
      amplitude_damping_error + phase_damping_error, both natively from
      Qiskit Aer, applied to a real `id` gate on a real circuit run through
      AerSimulator(method="density_matrix")) -- not closed-form Kraus
      algebra computed by hand.
    - Optical loss is a real erasure event: with probability
      (1 - Transmission_Efficiency), the round fails outright (no pair
      delivered) instead of just scaling a formula.
    - Loss_dB is *always* derived from Distance_km (never sampled
      independently); Transmission_Efficiency is *always* derived from
      Loss_dB; Photon_Rate is *always* derived from Transmission_Efficiency.
    - BER is derived from the SAME depolarizing probability and loss that
      drive F(t) -- coherent by construction, not an independent random
      draw that then artificially subtracts from F(t).
    - Quantum memory storage (T1/T2-driven decoherence while the pair
      waits in memory) is modeled as additional exposure time, separate
      from and in addition to transmission exposure time.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit.quantum_info import DensityMatrix, state_fidelity

from physics_config import PhysicsConfig


class QuantumChannel:
    """
    A single physical link segment: fiber of a given distance, with
    time-varying T1/T2/depolarization, causally producing telemetry and a
    Bell-pair fidelity via a REAL Qiskit Aer simulation.
    """

    def __init__(self, config: PhysicsConfig, rng: np.random.Generator = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng(config.SEED)

        # Pre-build and pre-transpile the circuit ONCE (structure never
        # changes -- only the noise model attached to the 'id' gates does).
        # optimization_level=0 is REQUIRED here: at higher optimization
        # levels the transpiler removes "no-op" id gates before the noise
        # model ever sees them, silently making the channel noiseless.
        self._circuit = self._build_circuit()
        self._base_sim = AerSimulator(method="density_matrix")
        self._compiled_circuit = transpile(self._circuit, self._base_sim, optimization_level=0)

        psi = np.zeros(4, dtype=complex)
        psi[0] = psi[3] = 1.0 / np.sqrt(2)
        self._ideal_bell = DensityMatrix(np.outer(psi, psi.conj()))

    @staticmethod
    def _build_circuit() -> QuantumCircuit:
        """Bell-pair preparation + an idle ('id') gate on each qubit, which
        is where the noise model's error channel is attached -- this idle
        gate represents the qubit's exposure to the channel/memory."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.id(0)
        qc.id(1)
        qc.save_density_matrix()
        return qc

    # -----------------------------------------------------------------
    # Causal derivation chain: Distance -> Loss -> Efficiency -> PhotonRate
    # -----------------------------------------------------------------
    def loss_db(self, distance_km: float) -> float:
        """Loss_dB = alpha * Distance_km -- ALWAYS derived, never sampled independently."""
        return self.config.ALPHA_DB_PER_KM * distance_km

    def transmission_efficiency(self, distance_km: float) -> float:
        """eta = 10^(-Loss_dB/10) -- ALWAYS derived from Loss_dB, never independent."""
        return float(10 ** (-self.loss_db(distance_km) / 10.0))

    def photon_rate(self, distance_km: float, source_stability_noise: float = 0.0) -> float:
        """Photon_Rate = base_rate * efficiency * (1 + small source jitter)
        -- ALWAYS derived from efficiency (hence from loss, hence from distance)."""
        eta = self.transmission_efficiency(distance_km)
        return max(self.config.PHOTON_RATE_BASE * eta * (1.0 + source_stability_noise), 0.0)

    def bit_error_rate(self, depol_prob: float, distance_km: float) -> float:
        """
        BER causally coupled to the SAME depolarizing probability and the
        SAME transmission efficiency that drive F(t) -- not an independent
        random draw. A depolarizing channel with probability p flips a
        fixed-basis measurement outcome with probability ~p/2; erasure loss
        contributes the remainder (an undetected/lost photon reads as an
        error at the classical layer).
        """
        eta = self.transmission_efficiency(distance_km)
        return float(np.clip(0.5 * depol_prob + 0.5 * (1.0 - eta), 0.0, 1.0))

    # -----------------------------------------------------------------
    # Causal noise model construction
    # -----------------------------------------------------------------
    def _build_noise_model(self, depol_prob: float, exposure_time: float) -> NoiseModel:
        """
        Composite noise model using Qiskit Aer's NATIVE noise channel
        constructors (not hand-derived Kraus operators):
            - depolarizing_error(p, 1)               -- gate/transmission errors
            - amplitude_damping_error(gamma(t))       -- T1 relaxation, gamma = 1 - e^(-t/T1)
            - phase_damping_error(lambda(t))          -- T2 dephasing, lambda = 1 - e^(-t/T2)
        Composed into a single QuantumError and attached to the 'id' gate,
        which is where each qubit "sits" while exposed to the channel.
        """
        gamma = 1.0 - np.exp(-exposure_time / self.config.T1) if self.config.T1 > 0 else 0.0
        lam = 1.0 - np.exp(-exposure_time / self.config.T2) if self.config.T2 > 0 else 0.0

        depol = depolarizing_error(depol_prob, 1)
        amp = amplitude_damping_error(gamma)
        phase = phase_damping_error(lam)
        combined = depol.compose(amp).compose(phase)

        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(combined, ["id"])
        return noise_model

    def simulate_fidelity(self, depol_prob: float, exposure_time: float) -> float:
        """
        Runs the ACTUAL noisy circuit through AerSimulator and returns the
        fidelity of the resulting density matrix against the ideal Bell
        state -- this is the causal replacement for the old closed-form
        Kraus-algebra shortcut.
        """
        noise_model = self._build_noise_model(depol_prob, exposure_time)
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        result = sim.run(self._compiled_circuit).result()
        rho = result.data(0)["density_matrix"]
        return float(state_fidelity(rho, self._ideal_bell))

    def apply_decoherence_to_state(self, input_density_matrix: DensityMatrix,
                                    depol_prob: float, exposure_time: float) -> DensityMatrix:
        """
        Applies this channel's noise model to an ARBITRARY input 2-qubit
        density matrix (not necessarily the ideal Bell state) for the
        given exposure time -- e.g., a pair that has already decohered
        during storage and is now decohering FURTHER, rather than a fresh
        ideal pair. Used by `quantum_memory.QuantumMemory` for a rigorous
        (density-matrix-level) treatment of repeated/compounding storage
        decay, replacing an earlier scalar-fidelity-multiplication
        approximation.

        Implemented via Qiskit Aer's `set_density_matrix` instruction
        (initializes the simulator's state to `input_density_matrix`
        instead of preparing a fresh Bell pair), then applies the same
        noisy 'id' gates as `simulate_fidelity`.
        """
        qc = QuantumCircuit(2)
        qc.set_density_matrix(input_density_matrix.data)
        qc.id(0)
        qc.id(1)
        qc.save_density_matrix()

        noise_model = self._build_noise_model(depol_prob, exposure_time)
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)
        compiled = transpile(qc, sim, optimization_level=0)
        result = sim.run(compiled).result()
        return result.data(0)["density_matrix"]

    # -----------------------------------------------------------------
    # Full causal transmission event: loss (erasure) -> noisy simulation -> telemetry
    # -----------------------------------------------------------------
    def transmit(self, distance_km: float, depol_prob: float, transmission_exposure_time: float,
                 storage_time: float = 0.0) -> dict:
        """
        A single causal transmission + (optional) storage event.

        1. Optical loss check: with probability (1 - Transmission_Efficiency),
           the photon is lost in transit (erasure) -- the round fails
           OUTRIGHT, and no fidelity is even computed (there is no pair).
           This makes Loss_dB/Transmission_Efficiency causally determine
           whether a usable pair exists at all, not just a cosmetic column.
        2. If the photon survives, the Bell pair is prepared and exposed to
           the noise channel for (transmission_exposure_time + storage_time)
           -- transmission exposure represents the fiber traversal itself;
           storage_time represents any additional time spent in quantum
           memory before use (T1/T2-driven degradation during storage, per
           the roadmap's "memoria quantica" requirement).
        3. F(t) is read off the ACTUAL simulated density matrix.
        4. BER is derived from the SAME depol_prob/efficiency (not from F(t)).

        Returns a dict with every derived quantity, causally consistent by
        construction.
        """
        loss_db = self.loss_db(distance_km)
        eta = self.transmission_efficiency(distance_km)
        photon_rate = self.photon_rate(distance_km)
        ber = self.bit_error_rate(depol_prob, distance_km)

        channel_available = self.rng.random() <= eta  # erasure/loss event

        if not channel_available:
            return {
                "F_t": 0.0, "channel_available": 0.0, "success": False,
                "Loss_dB": loss_db, "Transmission_Efficiency": eta, "Photon_Rate": photon_rate,
                "BER": ber, "Distance_km": distance_km,
            }

        total_exposure_time = transmission_exposure_time + storage_time
        fidelity = self.simulate_fidelity(depol_prob, total_exposure_time)

        return {
            "F_t": fidelity, "channel_available": 1.0, "success": True,
            "Loss_dB": loss_db, "Transmission_Efficiency": eta, "Photon_Rate": photon_rate,
            "BER": ber, "Distance_km": distance_km,
        }
