"""
network_topology.py
======================

Modular Node -> Channel -> Repeater -> Channel -> Node structure, per the
roadmap requirement: "Estruturar o canal de forma modular ... Evitar
implementar o código de forma monolítica. Cada enlace deve poder possuir
seus próprios T1, T2, perda, distância, BER, ruído."

Also defines lightweight ABSTRACT INTERFACES for capabilities not yet fully
implemented (entanglement swapping, Bell-state measurement, purification),
so the architecture can be EXTENDED later without restructuring:
"Não é obrigatório implementar tudo nessa etapa, mas o código deve permitir
essa extensão."
"""

from abc import ABC, abstractmethod
from dataclasses import replace

import numpy as np

from physics_config import PhysicsConfig
from quantum_channel_v3 import QuantumChannel


class QuantumNode:
    """
    A network endpoint (or intermediate repeater station). Deliberately
    minimal at this stage -- it exists as an explicit topology element so
    that `NetworkLink` and future multi-hop/multi-node topologies have a
    concrete object to attach to, per the modular Node->Channel->Repeater
    structure requested in the roadmap.
    """
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"QuantumNode({self.name!r})"


class NetworkLink:
    """
    One physical link segment between two QuantumNodes: `node_a --[channel]-- node_b`.
    Each link owns its OWN PhysicsConfig (own T1, T2, distance, loss,
    depolarization) and its own QuantumChannel instance -- links are never
    forced to share physics, satisfying "Cada enlace deve poder possuir
    seus próprios T1; T2; perda; distância; BER; ruído."
    """

    def __init__(self, node_a: QuantumNode, node_b: QuantumNode, config: PhysicsConfig, seed: int = None):
        self.node_a = node_a
        self.node_b = node_b
        self.config = config if seed is None else replace(config, SEED=seed)
        self.channel = QuantumChannel(self.config, rng=np.random.default_rng(self.config.SEED))

    def __repr__(self):
        return f"NetworkLink({self.node_a.name} <-> {self.node_b.name}, D={self.config.DISTANCE_KM}km)"

    def transmit(self, transmission_exposure_time: float = None, storage_time: float = 0.0,
                 depol_prob_override: float = None) -> dict:
        depol_prob = depol_prob_override if depol_prob_override is not None else self.config.DEPOLARIZATION_P
        exposure = (transmission_exposure_time if transmission_exposure_time is not None
                    else self.config.TRANSMISSION_EXPOSURE_TIME)
        return self.channel.transmit(distance_km=self.config.DISTANCE_KM, depol_prob=depol_prob,
                                      transmission_exposure_time=exposure, storage_time=storage_time)


class Repeater(QuantumNode):
    """
    A QuantumNode specialized as a repeater: sits between two NetworkLinks
    and (eventually) performs entanglement swapping between them. This
    class composes the two `NetworkLink`s it's attached to but does not
    yet perform swapping -- see `EntanglementSwappingProtocol` below for
    the extension point.
    """
    def __init__(self, name: str, link_left: NetworkLink = None, link_right: NetworkLink = None):
        super().__init__(name)
        self.link_left = link_left
        self.link_right = link_right


# ===========================================================================
# Extension points (interfaces) -- NOT fully implemented, intentionally.
# These exist so entanglement swapping / BSM / purification can be added
# later without restructuring the Node/Channel/Repeater topology above.
# ===========================================================================

class EntanglementSwappingProtocol(ABC):
    """
    Interface for a future entanglement-swapping implementation at a
    Repeater: given two independently-generated Bell pairs (one per
    adjacent link), produce a single longer-range entangled pair spanning
    both links' endpoints. Not implemented in this increment -- the
    interface exists so a concrete strategy (e.g., a real BSM circuit
    simulated in Qiskit Aer) can be dropped in without touching
    `NetworkLink` or `Repeater`.
    """

    @abstractmethod
    def swap(self, pair_left: dict, pair_right: dict) -> dict:
        """
        pair_left / pair_right: dicts as returned by NetworkLink.transmit()
        (containing at least 'F_t' and 'success'). Must return a dict of
        the same shape describing the resulting long-range pair.
        """
        raise NotImplementedError


class BellStateMeasurement(ABC):
    """
    Interface for a future Bell-state-measurement implementation (the
    physical operation entanglement swapping is built on: measure two
    qubits, one from each adjacent pair, in the Bell basis). Not
    implemented in this increment.
    """

    @abstractmethod
    def measure(self, qubit_a, qubit_b) -> str:
        """Should return one of the four Bell-basis outcome labels."""
        raise NotImplementedError


class PurificationProtocol(ABC):
    """
    Interface for a purification strategy applied at a node. The BBPSSW
    circuit already implemented in `repeater.py` (v2 core, still valid and
    reused by the orchestrator) is one concrete instance of this interface
    in spirit; formalizing it as an ABC here makes it swappable for
    alternative purification protocols later without touching the
    orchestrator's control-flow logic.
    """

    @abstractmethod
    def purify(self, pair_a: dict, pair_b: dict) -> dict:
        raise NotImplementedError
