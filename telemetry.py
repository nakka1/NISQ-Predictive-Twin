"""
telemetry.py
============

Gerador de telemetria sintética de um enlace óptico WDM (Wavelength Division
Multiplexing), modelando a camada física do canal entre nós do repetidor.

Perda de fibra:
    eta(L) = 10^(-alpha * L / 10),   alpha ~= 0.2 dB/km (fibra padrão)
"""

import numpy as np


class WDMTelemetryGenerator:
    """
    Simula o comportamento físico de um enlace óptico WDM: perda de fibra,
    potência recebida, taxa de fótons disponível, taxa de erro estimada e
    disponibilidade do canal, com variação temporal.
    """

    ALPHA_DB_PER_KM = 0.2  # atenuação típica de fibra óptica monomodo (dB/km)

    def __init__(self, distance_km: float = 20.0, tx_power_dbm: float = 0.0,
                 photon_rate_base: float = 1.0e6, seed: int = 123,
                 channel_outage_prob: float = 0.01):
        self.distance_km = distance_km
        self.tx_power_dbm = tx_power_dbm
        self.photon_rate_base = photon_rate_base
        self.channel_outage_prob = channel_outage_prob
        self.rng = np.random.default_rng(seed)

    def fiber_loss_db(self, distance_km: float = None) -> float:
        """Perda de fibra em dB: Loss_dB = alpha * L (sem ruído)."""
        L = distance_km if distance_km is not None else self.distance_km
        return self.ALPHA_DB_PER_KM * L

    def transmission_efficiency(self, distance_km: float = None) -> float:
        """eta(L) = 10^(-alpha*L/10): fração de fótons transmitidos com sucesso."""
        L = distance_km if distance_km is not None else self.distance_km
        return float(10 ** (-self.ALPHA_DB_PER_KM * L / 10.0))

    def generate_step(self, distance_km: float = None) -> dict:
        """
        Gera uma amostra de telemetria para um instante de tempo, incluindo
        ruído de medição realista (variações de potência, jitter de perda).
        """
        L = distance_km if distance_km is not None else self.distance_km

        loss_db = self.fiber_loss_db(L) + self.rng.normal(0, 0.05)
        eta = self.transmission_efficiency(L)
        received_power_dbm = self.tx_power_dbm - loss_db
        photon_rate = max(
            self.photon_rate_base * eta * (1 + self.rng.normal(0, 0.02)), 0.0
        )
        # BER cresce com a perda de eficiência de transmissão (mais erros com menos fótons)
        ber = float(np.clip(1e-3 * (1 - eta) + abs(self.rng.normal(0, 1e-4)), 0.0, 1.0))
        channel_available = 1.0 if self.rng.random() > self.channel_outage_prob else 0.0

        return {
            "loss_db": float(loss_db),
            "transmission_efficiency": float(eta),
            "received_power_dbm": float(received_power_dbm),
            "photon_rate": float(photon_rate),
            "ber": ber,
            "channel_available": channel_available,
            "distance_km": float(L),
        }
