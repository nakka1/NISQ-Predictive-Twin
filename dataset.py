"""
dataset.py
==========

QuantumNetworkDataset: substitui o gerador estatístico baseado em
Ornstein-Uhlenbeck por um ambiente de simulação físico, combinando:

    - QuantumNoiseChannel (quantum_channel.py): despolarização + amplitude
      damping + phase damping via operadores de Kraus.
    - WDMTelemetryGenerator (telemetry.py): perdas ópticas e telemetria de
      enlace WDM.

Mantém a MESMA INTERFACE do gerador anterior:
    generate_dataset() -> pandas.DataFrame
    preprocess(df, window_size, test_size) -> (X_train, y_train, X_test, y_test, scaler)

para que EdgeLSTM, CS_MSELoss e o restante do pipeline não precisem ser
reescritos -- apenas o `input_size` do EdgeLSTM muda (de 1/2 para
len(QuantumNetworkDataset.FEATURE_COLUMNS) = 10).

Vetor de estado físico por instante de tempo:
    X_t = [F_t, T1, T2, BER, Loss_dB, Distance_km,
           Transmission_Efficiency, Depolarization_Level, Photon_Rate, Latency]
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from quantum_channel import QuantumNoiseChannel
from telemetry import WDMTelemetryGenerator


class QuantumNetworkDataset:
    """
    Gerador de dataset físico para o Gêmeo Digital do repetidor quântico.

    A cada passo de tempo:
        1. A telemetria WDM é amostrada (distância, perda, potência, BER, fótons).
        2. O canal de ruído quântico físico é aplicado a um par de Bell recém
           criado, por um tempo de exposição `elapsed_time` (escala de
           microssegundos, compatível com T1/T2), produzindo a fidelidade F_t.
        3. A fidelidade é adicionalmente penalizada pela taxa de erro (BER)
           do enlace óptico, refletindo o impacto da camada física clássica
           sobre a qualidade do par entregue.
    """

    FEATURE_COLUMNS = [
        "F_t", "T1", "T2", "BER", "Loss_dB", "Distance_km",
        "Transmission_Efficiency", "Depolarization_Level",
        "Photon_Rate", "Latency",
    ]

    def __init__(self, n_steps: int = 4000, dt: float = 1.15e-5, seed: int = 42,
                 T1_base: float = 50e-6, T2_base: float = 30e-6,
                 depol_prob_base: float = 0.01, distance_km_base: float = 20.0):
        """
        Parâmetros
        ----------
        dt : tempo de exposição do par de Bell ao canal físico por passo, em
             segundos. Deve estar na escala de T1/T2 (microssegundos) para
             que a fidelidade resultante varie de forma informativa; valores
             muito maiores que T2 saturam a fidelidade no piso físico do
             canal (~0.5, ver quantum_channel.py) e destroem o sinal.
        """
        self.n_steps = n_steps
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.T1_base = T1_base
        self.T2_base = T2_base
        self.depol_prob_base = depol_prob_base
        self.distance_km_base = distance_km_base

        self.noise_channel = QuantumNoiseChannel(T1=T1_base, T2=T2_base, depol_prob=depol_prob_base)
        self.telemetry_gen = WDMTelemetryGenerator(distance_km=distance_km_base, seed=seed + 1)

    def _bounded_random_walk(self, base: float, rel_sigma: float, lower: float, upper: float,
                              mean_reversion: float = 0.02) -> np.ndarray:
        """
        Passeio aleatório COM reversão à média (estilo Ornstein-Uhlenbeck),
        usado para variar parâmetros físicos lentamente no tempo.

        Correção importante: uma versão anterior usava um passeio aleatório
        SEM reversão à média aqui. Ao longo de milhares de passos, esse
        passeio podia "vazar" para um regime persistentemente diferente do
        valor base e nunca voltar -- como o dataset é dividido
        cronologicamente (sem embaralhar, para não vazar informação do
        futuro), isso causava um desbalanceamento severo entre treino e
        teste (ex.: 9.2% das amostras de treino acima do limiar de
        fidelidade, mas 0% das amostras de teste). A reversão à média mantém
        a variabilidade lenta desejada sem permitir essa deriva sistemática.
        """
        val = base
        series = np.zeros(self.n_steps)
        for t in range(self.n_steps):
            val += mean_reversion * (base - val) + self.rng.normal(0, rel_sigma * base)
            val = float(np.clip(val, lower, upper))
            series[t] = val
        return series

    def generate_dataset(self) -> pd.DataFrame:
        """Gera o dataset físico completo: telemetria WDM + canal quântico + fidelidade."""
        T1_series = self._bounded_random_walk(self.T1_base, 0.01, self.T1_base * 0.5, self.T1_base * 1.5)
        T2_raw = self._bounded_random_walk(self.T2_base, 0.01, self.T2_base * 0.5, self.T2_base * 1.5)
        T2_series = np.minimum(T2_raw, 2.0 * T1_series)  # restrição física T2 <= 2*T1
        depol_series = np.clip(
            self._bounded_random_walk(self.depol_prob_base, 0.05, 0.001, 0.10), 0.001, 0.10
        )
        distance_series = np.clip(
            self._bounded_random_walk(self.distance_km_base, 0.005,
                                       self.distance_km_base * 0.5, self.distance_km_base * 2.0),
            1.0, None,
        )
        # Tempo de exposição ao canal: PASSEIO ALEATÓRIO LIMITADO (não i.i.d.!).
        # Correção importante: uma versão anterior usava ruído i.i.d. por passo
        # aqui (rng.normal independente a cada t). Como a fidelidade é MUITO
        # sensível ao tempo de exposição (ver quantum_channel.py), esse ruído
        # i.i.d. dominava toda a variância de F_t e afogava o sinal de deriva
        # lenta dos parâmetros físicos (T1, T2, distância) -- tornando F_t
        # essencialmente imprevisível a partir do histórico (o MAE de um
        # preditor de média constante empatava com o do EdgeLSTM treinado).
        # Ao modelar o tempo de exposição como um passeio aleatório limitado
        # (autocorrelacionado, como os demais parâmetros físicos), F_t volta a
        # ter estrutura temporal genuína e aprendível.
        elapsed_time_series = self._bounded_random_walk(
            self.dt, 0.03, self.dt * 0.3, self.dt * 3.0
        )

        rows = []
        for t in range(self.n_steps):
            telemetry_step = self.telemetry_gen.generate_step(distance_km=distance_series[t])

            fidelity = self.noise_channel.apply(
                elapsed_time=elapsed_time_series[t],
                depol_prob_override=depol_series[t],
            )
            # Penalização adicional pela taxa de erro do enlace óptico clássico
            fidelity = float(np.clip(fidelity - 0.5 * telemetry_step["ber"], 0.0, 1.0))

            # Latência do canal: tempo de exposição + tempo de propagação na fibra
            # (velocidade de propagação em fibra ~= 2e5 km/s)
            latency = float(elapsed_time_series[t] + telemetry_step["distance_km"] / 2.0e5)

            rows.append({
                "F_t": fidelity,
                "T1": T1_series[t],
                "T2": T2_series[t],
                "BER": telemetry_step["ber"],
                "Loss_dB": telemetry_step["loss_db"],
                "Distance_km": telemetry_step["distance_km"],
                "Transmission_Efficiency": telemetry_step["transmission_efficiency"],
                "Depolarization_Level": depol_series[t],
                "Photon_Rate": telemetry_step["photon_rate"],
                "Latency": latency,
            })

        return pd.DataFrame(rows)

    def preprocess(self, df: pd.DataFrame, window_size: int = 20, test_size: float = 0.2):
        """
        Normaliza TODAS as features do vetor de estado físico (10 colunas,
        incluindo F_t, que agora é parte da observação) com MinMaxScaler,
        constrói janelas deslizantes (batch, seq_len, 10) e separa
        treino/teste sem embaralhar (preserva a ordem cronológica). O alvo
        de predição continua sendo a fidelidade futura F_(t+1).
        """
        features = df[self.FEATURE_COLUMNS].values
        target = df[["F_t"]].values  # já está em [0, 1]

        feat_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        features_scaled = feat_scaler.fit_transform(features)

        X, y = [], []
        for i in range(len(features_scaled) - window_size):
            X.append(features_scaled[i:i + window_size])
            y.append(target[i + window_size])
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        split_idx = int(len(X) * (1.0 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Guarda também as linhas físicas cruas (não normalizadas) alinhadas ao
        # conjunto de teste, para permitir que o QuantumRepeaterNode receba
        # telemetria real durante a simulação orquestrada (ver repeater.py /
        # orchestrator.py).
        raw_test_rows = df.iloc[split_idx + window_size:].reset_index(drop=True)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        y_test_t = torch.tensor(y_test, dtype=torch.float32)

        return X_train_t, y_train_t, X_test_t, y_test_t, feat_scaler, raw_test_rows

    @property
    def input_size(self) -> int:
        return len(self.FEATURE_COLUMNS)
