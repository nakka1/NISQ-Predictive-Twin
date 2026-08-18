"""
models.py
=========

EdgeLSTM adaptado para receber o vetor de estado físico completo (10
variáveis, em vez de 1-2), e CS_MSELoss (mantida como mecanismo de controle
conservador, com o hiperparâmetro de penalidade ajustável).

Arquitetura comparável à versão anterior -- apenas a camada de entrada muda
de tamanho (input_size passa a ser dinâmico, injetado a partir de
QuantumNetworkDataset.input_size).
"""

import torch
import torch.nn as nn


class EdgeLSTM(nn.Module):
    """
    Rede neural recorrente leve para inferência rápida na borda.

    Arquitetura:
        entrada  -> (batch, seq_len, input_size)  [vetor de estado físico completo]
        LSTM     -> hidden_size compacto, poucas camadas
        saída    -> camada linear + sigmoid, produzindo F_hat(t+1) em [0, 1]
    """

    def __init__(self, input_size: int = 10, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        pred = self.activation(self.head(last_hidden))
        return pred


class CS_MSELoss(nn.Module):
    """
    Cost-Sensitive Mean Squared Error (CS-MSE), mantida como mecanismo de
    controle conservador do controle de admissão.

    Loss = MSE_ponderado(FP severo, FN moderado) + penalização de descarte excessivo

    lambda_penalty é o hiperparâmetro principal (peso multiplicativo sobre o
    erro quadrático dos Falsos Positivos), consistente com a formulação já
    validada nas versões anteriores do Gêmeo Digital -- preservada aqui como
    ponto de referência, conforme solicitado.
    """

    def __init__(self, threshold: float = 0.65, lambda_penalty: float = 10.0,
                 lambda_fn: float = 2.0, discard_penalty_weight: float = 5.0,
                 max_discard_rate: float = 0.60):
        super().__init__()
        self.threshold = threshold
        self.lambda_penalty = lambda_penalty
        self.lambda_fn = lambda_fn
        self.discard_penalty_weight = discard_penalty_weight
        self.max_discard_rate = max_discard_rate

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        squared_error = (y_pred - y_true) ** 2

        is_false_positive = (y_true < self.threshold) & (y_pred >= self.threshold)
        is_false_negative = (y_true >= self.threshold) & (y_pred < self.threshold)

        weights = torch.ones_like(squared_error)
        weights = torch.where(is_false_positive, torch.full_like(squared_error, self.lambda_penalty), weights)
        weights = torch.where(is_false_negative, torch.full_like(squared_error, self.lambda_fn), weights)

        weighted_mse = (squared_error * weights).mean()

        soft_discard_indicator = torch.sigmoid((self.threshold - y_pred) * 50.0)
        discard_rate = soft_discard_indicator.mean()
        excess_discard = torch.clamp(discard_rate - self.max_discard_rate, min=0.0)
        discard_penalty = self.discard_penalty_weight * (excess_discard ** 2)

        return weighted_mse + discard_penalty


def train_edge_lstm(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                     threshold: float = 0.65, lambda_penalty: float = 10.0, lambda_fn: float = 2.0,
                     discard_penalty_weight: float = 5.0, max_discard_rate: float = 0.60,
                     epochs: int = 150, lr: float = 0.012, device: torch.device = None,
                     verbose: bool = False):
    """Rotina de treinamento em batch único (dataset compacto)."""
    if device is not None:
        model = model.to(device)

    criterion = CS_MSELoss(threshold=threshold, lambda_penalty=lambda_penalty, lambda_fn=lambda_fn,
                            discard_penalty_weight=discard_penalty_weight,
                            max_discard_rate=max_discard_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            with torch.no_grad():
                discard_rate_now = (y_pred < threshold).float().mean().item()
            print(f"    Epoch {epoch + 1:3d}/{epochs} | CS-MSE Loss: {loss.item():.6f} | "
                  f"Taxa de descarte (treino): {discard_rate_now*100:.1f}%")
    return model
