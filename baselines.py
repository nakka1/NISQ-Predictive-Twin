"""
baselines.py
============

Implementa os baselines adicionais solicitados, para verificar se o
EdgeLSTM + CS_MSELoss realmente apresenta vantagem sobre alternativas mais
simples ou mais pesadas:

    - Baseline 1: LSTM tradicional (mesma arquitetura do EdgeLSTM) treinada
      apenas com MSE (sem o mecanismo de custo assimétrico).
    - Baseline 2: modelo clássico de regressão em árvore (Random Forest,
      Gradient Boosting, ou XGBoost quando disponível) sobre a janela
      "achatada" (flatten) em um vetor de features.
    - Baseline 3: modelo temporal baseado em Transformer (encoder leve).

Todos os baselines expõem a MESMA interface duck-typed usada pelo
DigitalTwinOrchestrator: um objeto com `.eval()` (no-op para os não-torch) e
`__call__(x) -> tensor de shape (1, 1)`, para que possam ser conectados
diretamente ao mesmo laço de simulação (run_intelligent) e ao mesmo limiar de
controle de admissão (F_threshold = 0.65) usados pelo EdgeLSTM + CS_MSELoss,
garantindo uma comparação cientificamente justa (mesmo protocolo de decisão,
mesmo dataplane quântico, muda apenas o preditor).
"""

import numpy as np
import torch
import torch.nn as nn

from models import EdgeLSTM

try:
    import xgboost as xgb
    _HAS_XGBOOST = True
except ImportError:
    _HAS_XGBOOST = False

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


# ===========================================================================
# Baseline 1: LSTM tradicional (mesma arquitetura), apenas MSE
# ===========================================================================
def train_lstm_mse_baseline(X_train: torch.Tensor, y_train: torch.Tensor, input_size: int,
                             hidden_size: int = 16, num_layers: int = 1,
                             epochs: int = 150, lr: float = 0.012,
                             device: torch.device = None, verbose: bool = False) -> EdgeLSTM:
    """
    Treina uma LSTM com a MESMA arquitetura do EdgeLSTM, porém usando
    nn.MSELoss padrão (sem a assimetria de custo da CS_MSELoss). Serve como
    baseline para isolar o efeito específico da função de perda
    cost-sensitive, mantendo a arquitetura de rede constante.
    """
    model = EdgeLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    if device is not None:
        model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    [LSTM+MSE] Epoch {epoch + 1:3d}/{epochs} | MSE Loss: {loss.item():.6f}")
    return model


# ===========================================================================
# Baseline 2: modelos clássicos de árvore (Random Forest / GBM / XGBoost)
# ===========================================================================
class SklearnRegressorAdapter:
    """
    Adapta um regressor scikit-learn/XGBoost (que opera sobre vetores fixos)
    para a interface duck-typed usada pelo DigitalTwinOrchestrator
    (`.eval()` no-op + `__call__(x) -> tensor (1, 1)`).

    A janela temporal (seq_len, n_features) é achatada em um único vetor de
    features (seq_len * n_features) antes da predição -- os modelos de árvore
    não têm noção nativa de sequência, então recebem a janela inteira como
    atributos independentes.
    """

    def __init__(self, regressor, window_size: int, n_features: int):
        self.regressor = regressor
        self.window_size = window_size
        self.n_features = n_features

    def eval(self):
        return self  # no-op, mantém a interface

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x: (1, seq_len, n_features) -> vetor achatado (1, seq_len*n_features)
        x_np = x.detach().cpu().numpy().reshape(1, -1)
        pred = self.regressor.predict(x_np)
        pred = float(np.clip(pred[0], 0.0, 1.0))
        return torch.tensor([[pred]], dtype=torch.float32)


def _flatten_windows(X: torch.Tensor) -> np.ndarray:
    """(N, seq_len, n_features) -> (N, seq_len*n_features), para modelos de árvore."""
    X_np = X.detach().cpu().numpy()
    return X_np.reshape(X_np.shape[0], -1)


def train_tree_baseline(X_train: torch.Tensor, y_train: torch.Tensor,
                         method: str = "random_forest", seed: int = 42,
                         verbose: bool = False) -> SklearnRegressorAdapter:
    """
    Treina um regressor clássico de árvore sobre as janelas achatadas.

    method: "random_forest" | "gradient_boosting" | "xgboost" (se disponível,
            senão cai automaticamente para "gradient_boosting").
    """
    X_flat = _flatten_windows(X_train)
    y_flat = y_train.detach().cpu().numpy().ravel()

    if method == "xgboost" and not _HAS_XGBOOST:
        if verbose:
            print("    [Baseline 2] xgboost indisponível, usando gradient_boosting como fallback.")
        method = "gradient_boosting"

    if method == "random_forest":
        regressor = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed, n_jobs=-1)
    elif method == "gradient_boosting":
        regressor = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=seed)
    elif method == "xgboost":
        regressor = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      random_state=seed, n_jobs=-1, verbosity=0)
    else:
        raise ValueError(f"Método de árvore desconhecido: {method}")

    if verbose:
        print(f"    [Baseline 2] Treinando {method} sobre {X_flat.shape[0]} amostras "
              f"({X_flat.shape[1]} features achatadas) ...")
    regressor.fit(X_flat, y_flat)

    _, seq_len, n_features = X_train.shape
    return SklearnRegressorAdapter(regressor, window_size=seq_len, n_features=n_features)


# ===========================================================================
# Baseline 3: modelo temporal baseado em Transformer
# ===========================================================================
class TransformerFidelityPredictor(nn.Module):
    """
    Modelo temporal leve baseado em um encoder Transformer, prevendo a
    fidelidade futura F(t+1) a partir da janela de estado físico. Mantém uma
    interface comparável ao EdgeLSTM (mesma entrada/saída), para permitir
    comparação direta sob o mesmo protocolo de admissão.
    """

    def __init__(self, input_size: int, d_model: int = 16, nhead: int = 2,
                 num_layers: int = 1, dim_feedforward: int = 32, max_seq_len: int = 64):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        h = self.input_proj(x) + self.pos_embedding[:, :seq_len, :]
        h = self.encoder(h)
        last_hidden = h[:, -1, :]
        pred = self.activation(self.head(last_hidden))
        return pred


def train_transformer_baseline(X_train: torch.Tensor, y_train: torch.Tensor, input_size: int,
                                d_model: int = 16, nhead: int = 2, num_layers: int = 1,
                                epochs: int = 150, lr: float = 0.005,
                                device: torch.device = None, verbose: bool = False) -> TransformerFidelityPredictor:
    """Treina o TransformerFidelityPredictor com MSE padrão (mesmo espírito do baseline 1)."""
    seq_len = X_train.shape[1]
    model = TransformerFidelityPredictor(input_size=input_size, d_model=d_model, nhead=nhead,
                                          num_layers=num_layers, max_seq_len=max(seq_len, 64))
    if device is not None:
        model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    [Transformer] Epoch {epoch + 1:3d}/{epochs} | MSE Loss: {loss.item():.6f}")
    return model


def train_transformer_with_cs_loss(X_train: torch.Tensor, y_train: torch.Tensor, input_size: int,
                                    threshold: float = 0.65, lambda_penalty: float = 4.0,
                                    lambda_fn: float = 4.0, discard_penalty_weight: float = 10.0,
                                    max_discard_rate: float = 0.60, d_model: int = 16, nhead: int = 2,
                                    num_layers: int = 1, epochs: int = 150, lr: float = 0.005,
                                    device: torch.device = None,
                                    verbose: bool = False) -> TransformerFidelityPredictor:
    """
    Ablation helper: trains the SAME Transformer architecture used in
    `train_transformer_baseline`, but with the CS_MSELoss instead of plain
    MSE -- isolates whether the Transformer's edge over EdgeLSTM (observed
    in Experiment 3 / multi-seed validation) comes from the ARCHITECTURE
    itself, or specifically from its interaction with the asymmetric
    cost-sensitive loss. Compare against `models.train_edge_lstm` (LSTM +
    CS_MSELoss) under identical hyperparameters to isolate the effect.
    """
    from models import CS_MSELoss  # local import to avoid a circular import at module load time

    seq_len = X_train.shape[1]
    model = TransformerFidelityPredictor(input_size=input_size, d_model=d_model, nhead=nhead,
                                          num_layers=num_layers, max_seq_len=max(seq_len, 64))
    if device is not None:
        model = model.to(device)

    criterion = CS_MSELoss(threshold=threshold, lambda_penalty=lambda_penalty, lambda_fn=lambda_fn,
                            discard_penalty_weight=discard_penalty_weight, max_discard_rate=max_discard_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if verbose and (epoch + 1) % 30 == 0:
            print(f"    [Transformer+CS] Epoch {epoch + 1:3d}/{epochs} | CS-MSE Loss: {loss.item():.6f}")
    return model
