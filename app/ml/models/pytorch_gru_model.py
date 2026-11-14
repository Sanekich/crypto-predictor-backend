"""Small CPU-friendly GRU direction challenger for full derivatives sequences."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.ml.models.base import ReturnModel


class PyTorchGRUDirectionModel(ReturnModel):
    name = "pytorch_gru_direction"

    def __init__(self, input_size: int = 1, hidden_size: int = 48, dropout: float = 0.20) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._state: dict[str, Any] | None = None
        self._torch = None
        self._net = None

    @staticmethod
    def _torch_module():
        try:
            import torch
            import torch.nn as nn
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyTorch is required. Install dependencies with: pip install -r requirements.txt") from exc
        return torch, nn

    def _build(self):
        torch, nn = self._torch_module()
        input_size, hidden_size, dropout = self.input_size, self.hidden_size, self.dropout

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.head = nn.Sequential(nn.Linear(hidden_size, 24), nn.ReLU(), nn.Dropout(dropout), nn.Linear(24, 1))

            def forward(self, x):
                _, hidden = self.gru(x)
                return self.head(self.dropout(hidden[-1])).squeeze(-1)

        self._net = Net()
        if self._state is not None:
            self._net.load_state_dict(self._state)
        self._torch = torch
        return self._net

    def _require_net(self):
        if self._net is None:
            self._build()
        return self._net

    def fit(self, X_train, y_train, X_val=None, y_val=None, *, epochs: int = 30, batch_size: int = 256, patience: int = 6, learning_rate: float = 1e-3) -> "PyTorchGRUDirectionModel":
        torch, nn = self._torch_module()
        torch.manual_seed(42)
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)
        self.mean_ = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        self.std_ = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        X_train = (X_train - self.mean_) / self.std_
        if X_val is not None:
            X_val = (np.asarray(X_val, dtype=np.float32) - self.mean_) / self.std_
        self._net = self._build()
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=learning_rate, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(epochs):
            self._net.train()
            order = np.random.default_rng(42).permutation(len(X_train))
            for start in range(0, len(order), batch_size):
                ix = order[start:start + batch_size]
                optimizer.zero_grad()
                loss = criterion(self._net(torch.from_numpy(X_train[ix])), torch.from_numpy(y_train[ix]))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
            if X_val is None:
                continue
            self._net.eval()
            with torch.no_grad():
                val_loss = float(criterion(self._net(torch.from_numpy(X_val)), torch.from_numpy(np.asarray(y_val, dtype=np.float32))))
            if val_loss < best_loss - 1e-5:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._state = {k: v.detach().cpu() for k, v in self._net.state_dict().items()}
        return self

    def predict_proba_up(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("model is not fitted")
        torch = self._torch_module()[0]
        X = (np.asarray(X, dtype=np.float32) - self.mean_) / self.std_
        self._require_net().eval()
        with torch.no_grad():
            return torch.sigmoid(self._net(torch.from_numpy(X))).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_up(X)

    def params(self) -> dict[str, Any]:
        return {"input_size": self.input_size, "hidden_size": self.hidden_size, "dropout": self.dropout}

    def training_info(self) -> dict[str, Any]:
        return {"sequence_model": "GRU", "sequence_length": 168, "scaled_on_train_only": True}

    def save(self, directory: Path) -> None:
        torch = self._torch_module()[0]
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self._state or self._require_net().state_dict(), directory / "model.pt")
        np.savez(directory / "scaler.npz", mean=self.mean_, std=self.std_)
        (directory / "params.json").write_text(json.dumps(self.params(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "PyTorchGRUDirectionModel":
        torch = cls._torch_module()[0]
        params = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        instance = cls(**params)
        scaler = np.load(directory / "scaler.npz")
        instance.mean_, instance.std_ = scaler["mean"], scaler["std"]
        instance._state = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
        instance._build()
        return instance
