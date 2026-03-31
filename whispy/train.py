"""Training module — ML and DL model training with evaluation metrics.

Usage:
    from whispy.load import load
    from whispy.train import train_model, evaluate

    train_ds, test_ds = load("OfficeLocalization")
    model, metrics = train_model(train_ds, test_ds, model="rf")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix as sk_cm,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from whispy.load import CSIDataset


# =========================================================================
# Metrics
# =========================================================================
@dataclass
class Metrics:
    """Classification evaluation metrics."""

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    kappa: float = 0.0
    mcc: float = 0.0
    precision_weighted: float = 0.0
    recall_weighted: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=list)
    mean_confidence: float = 0.0
    ece: float = 0.0
    train_time_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"  Accuracy:       {self.accuracy:.4f}",
            f"  Balanced Acc:   {self.balanced_accuracy:.4f}",
            f"  Cohen's Kappa:  {self.kappa:.4f}",
            f"  MCC:            {self.mcc:.4f}",
            f"  Precision (w):  {self.precision_weighted:.4f}",
            f"  Recall (w):     {self.recall_weighted:.4f}",
        ]
        if self.mean_confidence > 0:
            lines.append(f"  Mean Conf:      {self.mean_confidence:.4f}")
            lines.append(f"  ECE:            {self.ece:.4f}")
        lines.append(f"  Train time:     {self.train_time_s:.2f}s")
        return "\n".join(lines)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    train_time: float = 0.0,
) -> Metrics:
    """Compute all classification metrics."""
    n_cls = max(int(y_true.max()), int(y_pred.max())) + 1
    labels = list(range(n_cls))

    m = Metrics(
        accuracy=round(float(accuracy_score(y_true, y_pred)), 4),
        balanced_accuracy=round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        kappa=round(float(cohen_kappa_score(y_true, y_pred)), 4),
        mcc=round(float(matthews_corrcoef(y_true, y_pred)), 4),
        precision_weighted=round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        recall_weighted=round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        confusion_matrix=sk_cm(y_true, y_pred, labels=labels).tolist(),
        train_time_s=round(train_time, 2),
    )

    if y_prob is not None:
        max_p = y_prob.max(axis=1)
        m.mean_confidence = round(float(np.mean(max_p)), 4)
        # ECE (10 bins)
        edges = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (max_p > lo) & (max_p <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = float((y_pred[mask] == y_true[mask]).mean())
            bin_conf = float(max_p[mask].mean())
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
        m.ece = round(ece, 4)

    return m


# =========================================================================
# Model factory
# =========================================================================
def make_model(name: str, n_classes: int = 2, seed: int = 42) -> Any:
    """Create a model by short name.

    Supported: 'rf', 'xgb', 'mlp', 'conv1d', 'cnn_lstm'.
    """
    name = name.lower().strip()

    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
    elif name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=seed, n_jobs=-1, verbosity=0,
        )
    elif name in ("mlp", "conv1d", "cnn_lstm"):
        return _make_torch_model(name, n_classes)
    else:
        raise ValueError(f"Unknown model '{name}'. Choose: rf, xgb, mlp, conv1d, cnn_lstm")


def _make_torch_model(name: str, n_classes: int):
    """Create a PyTorch model. Requires torch."""
    import torch
    import torch.nn as nn

    class MLP(nn.Module):
        def __init__(self, in_features, n_cls):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_cls),
            )
            self._in = in_features
        def forward(self, x):
            return self.net(x)

    class Conv1D(nn.Module):
        def __init__(self, n_subcarriers, n_cls):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_subcarriers, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(128, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.BatchNorm1d(128),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(64, n_cls),
            )
            self._n_sc = n_subcarriers
        def forward(self, x):
            b = x.shape[0]
            x = x.view(b, self._n_sc, -1)
            x = self.conv(x).squeeze(-1)
            return self.head(x)

    class CNN_LSTM(nn.Module):
        def __init__(self, n_subcarriers, n_cls):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_subcarriers, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            )
            self.lstm = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, n_cls))
            self._n_sc = n_subcarriers
        def forward(self, x):
            b = x.shape[0]
            x = x.view(b, self._n_sc, -1)
            x = self.conv(x).permute(0, 2, 1)
            _, (h, _) = self.lstm(x)
            h = torch.cat([h[0], h[1]], dim=-1)
            return self.head(h)

    # These are deferred — actual instantiation requires knowing input shape
    return {"mlp": MLP, "conv1d": Conv1D, "cnn_lstm": CNN_LSTM}[name], n_classes


# =========================================================================
# Training
# =========================================================================
def train_model(
    train: CSIDataset,
    test: CSIDataset,
    model: str | Any = "rf",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    verbose: bool = True,
) -> tuple[Any, Metrics]:
    """Train a model and evaluate on test set.

    Parameters
    ----------
    train, test : CSIDataset
    model : short name ('rf','xgb','mlp','conv1d','cnn_lstm') or sklearn/torch instance.
    epochs : for DL models.
    batch_size : for DL models.
    lr : learning rate for DL.
    seed : random seed.
    verbose : print progress.

    Returns
    -------
    (trained_model, Metrics)
    """
    from whispy.core import set_seed
    set_seed(seed)

    X_tr, y_tr = train.X, train.y
    X_te, y_te = test.X, test.y

    if isinstance(model, str):
        model_obj = make_model(model, n_classes=train.n_classes, seed=seed)
    else:
        model_obj = model

    # Check if it's a torch class tuple (deferred init)
    is_torch = False
    if isinstance(model_obj, tuple) and len(model_obj) == 2:
        cls, n_cls = model_obj
        import torch.nn as nn
        if X_tr.ndim == 2:
            n_features = X_tr.shape[1]
        else:
            n_features = X_tr.shape[1]
        if cls.__name__ == "MLP":
            model_obj = cls(n_features, n_cls)
        else:
            model_obj = cls(52, n_cls)  # n_subcarriers=52
        is_torch = True
    else:
        try:
            import torch.nn as nn
            is_torch = isinstance(model_obj, nn.Module)
        except ImportError:
            is_torch = False

    if verbose:
        name = type(model_obj).__name__ if not isinstance(model_obj, tuple) else str(model_obj)
        print(f"[whispy.train] Model: {name}")
        print(f"[whispy.train] Train: {X_tr.shape}  Test: {X_te.shape}  Classes: {train.n_classes}")

    if is_torch:
        model_obj, metrics = _train_torch(model_obj, X_tr, y_tr, X_te, y_te,
                                           epochs=epochs, batch_size=batch_size,
                                           lr=lr, verbose=verbose)
    else:
        model_obj, metrics = _train_sklearn(model_obj, X_tr, y_tr, X_te, y_te, verbose=verbose)

    if verbose:
        print(metrics.summary())

    return model_obj, metrics


def _train_sklearn(model, X_tr, y_tr, X_te, y_te, verbose=True) -> tuple[Any, Metrics]:
    if verbose:
        print(f"  Fitting {type(model).__name__} ...")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y_te, y_pred, y_prob=y_prob, train_time=train_time)
    return model, metrics


def _train_torch(model, X_tr, y_tr, X_te, y_te,
                 epochs=50, batch_size=64, lr=1e-3, verbose=True) -> tuple[Any, Metrics]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if verbose:
        print(f"  Training on {device} for {epochs} epochs ...")

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        if verbose and (ep + 1) % max(1, epochs // 5) == 0:
            print(f"    Epoch {ep+1:3d}/{epochs}  loss={total_loss/total:.4f}  "
                  f"acc={correct/total:.4f}")
    train_time = time.time() - t0

    # Evaluate
    model.eval()
    all_pred, all_prob = [], []
    test_ds = TensorDataset(torch.FloatTensor(X_te), torch.LongTensor(y_te))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb)
            prob = torch.softmax(out, dim=1)
            all_pred.append(out.argmax(1).cpu().numpy())
            all_prob.append(prob.cpu().numpy())

    y_pred = np.concatenate(all_pred)
    y_prob = np.concatenate(all_prob)
    metrics = compute_metrics(y_te, y_pred, y_prob=y_prob, train_time=train_time)
    return model, metrics
