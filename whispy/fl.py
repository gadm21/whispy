"""Federated Learning module — Flower-based simulation.

Supports FedAvg, FedProx, FedAdam, FedYogi, and custom strategies.
Uses Dirichlet partitioning for non-IID client data splits.

Usage:
    from whispy.load import load
    from whispy.fl import run_fl_simulation

    train, test = load("OfficeLocalization")
    history = run_fl_simulation(train, test, strategy="fedavg", n_clients=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from whispy.load import CSIDataset


# =========================================================================
# Partitioning
# =========================================================================
def dirichlet_partition(
    dataset: CSIDataset,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[CSIDataset]:
    """Partition a dataset into n_clients using Dirichlet distribution.

    Lower alpha → more heterogeneous (non-IID).
    """
    rng = np.random.RandomState(seed)
    X, y = dataset.X, dataset.y
    classes = np.unique(y)

    client_indices: list[list[np.ndarray]] = [[] for _ in range(n_clients)]
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * n_clients)
        counts = (proportions * len(cls_idx)).astype(int)
        remainder = len(cls_idx) - counts.sum()
        for i in range(remainder):
            counts[i % n_clients] += 1
        start = 0
        for pid in range(n_clients):
            client_indices[pid].append(cls_idx[start : start + counts[pid]])
            start += counts[pid]

    clients = []
    for pid in range(n_clients):
        idx = np.concatenate(client_indices[pid])
        np.sort(idx)
        clients.append(CSIDataset(
            X=X[idx], y=y[idx],
            label_map=dataset.label_map,
            name=f"{dataset.name}_client{pid}",
            split="train",
        ))
    return clients


def iid_partition(
    dataset: CSIDataset,
    n_clients: int,
    seed: int = 42,
) -> list[CSIDataset]:
    """Partition dataset into n_clients with IID sampling."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(dataset.n_samples)
    splits = np.array_split(idx, n_clients)
    return [
        CSIDataset(
            X=dataset.X[s], y=dataset.y[s],
            label_map=dataset.label_map,
            name=f"{dataset.name}_client{i}",
            split="train",
        )
        for i, s in enumerate(splits)
    ]


# =========================================================================
# Flower client factory
# =========================================================================
def _make_flower_client(client_data: CSIDataset, test_data: CSIDataset,
                        model_name: str, seed: int):
    """Create a Flower NumPyClient for sklearn models."""
    try:
        import flwr as fl
    except ImportError:
        raise ImportError("Flower required — pip install whispy[fl]")

    from whispy.train import make_model, compute_metrics

    class WhispyClient(fl.client.NumPyClient):
        def __init__(self):
            self.model = make_model(model_name, n_classes=client_data.n_classes, seed=seed)
            self.X_train = client_data.X
            self.y_train = client_data.y
            self.X_test = test_data.X
            self.y_test = test_data.y

        def get_parameters(self, config):
            # For tree models: serialize feature importances as proxy
            if hasattr(self.model, "feature_importances_"):
                return [self.model.feature_importances_]
            return []

        def fit(self, parameters, config):
            self.model.fit(self.X_train, self.y_train)
            return self.get_parameters(config), len(self.X_train), {}

        def evaluate(self, parameters, config):
            y_pred = self.model.predict(self.X_test)
            y_prob = (self.model.predict_proba(self.X_test)
                      if hasattr(self.model, "predict_proba") else None)
            m = compute_metrics(self.y_test, y_pred, y_prob=y_prob)
            return float(1 - m.accuracy), len(self.X_test), {
                "accuracy": m.accuracy,
                "kappa": m.kappa,
            }

    return WhispyClient()


# =========================================================================
# Strategy factory
# =========================================================================
STRATEGIES = ["fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad"]


def _make_strategy(name: str, **kwargs) -> Any:
    """Create a Flower strategy by name."""
    try:
        import flwr as fl
    except ImportError:
        raise ImportError("Flower required — pip install whispy[fl]")

    name = name.lower().strip()
    common = dict(
        min_fit_clients=kwargs.get("min_clients", 2),
        min_evaluate_clients=kwargs.get("min_clients", 2),
        min_available_clients=kwargs.get("n_clients", 2),
    )

    if name == "fedavg":
        return fl.server.strategy.FedAvg(**common)
    elif name == "fedprox":
        return fl.server.strategy.FedProx(proximal_mu=kwargs.get("mu", 0.1), **common)
    elif name == "fedadam":
        return fl.server.strategy.FedAdam(
            eta=kwargs.get("eta", 1e-2),
            eta_l=kwargs.get("eta_l", 1e-2),
            tau=kwargs.get("tau", 1e-3),
            **common,
        )
    elif name == "fedyogi":
        return fl.server.strategy.FedYogi(
            eta=kwargs.get("eta", 1e-2),
            eta_l=kwargs.get("eta_l", 1e-2),
            tau=kwargs.get("tau", 1e-3),
            **common,
        )
    elif name == "fedadagrad":
        return fl.server.strategy.FedAdagrad(
            eta=kwargs.get("eta", 1e-2),
            eta_l=kwargs.get("eta_l", 1e-2),
            tau=kwargs.get("tau", 1e-3),
            **common,
        )
    else:
        raise ValueError(f"Unknown strategy '{name}'. Choose from: {STRATEGIES}")


# =========================================================================
# Simulation entry point
# =========================================================================
@dataclass
class FLResult:
    strategy: str
    n_clients: int
    n_rounds: int
    partition: str
    alpha: float
    round_accuracies: list[float]
    final_accuracy: float


def run_fl_simulation(
    train: CSIDataset,
    test: CSIDataset,
    strategy: str = "fedavg",
    n_clients: int = 4,
    n_rounds: int = 5,
    partition: str = "dirichlet",
    alpha: float = 0.5,
    model: str = "rf",
    seed: int = 42,
    verbose: bool = True,
    **strategy_kwargs,
) -> FLResult:
    """Run a Flower FL simulation.

    Parameters
    ----------
    train, test : CSIDataset
    strategy : one of STRATEGIES.
    n_clients : number of simulated clients.
    n_rounds : communication rounds.
    partition : 'dirichlet' or 'iid'.
    alpha : Dirichlet concentration (lower = more non-IID).
    model : model short name ('rf', 'xgb').
    seed : random seed.
    verbose : print progress.

    Returns
    -------
    FLResult with per-round accuracies.
    """
    try:
        import flwr as fl
    except ImportError:
        raise ImportError("Flower required — pip install whispy[fl]")

    if verbose:
        print(f"[whispy.fl] Strategy={strategy}  Clients={n_clients}  "
              f"Rounds={n_rounds}  Partition={partition}(α={alpha})")

    # Partition
    if partition == "dirichlet":
        clients_data = dirichlet_partition(train, n_clients, alpha=alpha, seed=seed)
    elif partition == "iid":
        clients_data = iid_partition(train, n_clients, seed=seed)
    else:
        raise ValueError(f"Unknown partition '{partition}'. Choose: dirichlet, iid")

    if verbose:
        for i, cd in enumerate(clients_data):
            classes, counts = np.unique(cd.y, return_counts=True)
            print(f"  Client {i}: {cd.n_samples} samples  "
                  f"classes={dict(zip(classes.tolist(), counts.tolist()))}")

    # Build client factory
    def client_fn(cid: str):
        idx = int(cid)
        return _make_flower_client(clients_data[idx], test, model, seed)

    # Strategy
    strat = _make_strategy(strategy, n_clients=n_clients,
                           min_clients=min(2, n_clients), **strategy_kwargs)

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strat,
    )

    # Extract per-round accuracies
    round_accs = []
    if hasattr(history, "metrics_distributed") and "accuracy" in history.metrics_distributed:
        for round_num, values in history.metrics_distributed["accuracy"]:
            round_accs.append(float(np.mean([v for _, v in values])))
    elif hasattr(history, "losses_distributed"):
        for round_num, loss in history.losses_distributed:
            round_accs.append(1.0 - loss)  # approximate

    final_acc = round_accs[-1] if round_accs else 0.0

    if verbose:
        print(f"\n[whispy.fl] Final accuracy: {final_acc:.4f}")
        if round_accs:
            print(f"[whispy.fl] Per-round: {[f'{a:.3f}' for a in round_accs]}")

    return FLResult(
        strategy=strategy,
        n_clients=n_clients,
        n_rounds=n_rounds,
        partition=partition,
        alpha=alpha,
        round_accuracies=round_accs,
        final_accuracy=final_acc,
    )
