"""Click-based CLI for Whispy.

Entry point: `whispy <command> [options]`

Commands:
    collect   — Collect standardized CSI data from ESP32
    deploy    — Run live inference on CSI stream (no file saved)
    load      — Download a built-in dataset from HuggingFace
    train     — Train a model on a dataset
    vis       — Visualize results or live CSI stream
    fl        — Run federated learning simulation
    info      — Show package info and available datasets
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

console = Console()


# =========================================================================
# Main group
# =========================================================================
@click.group()
@click.version_option(package_name="whispy")
def main():
    """🔮 Whispy — WiFi Intelligence on ESP.

    CSI sensing toolkit for collection, training, visualization, and FL.
    """
    pass


# =========================================================================
# info
# =========================================================================
@main.command()
def info():
    """Show package info and available built-in datasets."""
    from whispy import __version__
    from whispy.load import DATASETS
    from whispy.fl import STRATEGIES

    console.print(f"\n[bold cyan]🔮 Whispy v{__version__}[/bold cyan]")
    console.print("WiFi Intelligence on ESP — CSI sensing toolkit\n")

    table = Table(title="Built-in Datasets (gadgadgad/*)")
    table.add_column("Name", style="bold")
    table.add_column("Task")
    table.add_column("Env")
    table.add_column("Classes", justify="right")
    table.add_column("Files", justify="right")

    for name, ds in DATASETS.items():
        table.add_row(name, ds.task, ds.environment,
                      str(ds.n_classes), str(len(ds.files)))
    console.print(table)

    console.print(f"\n[bold]Models:[/bold] rf, xgb, mlp, conv1d, cnn_lstm")
    console.print(f"[bold]Pipelines:[/bold] amplitude, rv20, rv200, rv2000")
    console.print(f"[bold]FL Strategies:[/bold] {', '.join(STRATEGIES)}")
    console.print(f"[bold]Face detectors:[/bold] haar, dnn\n")


# =========================================================================
# collect
# =========================================================================
@main.command()
@click.option("--port", required=True, help="Serial port of receiver ESP32")
@click.option("--label", default=None, help="Activity label (default: indoor)")
@click.option("--duration", default=60.0, help="Duration per file in seconds")
@click.option("--total", default=None, type=float, help="Total collection time (auto-splits into files)")
@click.option("--sr", default=150, help="Guaranteed sampling rate (Hz)")
@click.option("--out", default="./whispy_data", help="Output directory")
@click.option("--camera/--no-camera", default=False, help="Enable face detection")
@click.option("--detector", default="haar", type=click.Choice(["haar", "dnn"]))
@click.option("--baud", default=115200, help="Serial baud rate")
@click.option("--pause", default=2.0, help="Pause between files (seconds)")
def collect(port, label, duration, total, sr, out, camera, detector, baud, pause):
    """Collect standardized CSI data from ESP32.

    Produces 1-minute .npz files with uniform sampling, guard removal,
    and per-file metadata. Optionally enables face detection (--camera).
    """
    from whispy.collect import collect as _collect
    _collect(
        port=port, label=label, duration=duration,
        total_duration=total, baud=baud, sr=sr,
        out_dir=out, camera=camera, detector=detector, pause=pause,
    )


# =========================================================================
# deploy
# =========================================================================
@main.command()
@click.option("--port", required=True, help="Serial port of receiver ESP32")
@click.option("--model", required=True, help="Path to trained model (.pkl)")
@click.option("--baud", default=115200)
@click.option("--sr", default=150)
@click.option("--var-window", default=20, help="Rolling variance window")
@click.option("--window-len", default=500, help="Classification window length")
def deploy(port, model, baud, sr, var_window, window_len):
    """Run live inference on CSI stream (no file saved).

    Reads CSI from serial, applies rolling variance + windowing,
    and prints model predictions to terminal in real time.
    """
    from whispy.collect import deploy as _deploy
    _deploy(port=port, model_path=model, baud=baud, sr=sr,
            var_window=var_window, window_len=window_len)


# =========================================================================
# load
# =========================================================================
@main.command()
@click.argument("name")
@click.option("--out", default=None, help="Output directory (default: ~/.cache/whispy)")
@click.option("--window-len", default=500, help="Window length")
@click.option("--var-window", default=20, help="Rolling variance window")
@click.option("--sr", default=150, help="Sampling rate")
def load(name, out, window_len, var_window, sr):
    """Download a built-in dataset from HuggingFace.

    Available datasets: OfficeLocalization, OfficeHAR, HomeHAR, HomeOccupation
    """
    from whispy.load import load as _load, list_datasets

    if name not in list_datasets():
        console.print(f"[red]Unknown dataset '{name}'[/red]")
        console.print(f"Available: {', '.join(list_datasets())}")
        return

    train, test = _load(
        name, guaranteed_sr=sr, window_len=window_len,
        var_window=var_window, cache_dir=out, verbose=True,
    )
    console.print(f"\n[green]✓[/green] Train: {train.X.shape}  Test: {test.X.shape}")
    console.print(f"  Classes: {train.n_classes}  Labels: {train.labels}")


# =========================================================================
# train
# =========================================================================
@main.command()
@click.option("--data", required=True, help="Dataset name (built-in) or path to data dir")
@click.option("--pipeline", default="rv20", help="Pipeline name: amplitude, rv20, rv200, rv2000")
@click.option("--model", default="rf", help="Model: rf, xgb, mlp, conv1d, cnn_lstm")
@click.option("--window-len", default=500)
@click.option("--sr", default=150)
@click.option("--epochs", default=50, help="Epochs (DL only)")
@click.option("--seed", default=42)
@click.option("--save", default=None, help="Save trained model to path (.pkl)")
def train(data, pipeline, model, window_len, sr, epochs, seed, save):
    """Train a model on a dataset.

    Examples:
        whispy train --data OfficeLocalization --model rf
        whispy train --data OfficeLocalization --pipeline rv200 --model conv1d --epochs 100
    """
    from whispy.load import load as _load, list_datasets
    from whispy.pipeline import Pipeline
    from whispy.train import train_model

    # Load dataset
    if data in list_datasets():
        pipe = Pipeline.from_name(pipeline)
        # Override window length in pipeline
        for step in pipe.steps:
            if hasattr(step, "length"):
                step.length = window_len
                step.stride = window_len
        train_ds, test_ds = _load(data, pipeline=pipe, guaranteed_sr=sr, verbose=True)
    else:
        console.print(f"[red]Dataset '{data}' not found as built-in.[/red]")
        console.print(f"Available: {', '.join(list_datasets())}")
        return

    # Train
    model_obj, metrics = train_model(
        train_ds, test_ds, model=model, epochs=epochs, seed=seed, verbose=True,
    )

    console.print(f"\n[bold green]Results:[/bold green]")
    console.print(metrics.summary())

    if save:
        import pickle
        with open(save, "wb") as f:
            pickle.dump(model_obj, f)
        console.print(f"\n[green]✓[/green] Model saved to {save}")


# =========================================================================
# vis
# =========================================================================
@main.command()
@click.option("--live", is_flag=True, help="Live CSI stream visualization")
@click.option("--port", default=None, help="Serial port (for --live)")
@click.option("--data", default=None, help="Dataset name for PCA scatter")
@click.option("--pipeline", default="rv20")
@click.option("--sr", default=150)
@click.option("--baud", default=115200)
def vis(live, port, data, pipeline, sr, baud):
    """Visualize CSI data — live stream or dataset PCA scatter."""
    if live:
        if not port:
            console.print("[red]--port required for live mode[/red]")
            return
        from whispy.vis import plot_live_stream
        plot_live_stream(port=port, baud=baud)
    elif data:
        from whispy.load import load as _load, list_datasets
        from whispy.pipeline import Pipeline
        from whispy.vis import plot_pca_scatter
        if data not in list_datasets():
            console.print(f"[red]Unknown dataset '{data}'[/red]")
            return
        pipe = Pipeline.from_name(pipeline)
        train_ds, _ = _load(data, pipeline=pipe, guaranteed_sr=sr, verbose=True)
        inv = {v: k for k, v in train_ds.label_map.items()}
        labels = [inv[i] for i in range(train_ds.n_classes)]
        plot_pca_scatter(train_ds.X, train_ds.y, labels=labels,
                         title=f"{data} — PCA 2D", show=True)
    else:
        console.print("[yellow]Specify --live --port <PORT> or --data <DATASET>[/yellow]")


# =========================================================================
# fl
# =========================================================================
@main.command()
@click.option("--data", required=True, help="Built-in dataset name")
@click.option("--strategy", default="fedavg",
              type=click.Choice(["fedavg", "fedprox", "fedadam", "fedyogi", "fedadagrad"]))
@click.option("--clients", default=4, help="Number of simulated clients")
@click.option("--rounds", default=5, help="Communication rounds")
@click.option("--partition", default="dirichlet", type=click.Choice(["dirichlet", "iid"]))
@click.option("--alpha", default=0.5, help="Dirichlet alpha (lower = more non-IID)")
@click.option("--model", default="rf", help="Model: rf, xgb")
@click.option("--pipeline", default="rv20")
@click.option("--sr", default=150)
@click.option("--seed", default=42)
def fl(data, strategy, clients, rounds, partition, alpha, model, pipeline, sr, seed):
    """Run federated learning simulation.

    Example:
        whispy fl --data OfficeLocalization --strategy fedavg --clients 4 --rounds 10
    """
    from whispy.load import load as _load, list_datasets
    from whispy.pipeline import Pipeline
    from whispy.fl import run_fl_simulation

    if data not in list_datasets():
        console.print(f"[red]Unknown dataset '{data}'[/red]")
        return

    pipe = Pipeline.from_name(pipeline)
    train_ds, test_ds = _load(data, pipeline=pipe, guaranteed_sr=sr, verbose=True)

    result = run_fl_simulation(
        train_ds, test_ds,
        strategy=strategy, n_clients=clients, n_rounds=rounds,
        partition=partition, alpha=alpha, model=model, seed=seed,
        verbose=True,
    )

    console.print(f"\n[bold green]FL Result:[/bold green]")
    console.print(f"  Strategy:  {result.strategy}")
    console.print(f"  Clients:   {result.n_clients}")
    console.print(f"  Rounds:    {result.n_rounds}")
    console.print(f"  Final Acc: {result.final_accuracy:.4f}")
    if result.round_accuracies:
        console.print(f"  Per-round: {[f'{a:.3f}' for a in result.round_accuracies]}")


if __name__ == "__main__":
    main()
