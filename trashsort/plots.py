from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI = "http://127.0.0.1:8080"
EXPERIMENT_NAME = "trashsort"


def create_plots():
    mlflow.set_tracking_uri(MLFLOW_URI)

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError("Experiment not found")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
    )

    latest_run = runs.iloc[0]
    run_id = latest_run["run_id"]

    # client = MlflowClient()
    # run = client.get_run(run_id)
    # print("\n=== AVAILABLE METRICS ===")
    # for k in run.data.metrics.keys():
    #     print(k)
    client = MlflowClient()

    train_loss = client.get_metric_history(run_id, "train_loss")

    val_loss = client.get_metric_history(run_id, "val_loss")

    train_acc = client.get_metric_history(run_id, "train_acc")

    val_acc = client.get_metric_history(run_id, "val_acc")

    Path("plots").mkdir(exist_ok=True)

    # ---- LOSS CURVE ----
    plt.figure(figsize=(8, 5))

    if train_loss:
        plt.plot(
            [m.step for m in train_loss],
            [m.value for m in train_loss],
            marker="o",
            label="train_loss",
        )

    if val_loss:
        plt.plot(
            [m.step for m in val_loss],
            [m.value for m in val_loss],
            marker="o",
            label="val_loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig("plots/loss_curve.png")
    plt.close()

    # ---- ACCURACY CURVE ----
    plt.figure(figsize=(8, 5))

    if train_acc:
        plt.plot(
            [m.step for m in train_acc],
            [m.value for m in train_acc],
            marker="o",
            label="train_acc",
        )

    if val_acc:
        plt.plot(
            [m.step for m in val_acc],
            [m.value for m in val_acc],
            marker="o",
            label="val_acc",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()

    plt.savefig("plots/accuracy_curve.png")
    plt.close()

    # ---- FINAL METRICS ----
    plt.figure(figsize=(8, 5))

    metrics = {
        "train_loss": train_loss[-1].value if train_loss else 0,
        "val_loss": val_loss[-1].value if val_loss else 0,
        "train_acc": train_acc[-1].value if train_acc else 0,
        "val_acc": val_acc[-1].value if val_acc else 0,
    }

    plt.bar(
        metrics.keys(),
        metrics.values(),
    )

    plt.title("Final Metrics")
    plt.tight_layout()

    plt.savefig("plots/metrics_summary.png")
    plt.close()

    print("Plots saved in /plots")


if __name__ == "__main__":
    create_plots()
