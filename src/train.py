import os
import subprocess

import hydra
import mlflow
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.datamodule import TrashDataModule
from src.lightning_module import TrashClassifier
from src.model import create_model


def get_git_commit():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"]
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"


def get_device(accelerator_cfg: str):
    if accelerator_cfg == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return accelerator_cfg


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    print(OmegaConf.to_yaml(cfg))

    # reproducibility
    pl.seed_everything(cfg.train.seed, workers=True)

    # device selection
    accelerator = get_device(cfg.train.accelerator)

    # MLflow setup
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    git_commit = get_git_commit()

    datamodule = TrashDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        image_size=cfg.data.image_size,
        seed=cfg.train.seed,
    )

    datamodule.setup()

    model = create_model(
        num_classes=datamodule.num_classes
    )

    lit_model = TrashClassifier(
        model=model,
        lr=cfg.train.lr,
        num_classes=datamodule.num_classes,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.checkpoint_dir,
        filename="best-model",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=10,
        callbacks=[
            checkpoint_callback,
            lr_monitor,
        ],
    )

    with mlflow.start_run():

        mlflow.log_params(
            {
                "epochs": cfg.train.epochs,
                "lr": cfg.train.lr,
                "batch_size": cfg.train.batch_size,
                "image_size": cfg.data.image_size,
                "model": "resnet18",
                "accelerator": accelerator,
                "git_commit": git_commit,
            }
        )

        trainer.fit(lit_model, datamodule)

        trainer.test(lit_model, datamodule)

        # log best model
        mlflow.log_artifact(
            checkpoint_callback.best_model_path
        )

    print("Training completed.")


if __name__ == "__main__":
    main()