from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from torchvision import transforms

from trashsort.datamodule import TrashDataModule
from trashsort.lightning_module import TrashClassifier
from trashsort.model import create_model


def get_device(device_cfg: str):
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device_cfg)


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig):

    device = get_device(cfg.inference.device)

    image_path = Path(cfg.inference.image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # load classes from datamodule
    datamodule = TrashDataModule(
        data_dir=cfg.data.data_dir
    )
    datamodule.setup()

    class_names = datamodule.classes

    model = create_model(
        num_classes=datamodule.num_classes
    )

    checkpoint_path = (
        cfg.inference.checkpoint_path
    )

    lit_model = TrashClassifier.load_from_checkpoint(
        checkpoint_path,
        model=model,
        num_classes=datamodule.num_classes,
        map_location=device,
    )

    lit_model.eval()
    lit_model.to(device)

    transform = transforms.Compose([
        transforms.Resize(
            (cfg.data.image_size,
             cfg.data.image_size)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = (
        Image.open(image_path)
        .convert("RGB")
    )

    x = transform(image).unsqueeze(0)
    x = x.to(device)

    with torch.no_grad():
        logits = lit_model(x)
        prediction = torch.argmax(
            logits,
            dim=1
        ).item()

    predicted_class = class_names[prediction]

    print("\nPrediction:")
    print(f"Class ID: {prediction}")
    print(f"Class Name: {predicted_class}")


if __name__ == "__main__":
    main()