import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from trashsort.datamodule import TrashDataModule
from trashsort.lightning_module import TrashClassifier
from trashsort.model import create_model

app = FastAPI(title="TrashSort API")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# load datamodule just to get classes
dm = TrashDataModule()
dm.setup()

model = create_model(num_classes=dm.num_classes)

lit_model = TrashClassifier.load_from_checkpoint(
    "models/checkpoints/best-model.ckpt",
    model=model,
    map_location=DEVICE,
)

lit_model.eval()


def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    x = torch.tensor(torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes())))
    x = x.float() / 255.0
    x = x.view(1, 3, 224, 224)
    return x.to(DEVICE)


@app.get("/")
def root():
    return {"message": "TrashSort API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")

    x = preprocess(image)

    with torch.no_grad():
        logits = lit_model(x)
        pred = torch.argmax(logits, dim=1).item()

    return {
        "class_id": pred,
        "class_name": dm.classes[pred],
    }
