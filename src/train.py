import lightning as lit
from configs.config import config
from src.model import ModelLit
from src.data import DeepFakeDataModule

model = ModelLit()
data = DeepFakeDataModule(
    train_dir=config["data"]["train_path"],
    valid_dir=config["data"]["valid_path"],
    test_dir=config["data"]["test_path"],
    batch_size=config["training"]["batch_size"]
)

trainer = lit.Trainer(devices=1, max_epochs=config["training"]["epochs"])
trainer.fit(model, datamodule=data)
