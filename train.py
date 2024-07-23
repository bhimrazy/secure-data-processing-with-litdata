import os

import lightning as L
import torch
import torch.nn as nn
from litdata import StreamingDataLoader, StreamingDataset
from litdata.utilities.encryption import FernetEncryption
from torchmetrics import Accuracy, F1Score
from torchvision import transforms as T

fernet = FernetEncryption.load("fernet.pem", password="your_super_secret_password")

image_transform = T.Compose(
    [
        T.Resize((28, 28)),
        T.ToTensor(),
    ]
)


# define dataset and datamodule
class DermamnistDataset(StreamingDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        image = image_transform(item["image"])
        label = int(item["class"])
        return image, label


class DermamnistDataModule(L.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = DermamnistDataset(
            input_dir="data/dermamnist_optimized/train", encryption=fernet
        )
        self.val_dataset = DermamnistDataset(
            input_dir="data/dermamnist_optimized/validation", encryption=fernet
        )
        self.test_dataset = DermamnistDataset(
            input_dir="data/dermamnist_optimized/test", encryption=fernet
        )

    def train_dataloader(self):
        return StreamingDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() - 1,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return StreamingDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() - 1,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return StreamingDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() - 1,
            persistent_workers=True,
        )


# define model
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * 28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=10)
        self.f1 = F1Score(task="multiclass", num_classes=10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_f1", f1)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.accuracy(y_hat, y)
        f1 = self.f1(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_f1", f1)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


if __name__ == "__main__":
    L.seed_everything(42)
    dm = DermamnistDataModule()
    model = LitModel()
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, dm)
    trainer.test(model, dm.test_dataloader())
