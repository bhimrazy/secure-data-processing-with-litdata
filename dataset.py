import torch
import numpy as np
from litdata import StreamingDataset, StreamingDataLoader
from litdata.utilities.encryption import FernetEncryption

fernet = FernetEncryption.load("fernet.pem", password="your_super_secret_password")


def collate_fn(batch):
    """Collate function for the streaming data loader."""
    images = np.array([np.array(item["image"]) for item in batch])
    classes = [item["class"] for item in batch]

    images_tensor = torch.tensor(images, dtype=torch.float32)
    classes_tensor = torch.tensor(classes, dtype=torch.long)

    return {"image": images_tensor, "class": classes_tensor}


def get_dataloader(
    dataset: StreamingDataset, batch_size: int, shuffle: bool
) -> StreamingDataLoader:
    """Helper function to create a dataloader."""
    return StreamingDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )


def main():
    # Create streaming datasets
    train_dataset = StreamingDataset(
        input_dir="data/dermamnist_optimized/train", encryption=fernet
    )
    valid_dataset = StreamingDataset(
        input_dir="data/dermamnist_optimized/validation", encryption=fernet
    )
    test_dataset = StreamingDataset(
        input_dir="data/dermamnist_optimized/test", encryption=fernet
    )

    # Create dataloaders
    train_loader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = get_dataloader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # Log dataset sizes
    print("Number of training images:", len(train_dataset))
    print("Number of validation images:", len(valid_dataset))
    print("Number of test images:", len(test_dataset))

    # Log example batches
    train_batch = next(iter(train_loader))
    print("Train Batch classes:", train_batch["class"])

    valid_batch = next(iter(valid_loader))
    print("Validation Batch classes:", valid_batch["class"])

    test_batch = next(iter(test_loader))
    print("Test Batch classes:", test_batch["class"])


if __name__ == "__main__":
    main()
