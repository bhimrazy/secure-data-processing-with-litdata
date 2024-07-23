# Unlock Secure Data Processing at Scale with LitData's Advanced Encryption Features

In today's data-driven world, processing large datasets securely and efficiently is more crucial than ever. Enter [LitData](https://github.com/Lightning-AI/litdata), a powerful Python library that's revolutionizing how we handle and optimize big data. Let's dive into how LitData's advanced encryption features can help you process data at scale while maintaining top-notch security.

<div align="center">
    <img src="https://github.com/user-attachments/assets/6bb2474d-2c0c-42af-8c6e-43c495034341" alt="Cover Image" width="640" height="360" style="object-fit: cover;">
</div>

## 🔒 Why Encryption Matters in Data Processing

Before we jump into the technical details, let's consider why encryption is so important:

1. **Data Privacy**: Protect sensitive information from unauthorized access.
2. **Compliance**: Meet regulatory requirements like GDPR or HIPAA.
3. **Intellectual Property**: Safeguard valuable datasets and algorithms.
4. **Trust**: Build confidence with clients and stakeholders.

## 🚀 Introducing LitData's Advanced Encryption Capabilities

LitData takes data security to the next level by offering powerful and flexible encryption options designed for secure, efficient data processing at scale:

1. **Flexible Encryption Levels**

   - Sample-level: Encrypt each data point individually
   - Chunk-level: Encrypt groups of samples for balanced security and performance

2. **Always-Encrypted Data**

   - Data remains encrypted at rest and in transit
   - On-the-fly decryption only when data is actively used

3. **Key Advantages**

   - Process parts of your dataset without decrypting everything
   - Minimize attack surface with data encrypted most of the time
   - Optimize performance with customizable encryption granularity
   - Seamless integration with existing data pipelines
   - Cloud-ready security for protected data wherever it resides

4. **Efficient Resource Use**
   - Decrypt only what's needed, when it's needed
   - Decrypted data doesn't persist in memory, enhancing security

With these features, LitData empowers you to work with sensitive data at scale, maintaining high security standards without compromising on performance or ease of use.

## 💡 How to Use LitData's Encryption Features

In this guide, we'll walk through a practical example of using LitData's encryption features with the `dermamnist` dataset from the [`medmnist`](https://medmnist.com/) library.

### Prerequisites

First, install the required packages and download the dataset:

```bash
pip install -r requirements.txt
python -m medmnist save --flag=dermamnist --folder=data/ --postfix=png --download=True --size=28
```

### Optimizing and Encrypting Data

Create a Python script named `optimize_medmnist.py`:

```python
# optimize_medmnist.py
import logging
from typing import Dict, List

import pandas as pd
from litdata import optimize
from litdata.utilities.encryption import FernetEncryption
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the CSV file."""
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path, header=None)
    df.columns = ["split", "image", "class"]
    return df


def split_data(df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Split the data into training, validation, and test sets."""
    logging.info("Splitting data into training, validation, and test sets")
    return {
        "train": df[df["split"] == "TRAIN"].to_dict(orient="records"),
        "validation": df[df["split"] == "VALIDATION"].to_dict(orient="records"),
        "test": df[df["split"] == "TEST"].to_dict(orient="records"),
    }


def read_data(record: Dict) -> Dict:
    """Read and preprocess an image record."""
    try:
        image = Image.open(f"data/dermamnist/{record['image']}")
        return {"image": image, "class": record["class"], "split": record["split"]}
    except Exception as e:
        logging.error(f"Error reading image {record['image']}: {e}")
        return {}


def optimize_data(records: List[Dict], output_dir: str, encryption: FernetEncryption):
    """Optimize data using the provided records and encryption."""
    logging.info(f"Optimizing data and saving to {output_dir}")
    optimize(
        fn=read_data,
        inputs=records,
        output_dir=output_dir,
        chunk_bytes="60MB",
        num_workers=2,
        encryption=encryption,
    )


def prepare():
    """Main preparation function."""
    # Load the data
    df = load_data("data/dermamnist.csv")
    data_splits = split_data(df)

    # Set up encryption
    fernet = FernetEncryption(password="your_super_secret_password", level="chunk")

    # Optimize data
    optimize_data(data_splits["train"], "data/dermamnist_optimized/train", fernet)
    optimize_data(
        data_splits["validation"], "data/dermamnist_optimized/validation", fernet
    )
    optimize_data(data_splits["test"], "data/dermamnist_optimized/test", fernet)

    # Save the encryption key
    logging.info("Saving encryption key to fernet.pem")
    fernet.save("fernet.pem")


if __name__ == "__main__":
    prepare()
```

Run the script:

```bash
python optimize_medmnist.py
```

## 🔍 Decrypting and Using the Data

When it's time to use your encrypted data, LitData makes it just as easy:
To test it, create a script named `test_dataset.py`:

```python
# test_dataset.py
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
```

Run the script:

```bash
python test_dataset.py

# Number of training images: 7007
# Number of validation images: 1003
# Number of test images: 2005
# ...
```

And that's it! You've successfully encrypted, optimized, and used your data with LitData's advanced encryption features.

You can further test by training a model using the PyTorch Lightning library:

```bash
python train.py
```

## 🌟 Benefits of LitData's Approach

1. **Scalability**: Process enormous datasets without compromising on security.
2. **Cloud-Ready**: Works seamlessly with cloud storage solutions like AWS S3.
3. **Customizable**: Implement your own encryption methods by subclassing `Encryption`.
4. **Efficient**: Optimized for fast data loading and processing.

## 🚧 Best Practices for Secure Data Processing

1. **Key Management**: Store encryption keys securely, separate from the data.
2. **Regular Rotation**: Change encryption keys periodically for enhanced security.
3. **Access Control**: Limit who can decrypt and access the sensitive data.
4. **Audit Trails**: Keep logs of data access and processing for compliance.

## 🔮 The Future of Secure Data Processing

As datasets grow larger and privacy concerns intensify, tools like LitData will become increasingly crucial. By combining efficient data processing with robust security features, LitData is at the forefront of this evolution.

## 🎓 Conclusion

LitData's advanced encryption features offer a powerful solution for organizations looking to process large datasets securely and efficiently. By leveraging sample-level encryption and cloud-ready streaming, you can unlock new possibilities in data science and machine learning while maintaining the highest standards of data protection.

Ready to supercharge your data processing with top-tier security? Give LitData a try today!

---

Remember, security is an ongoing process. Stay updated with the latest best practices and keep your data safe!

For more information on LitData and its features, check out the official documentation and [GitHub](https://github.com/Lightning-AI/litdata) repository.

Happy encrypting! 🔒✨

## Author

- [Bhimraj Yadav](https://github.com/bhimrazy)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
