import os
import pandas as pd
from PIL import Image
import logging
from litdata import optimize
from litdata.utilities.encryption import FernetEncryption
from typing import List, Dict


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


def optimize_data(records: List[Dict], output_dir: str, fernet: FernetEncryption):
    """Optimize data using the provided records and encryption."""
    logging.info(f"Optimizing data and saving to {output_dir}")
    optimize(
        fn=read_data,
        inputs=records,
        output_dir=output_dir,
        chunk_bytes="60MB",
        num_workers=4,
        encryption=fernet,
    )


def prepare():
    """Main preparation function."""
    # Load the data
    df = load_data("data/dermamnist.csv")
    data_splits = split_data(df)

    # Set up encryption
    fernet = FernetEncryption(password="mysecretkey", level="chunk")

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
