import pandas as pd
from PIL import Image
from litdata import optimize
from litdata.utilities.encryption import FernetEncryption

# Load the CSV file
DATA_DIR = "data/dermamnist"
CSV_PATH = "data/dermamnist.csv"
df = pd.read_csv(CSV_PATH, header=None)
df.columns = ["split", "image", "class"]

# Convert the dataframe to a list of dictionaries
train_records = df[df["split"] == "TRAIN"].to_dict(orient="records")
valid_records = df[df["split"] == "VALIDATION"].to_dict(orient="records")
test_records = df[df["split"] == "TEST"].to_dict(orient="records")


def read_data(record):
    image = Image.open(f"{DATA_DIR}/{record['image']}")
    return {"image": image, "class": record["class"], "split": record["split"]}


if __name__ == "__main__":
    fernet = FernetEncryption(password="mysecretkey", level="chunk")
    # optimize train data
    optimize(
        fn=read_data,
        inputs=train_records,
        output_dir="data/dermamnist_optimized/train",
        chunk_bytes="60MB",
        num_workers=4,
        encryption=fernet,
    )

    # optimize validation data
    optimize(
        fn=read_data,
        inputs=valid_records,
        output_dir="data/dermamnist_optimized/validation",
        chunk_bytes="60MB",
        num_workers=4,
        encryption=fernet,
    )

    # optimize test data
    optimize(
        fn=read_data,
        inputs=test_records,
        output_dir="data/dermamnist_optimized/test",
        chunk_bytes="60MB",
        num_workers=4,
        encryption=fernet,
    )

    # save the fernet key
    fernet.save("fernet_key.pem")
