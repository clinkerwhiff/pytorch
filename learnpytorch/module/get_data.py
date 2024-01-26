import os
import requests
import zipfile

from pathlib import Path

# Defining paths to data folders
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# Checking if folder exists
if image_path.is_dir():
    print(f"{image_path} directory exists")
else:
    print(f"Creating {image_path} directory")
    image_path.mkdir(parents=True, exist_ok=True)

# Downloading zip file
with open(data_path / "pizza_steak_sushi.zip", "wb") as file:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading data")
    file.write(request.content)

# Unzipping zip file
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping data")
    zip_ref.extractall(image_path)

# Deleting zip file
os.remove(data_path / "pizza_steak_sushi.zip")