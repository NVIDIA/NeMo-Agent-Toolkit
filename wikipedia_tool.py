import requests
import pandas as pd
import time
import logging
from io import BytesIO

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)

# =========================
# CONFIG
# =========================
TOPICS = [
    "Artificial Intelligence",
    "Robot",
    "Machine Learning",
    "Deep Learning",
    "Computer Vision"
]

IMAGES_PER_TOPIC = 10
OUTPUT_FILE = "data.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# DATA SCRAPER
# =========================
def get_image_titles(title, limit=10):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "titles": title,
        "prop": "images",
        "format": "json"
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
    except:
        return []

    images = []
    pages = data["query"]["pages"]

    for page_id in pages:
        if "images" in pages[page_id]:
            for img in pages[page_id]["images"]:
                name = img["title"]

                if any(ext in name.lower() for ext in [".jpg", ".jpeg", ".png"]):
                    images.append(name)

    return images[:limit]


def get_image_url(image_title):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "titles": image_title,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }

    try:
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
    except:
        return None

    pages = data["query"]["pages"]

    for page_id in pages:
        if "imageinfo" in pages[page_id]:
            return pages[page_id]["imageinfo"][0]["url"]

    return None


def build_dataset(topics, images_per_topic):
    dataset = []

    for label, topic in enumerate(topics):
        logging.info(f"Processing: {topic}")

        image_titles = get_image_titles(topic, images_per_topic)

        count = 0
        for img_title in image_titles:
            img_url = get_image_url(img_title)

            if img_url:
                dataset.append({
                    "label": label,
                    "topic": topic,
                    "image_url": img_url
                })
                count += 1

            if count >= images_per_topic:
                break

            time.sleep(0.3)

        logging.info(f"Collected {count} images for {topic}")

    return pd.DataFrame(dataset)


# =========================
# DATASET CLASS
# =========================
class AIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        try:
            response = requests.get(row["image_url"], timeout=5)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))

        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


# =========================
# MODEL (NVIDIA STYLE)
# =========================
class NvidiaModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.vit_b_16(pretrained=True)
        self.model.heads.head = nn.Linear(
            self.model.heads.head.in_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


# =========================
# TRAINING
# =========================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# MAIN TRAIN PIPELINE
# =========================
def train_pipeline():
    df = build_dataset(TOPICS, IMAGES_PER_TOPIC)
    df.to_csv(OUTPUT_FILE, index=False)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = AIDataset(OUTPUT_FILE, transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = NvidiaModel(num_classes=len(TOPICS)).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        loss = train(model, loader, optimizer, criterion)
        logging.info(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    torch.save(model.state_dict(), "model.pth")
    return model


# =========================
# API (REAL SYSTEM)
# =========================
app = Flask(__name__)

model = NvidiaModel(num_classes=len(TOPICS))
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]

    image = Image.open(file).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)

    return jsonify({
        "prediction": TOPICS[pred.item()]
    })


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)