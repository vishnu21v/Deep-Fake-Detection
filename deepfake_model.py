# model.py
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(backbone="tf_efficientnet_b0"):
    m = timm.create_model(backbone, pretrained=False, num_classes=0, global_pool="avg")
    n_features = m.num_features
    classifier = nn.Sequential(
        nn.Linear(n_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    model = nn.Sequential(m, classifier)
    return model

class DeepfakeModel:
    def __init__(self, checkpoint_path="models/deepfake_tf_efficientnet_b0_best.pth", img_size=224):
        self.img_size = img_size
        ck = torch.load(checkpoint_path, map_location=DEVICE)
        backbone = ck.get("backbone", "tf_efficientnet_b0")
        self.model = build_model(backbone=backbone)
        self.model.load_state_dict(ck["model_state"])
        self.model.to(DEVICE)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).cpu().item()  # probability of class 'fake'
        label = "fake" if prob >= 0.5 else "real"
        return {"label": "deepfake" if label == "fake" else "real", "score": round(prob, 4)}
