from cog import BasePredictor, Input
import torch
import torch.nn as nn
import json
import numpy as np

class AISAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8))  # lat, long, SOG
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Predictor(BasePredictor):
    def setup(self):
        """Load model on startup"""
        self.model = AISAutoencoder()
        self.model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        self.model.eval()

    @torch.inference_mode()
    def predict(self, ais_json: str = Input(description="JSON string of AIS data")) -> str:
        """Score for anomalies; return JSON with techniques/impacts"""
        input_data = json.loads(ais_json)
        positions = np.array(input_data['positions'])  # e.g., [[lat1, long1, sog1], ...]
        inputs = torch.tensor(positions, dtype=torch.float32)
        outputs = self.model(inputs)
        mse = torch.mean((inputs - outputs) ** 2).item()
        score = min(1.0, mse / 0.1)  # 0-1 anomaly score
        techniques = [2, 4] if score > 0.7 else []  # e.g., Self-Spoof, GNSS Spoofing
        return json.dumps({
            "score": score,
            "techniques": techniques,
            "impact": "Navigation hazard per PDF" if techniques else "Normal"
        })
