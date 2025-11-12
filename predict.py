import torch
import torch.nn as nn
import json
import numpy as np

class AISAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8))  # Features: lat, long, SOG
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AISAutoencoder()
model.load_state_dict(torch.load('model.pth', map_location='cpu'))  # Your trained weights
model.eval()

def predict(data: bytes) -> bytes:
    input_data = json.loads(data.decode('utf-8'))
    positions = np.array(input_data['positions'])  # e.g., [[lat1, long1, sog1], ...]
    with torch.no_grad():
        inputs = torch.tensor(positions, dtype=torch.float32)
        outputs = model(inputs)
        mse = torch.mean((inputs - outputs) ** 2).item()
        score = min(1.0, mse / 0.1)  # Normalize anomaly score (0-1; high = anomalous)
        techniques = [2, 4] if score > 0.7 else []  # Flag Self-Spoof/GNSS Spoofing
    return json.dumps({"score": score, "techniques": techniques, "impact": "Navigation hazard"}).encode('utf-8')
