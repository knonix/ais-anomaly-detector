from cog import BasePredictor, Input
import torch
import torch.nn as nn
import json
import numpy as np
from sklearn.ensemble import IsolationForest  # Proxy for XGBoost; use import xgboost as xgb for full
import pickle  # For loading .pkl
from math import radians, sin, cos, sqrt, atan2

class AISAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8))  # Features: lat, long, SOG
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Predictor(BasePredictor):
    def setup(self):
        """Load models on startup"""
        self.ae_model = AISAutoencoder()
        # self.ae_model.load_state_dict(torch.load('model.pth', map_location='cpu'))  # Uncomment if using
        self.ae_model.eval()
        
        # Load XGBoost/IsolationForest for kinematics (train on deltas/speeds)
        with open('xgboost_model.pkl', 'rb') as f:
            self.kinematics_model = pickle.load(f)  # Assumes IsolationForest or XGBClassifier saved

    @torch.inference_mode()
    def predict(self, ais_json: str = Input(description="JSON string of AIS data")) -> str:
        """Score anomalies across techniques"""
        input_data = json.loads(ais_json)
        positions = np.array(input_data['positions'])  # [[lat, long, sog, ...]]; extend features as needed
        mmsi = input_data.get('mmsi', None)
        timestamps = input_data.get('timestamps', [])  # Optional for gaps/timing

        # 1. Autoencoder for general track reconstruction
        if len(positions) > 0:
            inputs = torch.tensor(positions[:, :3], dtype=torch.float32)  # Assume first 3 cols: lat/long/sog
            outputs = self.ae_model(inputs)
            mse = torch.mean((inputs - outputs) ** 2).item()
            ae_score = min(1.0, mse / 0.1)  # Normalize 0-1
        else:
            ae_score = 0.0

        # 2. XGBoost for kinematics (e.g., jumps, speeds)
        kinem_score = 0.0
        if len(positions) > 1:
            lats, lons, sogs = positions[:, 0], positions[:, 1], positions[:, 2]
            # Haversine distance deltas for jumps (>10km = Technique 2 flag)
            def haversine(lat1, lon1, lat2, lon2):
                R = 6371  # km
                dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                c = 2 * atan2(sqrt(a), sqrt(1 - a))
                return R * c
            dist_deltas = np.array([haversine(lats[i], lons[i], lats[i+1], lons[i+1]) for i in range(len(lats)-1)])
            max_delta = np.max(dist_deltas)
            speed_deltas = np.diff(sogs)  # Abrupt changes
            
            # Features for model: [mean_dist_delta, max_speed, std_sog]
            features = np.array([[np.mean(dist_deltas), np.max(sogs), np.std(sogs)]])
            
            # Predict with loaded model (anomaly score; -1 outlier for IsolationForest)
            kinem_pred = self.kinematics_model.predict(features)[0]
            kinem_score = 1.0 if kinem_pred == -1 or max_delta > 10 else 0.0  # >0.8 threshold logic

        overall_score = max(ae_score, kinem_score)

        # 3. Extend for all 23 techniques: Rule/ML checks (expand with PDF manifests)
        techniques = []
        if overall_score > 0.7:
            techniques.extend([2, 4])  # Self-Spoof, GNSS Spoofing (jumps/drifts)
        if kinem_score > 0.8:
            techniques.append(1)  # Going Dark (add gap check on timestamps)
        
        # Modular checks dict (1-23; add ML/rules per PDF, e.g., hash for replays)
        tech_checks = {
            3: lambda d: 'mmsi' in d and len(str(d.get('mmsi', ''))) < 9,  # Dummy: Short MMSI (hijacking duplicate)
            5: lambda d: len(set(tuple(map(tuple, np.array(d['positions'])))) < len(d['positions']),  # Replay: Duplicate positions
            6: lambda d: np.std([p[2] for p in d['positions']]) > 50,  # Disruption: High SOG variance (flooding noise)
            7: lambda d: 'mmsi' in d and str(d['mmsi']).startswith('000'),  # Zombie: Scrapped-like ID
            8: lambda d: len(set([str(p) for p in d.get('static_fields', [])])) < len(d['static_fields']),  # Swap: Duplicate statics
            9: lambda d: len(d['techniques']) > 2 if 'techniques' in d else False,  # Hybrid: Multiple flags (recursive)
            10: lambda d: any('aton' in str(p) for p in d['positions']),  # V-AtoN: Dummy AtoN keywords
            11: lambda d: np.mean(dist_deltas) < 0.1 if 'dist_deltas' in locals() else False,  # Aggregator: Static tracks
            12: lambda d: np.var(sogs) < 0.01,  # Cloning: Low variance (mimic)
            13: lambda d: len(timestamps) > 0 and np.mean(np.diff(timestamps)) == 0,  # Firmware: Uniform timing
            14: lambda d: len(positions) % 2 == 0 and np.mean(lats) > 50,  # S-AIS: Even count, high lat (satellite pass)
            15: lambda d: len(set(np.round(sogs, 1))) < 3,  # Frequency hopping: Few discrete speeds
            16: lambda d: any(abs(sog) > 100 for sog in sogs),  # Sensor corruption: Extreme values
            17: lambda d: 'manual_edit' in d,  # Social-eng: Flag in input
            18: lambda d: any(np.allclose(positions[i], positions[i+1]) for i in range(len(positions)-1)),  # Surgical replay: Identical seq
            19: lambda d: len(positions) > 20 and np.std(lats) < 1,  # Disinfo: Theater-wide static
            20: lambda d: any(sog < 0 for sog in sogs),  # Distress: Negative speeds (improbable)
            21: lambda d: 'binary_payload' in d and len(d['binary_payload']) < 10,  # Binary abuse: Short payloads
            22: lambda d: np.mean(np.diff(timestamps)) > 3600 if timestamps else False,  # Channel abuse: Long intervals
            23: lambda d: np.std(np.diff([p[2] for p in d['positions']])) < 0.01 if len(d['positions']) > 1 else False  # Covert: Non-natural low variation
        }
        for tid, check in tech_checks.items():
            if check(input_data):
                techniques.append(tid)

        impact = "High risk: Verify PDF impacts (e.g., MDA loss for Technique 1)" if techniques else "Normal operation"

        return json.dumps({
            "score": overall_score,
            "techniques": sorted(set(techniques)),  # Unique IDs
            "impact": impact
        })
