from cog import BasePredictor, Input
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib  # For loading pre-trained model
from datetime import datetime
import hashlib

# PDF Impacts (embedded for output)
PDF_IMPACTS = {
    1: "Loss of MDA, increased enforcement difficulty, greater collision risk in busy waters",
    2: "Misattribution of movements, navigational hazards if others rely on false data, reduced trust in AIS feeds",
    3: "Attribution problems, legal complications, masking of illicit operations behind legitimate identity",
    4: "Navigation errors, collision risk, loss of trust in GNSS-dependent systems and AIS-derived locations",
    5: "False situational pictures, wasted enforcement effort, contaminated archives",
    6: "Regional loss of MDA, increased maritime safety risk, concealment opportunities for illicit acts",
    7: "Hinders vetting, enforcement, and provenance checks; complicates legal accountability",
    8: "Obscures enforcement trails, confuses historical logs, complicates seizure or sanction processes",
    9: "High-severity strategic deception with amplified operational and geopolitical consequences",
    10: "Safety hazards (groundings, collisions), disruption of established routing, potential for malicious entrapment",
    11: "Widespread misinformation, reputational damage, misdirected operational responses",
    12: "Misleading DF efforts, erroneous localization, greater difficulty proving origin in investigations",
    13: "Deep, hard-to-detect persistence that undermines device trust and facilitates sophisticated deception campaigns",
    14: "Massive dissemination of false data, affecting maritime stakeholders globally and eroding confidence in satellite-derived AIS",
    15: "Wider blind areas, more complex attribution, difficulty maintaining continuous coverage",
    16: "Weakens sensor fusion, increases false positives/negatives, complicates incident resolution",
    17: "Scales deception via trusted channels, undermines procedural controls, hard to detect without audit trails",
    18: "Tactical confusion, wasted responses, contamination of operational decision-making",
    19: "Diplomatic tension, misinformed policy decisions, potential unintended escalation",
    20: "Dangerous diversion of rescue assets, potential ambush scenarios, erosion of trust in genuine distress alerts",
    21: "Operational disruption, navigational hazards, potential economic or safety consequences",
    22: "Fragmented MDA, selective monitoring blind spots, opportunity for focused illicit activity",
    23: "Enables clandestine coordination while preserving plausible deniability; degrades confidence that AIS-only monitoring captures all communications"
}

class Predictor(BasePredictor):
    def setup(self):
        # Load pre-trained multi-label classifier (trained on labeled AIS data for 23 techniques)
        # Upload 'ais_23_model.pkl' to repo during Cog build
        self.model = joblib.load('ais_23_model.pkl')  # MultiOutput RandomForest for binary flags per technique
        self.scaler = StandardScaler()  # For feature normalization
        # Dummy pre-fit on sample (in prod, fit during training)
        self.model.fit(np.random.rand(100, 10), np.random.randint(0, 2, (100, 23)))  # Placeholder; real fit in training

    def predict(self, ais_json: str = Input(description="AIS JSON batch")) -> str:
        data = json.loads(ais_json)
        mmsi = data.get('mmsi', 0)
        positions = np.array(data['positions'])  # [[lat, long, sog, cog], ...]
        timestamps = data.get('timestamps', [str(datetime.now())] * len(positions))  # ISO strings
        timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]

        if len(positions) < 2:
            return json.dumps({'score': 0.0, 'techniques': [], 'impacts': [], 'details': {f'technique_{i}': 0.0 for i in range(1, 24)}})

        # Step 1: Compute Features Based on PDF Manifests
        df = pd.DataFrame(positions, columns=['lat', 'long', 'sog', 'cog'])
        df['timestamp'] = timestamps
        df['time_delta_min'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
        df['lat_delta'] = df['lat'].diff().fillna(0)
        df['long_delta'] = df['long'].diff().fillna(0)
        df['sog_delta'] = df['sog'].diff().fillna(0)
        df['cog_delta'] = df['cog'].diff().fillna(0)
        df['dist_delta'] = np.sqrt(df['lat_delta']**2 + df['long_delta']**2)
        df['msg_rate'] = 1 / df['time_delta_min'].replace(0, 1)
        df['payload_hash'] = df[['lat', 'long', 'sog', 'cog']].round(2).apply(tuple, axis=1).map(hash)
        df['hour'] = df['timestamp'].dt.hour
        df['sog_roll_mean'] = df['sog'].rolling(5, min_periods=1).mean().diff().fillna(0)
        df['mmsi_dup'] = df['mmsi'].duplicated() if 'mmsi' in df.columns else False  # Assume per-batch MMSI

        # Features vector for ML (10 key features covering manifests)
        features = ['time_delta_min', 'lat_delta', 'long_delta', 'sog_delta', 'cog_delta', 'dist_delta', 'msg_rate', 'sog_roll_mean', 'hour.var' if len(df) > 1 else 0, 'payload_hash.diff']
        X = df[features[:10]].fillna(0).values  # Pad to 10

        # Step 2: ML Scoring (Multi-Label Classification)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, :, 1]  # Prob per technique (23 outputs)
        overall_score = probs.mean()  # Average risk

        # Step 3: Rule-Based Overrides for PDF Manifests (Boost Accuracy)
        details = {f'technique_{i}': probs[:, i-1].mean() for i in range(1, 24)}
        flagged = []
        # Technique 1: Going Dark (gaps >30min)
        if df['time_delta_min'].max() > 30:
            details['technique_1'] = max(details['technique_1'], 0.9)
            flagged.append(1)
        # Technique 2: Self-Spoof (jumps >10km or speeds >50)
        if df['dist_delta'].max() > 0.1 or df['sog'].max() > 50:
            details['technique_2'] = max(details['technique_2'], 0.9)
            flagged.append(2)
        # Technique 3: Hijacking (duplicates distant)
        if df['mmsi_dup'].any() and df['dist_delta'].max() > 0.1:
            details['technique_3'] = max(details['technique_3'], 0.9)
            flagged.append(3)
        # Technique 4: GNSS Spoof (low std drifts)
        dist_std = df['dist_delta'].std()
        if dist_std < 0.001:
            details['technique_4'] = max(details['technique_4'], 0.9)
            flagged.append(4)
        # Technique 5: Replay (identical hashes)
        if df['payload_hash'].duplicated().any():
            details['technique_5'] = max(details['technique_5'], 0.9)
            flagged.append(5)
        # Technique 6: Flooding (high rate >2x median)
        if df['msg_rate'].mean() > df['msg_rate'].median() * 2:
            details['technique_6'] = max(details['technique_6'], 0.9)
            flagged.append(6)
        # Technique 7: Zombie (random 5% as proxy; integrate registry in prod)
        if np.random.random() < 0.05:  # Placeholder; use Equasis API
            details['technique_7'] = max(details['technique_7'], 0.9)
            flagged.append(7)
        # Technique 8: Swap (MMSI changes)
        if 'mmsi_change' in df.columns and df['mmsi_change'].any():
            details['technique_8'] = max(details['technique_8'], 0.9)
            flagged.append(8)
        # Technique 9: Hybrid (multiple flags co-occur)
        if len(flagged) > 1:
            details['technique_9'] = max(details['technique_9'], 0.9)
            flagged.append(9)
        # Technique 10: V-AtoN (unusual AtoN-like; dummy if 'aton' in data)
        if 'aton' in json.dumps(data).lower():
            details['technique_10'] = max(details['technique_10'], 0.9)
            flagged.append(10)
        # Technique 11: Aggregator Injection (high variance)
        if df['sog'].std() > 20:
            details['technique_11'] = max(details['technique_11'], 0.9)
            flagged.append(11)
        # Technique 12: Cloning (high COG variance)
        if df['cog'].std() > 180:
            details['technique_12'] = max(details['technique_12'], 0.9)
            flagged.append(12)
        # Technique 13: Firmware (systematic sog shifts)
        if df['sog_roll_mean'].abs().max() > 5:
            details['technique_13'] = max(details['technique_13'], 0.9)
            flagged.append(13)
        # Technique 14: S-AIS Attacks (low std in batch)
        if df['score'].std() < 0.1 and len(df) > 10:
            details['technique_14'] = max(details['technique_14'], 0.9)
            flagged.append(14)
        # Technique 15: Hopping (high delta std)
        if df['sog_delta'].std() > 10:
            details['technique_15'] = max(details['technique_15'], 0.9)
            flagged.append(15)
        # Technique 16: Sensor Corruption (high residuals)
        if (df['lat_delta']**2 + df['long_delta']**2).max() > 0.05:
            details['technique_16'] = max(details['technique_16'], 0.9)
            flagged.append(16)
        # Technique 17: Social Engineering (random 5%; prod: log anomalies)
        if np.random.random() < 0.05:
            details['technique_17'] = max(details['technique_17'], 0.9)
            flagged.append(17)
        # Technique 18: Surgical Replay (short identical sequences)
        if df['payload_hash'].rolling(3).nunique().min() == 1:
            details['technique_18'] = max(details['technique_18'], 0.9)
            flagged.append(18)
        # Technique 19: Disinfo (high hour variance)
        if df['hour'].var() > 4:
            details['technique_19'] = max(details['technique_19'], 0.9)
            flagged.append(19)
        # Technique 20: Distress Spoof (out-of-bounds lat/long)
        if df['lat'].min() < -90 or df['lat'].max() > 90 or df['long'].min() < -180 or df['long'].max() > 180:
            details['technique_20'] = max(details['technique_20'], 0.9)
            flagged.append(20)
        # Technique 21: Binary Manipulation (sog sum mod 10 == 0)
        if df['sog'].cumsum().iloc[-1] % 10 == 0:
            details['technique_21'] = max(details['technique_21'], 0.9)
            flagged.append(21)
        # Technique 22: Channel Abuse (cog jumps >180)
        if df['cog_delta'].abs().max() > 180:
            details['technique_22'] = max(details['technique_22'], 0.9)
            flagged.append(22)
        # Technique 23: Covert Messaging (low entropy in hours)
        timings = df['hour'].value_counts(normalize=True)
        ent = entropy(timings) if len(timings) > 0 else 2.0
        if ent < 1.0:
            details['technique_23'] = max(details['technique_23'], 0.9)
            flagged.append(23)

        # Update overall score with rules
        overall_score = max(overall_score, np.mean([details[f'technique_{i}'] for i in flagged]))

        impacts = [PDF_IMPACTS.get(t, 'Unknown impact') for t in flagged]

        return json.dumps({
            'score': float(overall_score),
            'techniques': flagged,
            'impacts': impacts,
            'details': details
        })
