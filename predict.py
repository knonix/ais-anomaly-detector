from cog import BasePredictor, Input
import json
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

# PDF Impacts (embedded for output)
PDF_IMPACTS = {
    1: "Loss of MDA, increased enforcement difficulty, greater collision risk in busy waters",
    2: "Misattribution of movements, navigational hazards if others rely on false data, reduced trust in AIS feeds",
    3: "Attribution problems, legal complications, masking of illicit operations behind a legitimate identity",
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
        # No heavy loads here to avoid startup failure; lazy in predict()
        pass

    def predict(self, ais_json: str = Input(description="AIS JSON batch")) -> str:
        try:
            data = json.loads(ais_json)
            mmsi = data.get('mmsi', 0)
            positions = np.array(data.get('positions', []))  # [[lat, long, sog, cog], ...]
            timestamps_str = data.get('timestamps', [str(datetime.now())] * len(positions))
            timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps_str]

            if len(positions) < 2:
                return json.dumps({
                    'score': 0.0,
                    'techniques': [],
                    'impacts': [],
                    'details': {f'technique_{i}': 0.0 for i in range(1, 24)}
                })

            # Build DataFrame
            df = pd.DataFrame(positions, columns=['lat', 'long', 'sog', 'cog'])
            df['timestamp'] = timestamps
            df['mmsi'] = mmsi
            df = df.sort_values('timestamp')

            # Compute Features (PDF Manifest-Aligned)
            df['time_delta_min'] = df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
            df['lat_delta'] = df['lat'].diff().fillna(0)
            df['long_delta'] = df['long'].diff().fillna(0)
            df['sog_delta'] = df['sog'].diff().fillna(0)
            df['cog_delta'] = df['cog'].diff().fillna(0)
            df['dist_delta'] = np.sqrt(df['lat_delta']**2 + df['long_delta']**2)
            df['dist_delta_km'] = df['dist_delta'] * 111  # Approx km conversion
            df['msg_rate'] = 1 / df['time_delta_min'].replace(0, 1)
            df['payload_hash'] = df[['lat', 'long', 'sog', 'cog']].round(2).apply(tuple, axis=1).map(hash)
            df['hour'] = df['timestamp'].dt.hour
            df['sog_roll_mean'] = df['sog'].rolling(5, min_periods=1).mean().diff().fillna(0)
            df['mmsi_dup'] = df['mmsi'].duplicated()

            # General ML Scoring (Isolation Forest on kinematics)
            features_ml = ['lat_delta', 'long_delta', 'sog_delta', 'cog_delta', 'dist_delta']
            X = df[features_ml].fillna(0).values
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            isolation_model = IsolationForest(contamination=0.1)
            isolation_model.fit(X_scaled)
            anomaly_scores = -isolation_model.decision_function(X_scaled)
            overall_score = anomaly_scores.mean()

            # Technique Details (Rule-Boosted ML)
            details = {f'technique_{i}': 0.0 for i in range(1, 24)}
            flagged = []

            # Technique 1: Going Dark (gaps >30min)
            if df['time_delta_min'].max() > 30:
                prob1 = 0.9 + anomaly_scores.mean() * 0.1
                details['technique_1'] = min(prob1, 1.0)
                if prob1 > 0.5:
                    flagged.append(1)

            # Technique 2: Self-Spoof (jumps >10km or SOG >50)
            if df['dist_delta_km'].max() > 10 or df['sog'].max() > 50:
                prob2 = 0.9 + anomaly_scores.max() * 0.1
                details['technique_2'] = min(prob2, 1.0)
                if prob2 > 0.5:
                    flagged.append(2)

            # Technique 3: Hijacking (duplicates distant)
            if df['mmsi_dup'].any() and df['dist_delta_km'].max() > 10:
                prob3 = 0.8 + (df['dist_delta_km'].std() * 0.2)
                details['technique_3'] = min(prob3, 1.0)
                if prob3 > 0.5:
                    flagged.append(3)

            # Technique 4: GNSS Spoof (low std drifts)
            dist_std = df['dist_delta_km'].std()
            if dist_std < 0.1:
                prob4 = 0.9 + (1 - anomaly_scores.std()) * 0.1
                details['technique_4'] = min(prob4, 1.0)
                if prob4 > 0.5:
                    flagged.append(4)

            # Technique 5: Replay (identical hashes)
            if df['payload_hash'].duplicated().any():
                prob5 = 0.85 + (len(df[df['payload_hash'].duplicated()]) / len(df)) * 0.15
                details['technique_5'] = min(prob5, 1.0)
                if prob5 > 0.5:
                    flagged.append(5)

            # Technique 6: Flooding (high rate >2x median)
            if df['msg_rate'].mean() > df['msg_rate'].median() * 2:
                prob6 = 0.8 + anomaly_scores.mean() * 0.2
                details['technique_6'] = min(prob6, 1.0)
                if prob6 > 0.5:
                    flagged.append(6)

            # Technique 7: Zombie (high variance in physical; proxy)
            if df['sog'].std() > 20:
                prob7 = 0.7 + anomaly_scores.std() * 0.3
                details['technique_7'] = min(prob7, 1.0)
                if prob7 > 0.5:
                    flagged.append(7)

            # Technique 8: Swap (MMSI changes)
            mmsi_changes = df['mmsi'].diff().abs().sum()
            if mmsi_changes > 0:
                prob8 = 0.85 + (mmsi_changes / len(df)) * 0.15
                details['technique_8'] = min(prob8, 1.0)
                if prob8 > 0.5:
                    flagged.append(8)

            # Technique 9: Hybrid (multiple flags co-occur)
            num_flags = len([k for k, v in details.items() if v > 0.4])
            if num_flags > 2:
                prob9 = min(0.9 + (num_flags / 23) * 0.1, 1.0)
                details['technique_9'] = prob9
                if prob9 > 0.5:
                    flagged.append(9)

            # Technique 10: V-AtoN (high COG variance proxy for fake markers)
            if df['cog'].std() > 180:
                prob10 = 0.8 + anomaly_scores.max() * 0.2
                details['technique_10'] = min(prob10, 1.0)
                if prob10 > 0.5:
                    flagged.append(10)

            # Technique 11: Aggregator Injection (high position variance)
            if df['lat'].std() + df['long'].std() > 1.0:
                prob11 = 0.75 + anomaly_scores.mean() * 0.25
                details['technique_11'] = min(prob11, 1.0)
                if prob11 > 0.5:
                    flagged.append(11)

            # Technique 12: Cloning (high COG jumps)
            if df['cog_delta'].abs().max() > 180:
                prob12 = 0.8 + anomaly_scores.std() * 0.2
                details['technique_12'] = min(prob12, 1.0)
                if prob12 > 0.5:
                    flagged.append(12)

            # Technique 13: Firmware (systematic SOG shifts)
            if df['sog_roll_mean'].abs().max() > 5:
                prob13 = 0.85 + anomaly_scores.mean() * 0.15
                details['technique_13'] = min(prob13, 1.0)
                if prob13 > 0.5:
                    flagged.append(13)

            # Technique 14: S-AIS Attacks (low std scores for ghosts)
            if anomaly_scores.std() < 0.1 and len(df) > 10:
                prob14 = 0.9 + (1 - anomaly_scores.std()) * 0.1
                details['technique_14'] = min(prob14, 1.0)
                if prob14 > 0.5:
                    flagged.append(14)

            # Technique 15: Hopping (high delta std)
            if df['sog_delta'].std() > 10:
                prob15 = 0.8 + anomaly_scores.std() * 0.2
                details['technique_15'] = min(prob15, 1.0)
                if prob15 > 0.5:
                    flagged.append(15)

            # Technique 16: Sensor Corruption (high residuals)
            residuals = (df['lat_delta']**2 + df['long_delta']**2)
            if residuals.max() > 0.05:
                prob16 = 0.8 + residuals.mean() * 0.2
                details['technique_16'] = min(prob16, 1.0)
                if prob16 > 0.5:
                    flagged.append(16)

            # Technique 17: Social Engineering (high variance proxy)
            if df['sog'].var() > 100:
                prob17 = 0.7 + anomaly_scores.mean() * 0.3
                details['technique_17'] = min(prob17, 1.0)
                if prob17 > 0.5:
                    flagged.append(17)

            # Technique 18: Surgical Replay (short identical sequences)
            if df['payload_hash'].rolling(3).nunique().min() == 1:
                prob18 = 0.85 + (len(df[df['payload_hash'].duplicated()]) / len(df)) * 0.15
                details['technique_18'] = min(prob18, 1.0)
                if prob18 > 0.5:
                    flagged.append(18)

            # Technique 19: Disinfo (high hour variance)
            if df['hour'].var() > 4:
                prob19 = 0.8 + anomaly_scores.std() * 0.2
                details['technique_19'] = min(prob19, 1.0)
                if prob19 > 0.5:
                    flagged.append(19)

            # Technique 20: Distress Spoof (out-of-bounds)
            if df['lat'].min() < -90 or df['lat'].max() > 90 or df['long'].min() < -180 or df['long'].max() > 180:
                prob20 = 0.95
                details['technique_20'] = prob20
                flagged.append(20)

            # Technique 21: Binary Manipulation (sog sum mod 10 == 0 proxy)
            if df['sog'].cumsum().iloc[-1] % 10 == 0:
                prob21 = 0.8 + anomaly_scores.mean() * 0.2
                details['technique_21'] = min(prob21, 1.0)
                if prob21 > 0.5:
                    flagged.append(21)

            # Technique 22: Channel Abuse (cog jumps >180)
            if df['cog_delta'].abs().max() > 180:
                prob22 = 0.85 + anomaly_scores.max() * 0.15
                details['technique_22'] = min(prob22, 1.0)
                if prob22 > 0.5:
                    flagged.append(22)

            # Technique 23: Covert Messaging (low entropy in hours)
            timings = df['hour'].value_counts(normalize=True)
            ent = entropy(timings) if len(timings) > 0 else 2.0
            if ent < 1.0:
                prob23 = 1.0 - ent
                details['technique_23'] = prob23
                if prob23 > 0.5:
                    flagged.append(23)

            # Final Score Adjustment
            overall_score = np.mean([details[f'technique_{i}'] for i in flagged]) if flagged else anomaly_scores.mean()

            impacts = [PDF_IMPACTS.get(t, 'Unknown impact') for t in flagged]

            return json.dumps({
                'score': float(overall_score),
                'techniques': flagged,
                'impacts': impacts,
                'details': details
            })

        except Exception as e:
            return json.dumps({'score': 0.0, 'techniques': [], 'impacts': [], 'details': {f'technique_{i}': 0.0 for i in range(1, 24)}, 'error': str(e)})
