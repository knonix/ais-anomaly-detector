# predict.py
from cog import BasePredictor, Input
import json
import numpy as np
import pandas as pd
from datetime import datetime
import hashlib
import pickle
from scipy.stats import entropy
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict

# Embedded PDF Impacts (unchanged)
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
        default_factory = lambda: {'msg_rate': 2.0, 'sog_mean': 10.0}
        try:
            with open('baselines.pkl', 'rb') as f:
                loaded = pickle.load(f)
                self.baselines = defaultdict(default_factory, loaded)
        except FileNotFoundError:
            self.baselines = defaultdict(default_factory)

    def predict(self, ais_json: str = Input(description="AIS JSON batch with extended fields (e.g., static, msg_type)")) -> str:
        try:
            data = json.loads(ais_json)
            if isinstance(data, list):
                rows = []
                for item in data:
                    row = {
                        'mmsi': item.get('mmsi', 0),
                        'lat': item.get('latitude', item.get('lat', 0.0)),
                        'long': item.get('longitude', item.get('long', 0.0)),
                        'sog': item.get('speedOverGround', item.get('sog', 0.0)),
                        'cog': item.get('courseOverGround', item.get('cog', 0.0)),
                        'timestamp': datetime.fromisoformat(item.get('timestamp', '2025-11-14T00:00:00Z').replace('Z', '+00:00')),
                        'vessel_name': item.get('vessel_name', ''),
                        'imo': item.get('imo', 0),
                        'callsign': item.get('callsign', ''),
                        'vessel_type': item.get('vessel_type', 0),
                        'msg_type': item.get('message_type', 1),  # Assume added
                        'heading': item.get('heading', 0.0),
                        'signal_strength': item.get('signal_strength', -70.0),  # dBm, assume added
                        'raim_flag': item.get('raim_flag', False),
                        'payload_raw': str(item)  # For hashing
                    }
                    rows.append(row)
                df = pd.DataFrame(rows)
            else:
                # Batch handling (similar, extended)
                mmsi = data.get('mmsi', 0)
                raw_positions = data.get('positions', [])
                positions = [p[:4] + [0.0] * (4 - len(p)) for p in raw_positions]  # Pad to lat,long,sog,cog
                timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in data.get('timestamps', [str(datetime.now())] * len(positions))]
                df = pd.DataFrame(positions, columns=['lat', 'long', 'sog', 'cog'])
                df['timestamp'] = timestamps
                df['mmsi'] = mmsi
                df['vessel_name'] = data.get('vessel_name', '')
                df['imo'] = data.get('imo', 0)
                df['callsign'] = data.get('callsign', '')
                df['vessel_type'] = data.get('vessel_type', 0)
                df['msg_type'] = 1
                df['heading'] = 0.0
                df['signal_strength'] = -70.0
                df['raim_flag'] = False
                df['payload_raw'] = df.apply(str, axis=1)

            if len(df) < 2:
                with open('baselines.pkl', 'wb') as f:
                    pickle.dump(dict(self.baselines), f)
                return json.dumps({'score': 0.0, 'techniques': [], 'impacts': [], 'details': {f'technique_{i}': 0.0 for i in range(1,24)}, 'explanations': {}})

            # Group by MMSI for multi-vessel analysis
            grouped = df.groupby('mmsi')
            overall_flags = set()
            details = {f'technique_{i}': 0.0 for i in range(1, 24)}
            explanations = {}

            for mmsi, group in grouped:
                gdf = group.sort_values('timestamp').reset_index(drop=True)
                gdf['time_delta_min'] = gdf['timestamp'].diff().dt.total_seconds().fillna(0) / 60
                gdf['lat_delta'] = gdf['lat'].diff().fillna(0)
                gdf['long_delta'] = gdf['long'].diff().fillna(0)
                gdf['sog_delta'] = gdf['sog'].diff().fillna(0)
                gdf['cog_delta'] = gdf['cog'].diff().fillna(0)
                gdf['dist_delta_km'] = np.sqrt(gdf['lat_delta']**2 + gdf['long_delta']**2) * 111
                gdf['msg_rate'] = 1 / gdf['time_delta_min'].replace(0, 1)
                gdf['payload_hash'] = [hashlib.md5(str(payload).encode()).hexdigest() for payload in gdf['payload_raw']]
                gdf['static_tuple'] = gdf[['vessel_name', 'imo', 'callsign']].apply(tuple, axis=1)
                gdf['hour'] = gdf['timestamp'].dt.hour
                baseline = self.baselines[mmsi]

                # Update baselines with EMA after processing (for future calls)
                if len(gdf) > 0:
                    new_msg_rate = gdf['msg_rate'].mean()
                    new_sog_mean = gdf['sog'].mean()
                    alpha = 0.1  # Learning rate
                    self.baselines[mmsi]['msg_rate'] = (1 - alpha) * baseline['msg_rate'] + alpha * new_msg_rate
                    self.baselines[mmsi]['sog_mean'] = (1 - alpha) * baseline['sog_mean'] + alpha * new_sog_mean

                # Enhanced Features
                kinematic_features = ['lat_delta', 'long_delta', 'sog_delta', 'cog_delta', 'dist_delta_km']
                X_kin = gdf[kinematic_features].fillna(0).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_kin)
                iso = IsolationForest(contamination=0.1, random_state=42)
                iso.fit(X_scaled)
                kin_anomalies = -iso.decision_function(X_scaled).mean()

                # Clustering for coordinated patterns
                db = DBSCAN(eps=0.5, min_samples=3).fit(X_scaled)
                n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                cluster_score = n_clusters / len(gdf) if len(gdf) > 0 else 0

                # Entropy for covert/timing patterns
                timing_entropy = entropy(gdf['time_delta_min'].value_counts(normalize=True).values) if len(gdf) > 1 else 0
                static_entropy = entropy(gdf['static_tuple'].value_counts(normalize=True).values) if len(gdf) > 1 else 0

                # Rule-Based Detection for All 23 (PDF-Aligned, with Scores)
                flags = []

                # 1) Going Dark: Long gaps
                dark_score = min(1.0, gdf['time_delta_min'].max() / 30)
                if dark_score > 0.5: flags.append(1); details['technique_1'] = dark_score; explanations[1] = f"Max gap: {gdf['time_delta_min'].max():.1f} min"

                # 2) Self-Spoof: Impossible kinematics
                spoof_score = 1.0 if gdf['dist_delta_km'].max() > 10 or gdf['sog'].max() > 50 else kin_anomalies
                if spoof_score > 0.5: flags.append(2); details['technique_2'] = spoof_score; explanations[2] = f"Max dist: {gdf['dist_delta_km'].max():.1f}km, Max SOG: {gdf['sog'].max():.1f}kn"

                # 3) Hijacking: Dupe MMSI distant
                hijack_score = 1.0 if len(grouped.get_group(mmsi)) > 1 and gdf['dist_delta_km'].max() > 10 else 0
                if hijack_score > 0.5: flags.append(3); details['technique_3'] = hijack_score; explanations[3] = "Dupe MMSI across distances"

                # 4) GNSS Spoof: Synced drifts/low variance + clusters
                gnss_score = (1 - gdf['dist_delta_km'].std()) * cluster_score if len(gdf) > 5 else 0
                if gnss_score > 0.5: flags.append(4); details['technique_4'] = gnss_score; explanations[4] = f"Low std dev: {gdf['dist_delta_km'].std():.3f}, Clusters: {n_clusters}"

                # 5) Replay: Dupe payloads
                replay_score = gdf['payload_hash'].duplicated().sum() / len(gdf)
                if replay_score > 0.5: flags.append(5); details['technique_5'] = replay_score; explanations[5] = f"Dupe payloads: {gdf['payload_hash'].duplicated().sum()}"

                # 6) Flooding: High msg rate + low signal
                flood_score = min(1.0, gdf['msg_rate'].mean() / 5) * (1 if gdf['signal_strength'].mean() < -90 else 0.5)
                if flood_score > 0.5: flags.append(6); details['technique_6'] = flood_score; explanations[6] = f"Mean rate: {gdf['msg_rate'].mean():.1f}/min, Signal: {gdf['signal_strength'].mean():.1f}dBm"

                # 7) Zombie: Mock registry check (assume 'scrapped' flag in data; else 0)
                zombie_score = 1.0 if gdf['imo'].iloc[0] in [12345, 67890] else 0  # Mock scrapped IMOs; integrate CSV
                if zombie_score > 0.5: flags.append(7); details['technique_7'] = zombie_score; explanations[7] = "IMO matches scrapped registry"

                # 8) Handshakes: Static swaps
                swap_score = gdf['static_tuple'].nunique() / len(gdf) if len(gdf) > 2 else 0
                if swap_score > 0.5: flags.append(8); details['technique_8'] = swap_score; explanations[8] = f"Static changes: {gdf['static_tuple'].nunique()}"

                # 9) Hybrid: Multi-flag combo
                hybrid_score = len([f for f in flags if f in [1,2,4,8]]) / 4
                if hybrid_score > 0.5: flags.append(9); details['technique_9'] = hybrid_score; explanations[9] = "Layered anomalies detected"

                # 10) V-AtoN: Msg type 21 + new statics
                aton_score = 1.0 if gdf['msg_type'].eq(21).any() and gdf['vessel_type'].eq(0).any() else 0  # Type 0 = AtoN
                if aton_score > 0.5: flags.append(10); details['technique_10'] = aton_score; explanations[10] = "Unexpected Msg 21 AtoN"

                # 11) Aggregator Injection: RAIM false + low entropy
                inj_score = (1 if not gdf['raim_flag'].all() else 0) * (1 - static_entropy)
                if inj_score > 0.5: flags.append(11); details['technique_11'] = inj_score; explanations[11] = f"RAIM fails, Low entropy: {static_entropy:.3f}"

                # 12) Cloning: Signal variance low (mimic legit)
                clone_score = 1 - gdf['signal_strength'].std() if len(gdf) > 1 else 0
                if clone_score > 0.5: flags.append(12); details['technique_12'] = clone_score; explanations[12] = f"Stable signal std: {gdf['signal_strength'].std():.1f}"

                # 13) Firmware: Systematic changes (high delta variance)
                fw_score = gdf['sog_delta'].var() / baseline['sog_mean'] if baseline['sog_mean'] > 0 else 0
                if fw_score > 2: flags.append(13); details['technique_13'] = min(1.0, fw_score / 2); explanations[13] = f"Var > baseline: {fw_score:.2f}"

                # 14) S-AIS: Assume 'source' field; high global spread
                sais_score = 1.0 if gdf['dist_delta_km'].sum() > 1000 else 0  # Mock wide-area
                if sais_score > 0.5: flags.append(14); details['technique_14'] = sais_score; explanations[14] = "Wide-area ghosts"

                # 15) Freq Hop: Assume 'channel' field; shifts
                hop_score = gdf['msg_type'].diff().abs().mean() if 'channel' in gdf else 0  # Proxy
                if hop_score > 1: flags.append(15); details['technique_15'] = min(1.0, hop_score); explanations[15] = f"Channel shifts: {hop_score:.1f}"

                # 16) Sensor Corruption: Assume 'radar_match' =0; mismatch
                sensor_score = 1.0 if 'radar_match' in gdf.columns and (gdf['radar_match'] == 0).any() else 0
                if sensor_score > 0.5: flags.append(16); details['technique_16'] = sensor_score; explanations[16] = "Radar mismatches"

                # 17) Social Eng: Manual edit proxy (sudden static change) – FIXED: Use .ne(shift()) instead of .diff()
                soc_score = 1.0 if gdf['static_tuple'].ne(gdf['static_tuple'].shift()).any() else 0
                if soc_score > 0.5: flags.append(17); details['technique_17'] = soc_score; explanations[17] = "Unexplained static edits"

                # 18) Surgical Replay: Short dupe sequences
                surg_score = sum(gdf['payload_hash'].rolling(3).apply(lambda x: len(set(x)) < 3).fillna(0)) / len(gdf)
                if surg_score > 0.5: flags.append(18); details['technique_18'] = surg_score; explanations[18] = f"Short dupes: {surg_score:.2f}"

                # 19) Coordinated Disinfo: Timed with hour (mock events)
                dis_score = 1.0 if gdf['hour'].nunique() == 1 and len(flags) > 2 else 0  # Proxy for theater
                if dis_score > 0.5: flags.append(19); details['technique_19'] = dis_score; explanations[19] = "Event-timed anomalies"

                # 20) Distress Spoof: Msg type 14 + improbable pos
                dist_score = 1.0 if gdf['msg_type'].eq(14).any() and gdf['sog'].mean() == 0 else 0
                if dist_score > 0.5: flags.append(20); details['technique_20'] = dist_score; explanations[20] = "Msg 14 stationary"

                # 21) Binary/Safety: Msg 8/12/14 unusual payloads
                bin_score = gdf['msg_type'].isin([8,12,14]).sum() / len(gdf) * (1 - static_entropy)
                if bin_score > 0.5: flags.append(21); details['technique_21'] = bin_score; explanations[21] = f"Binary msgs: {bin_score:.2f}"

                # 22) Channel Abuse: Msg 22 + switch proxy (cog jumps)
                ch_score = 1.0 if gdf['msg_type'].eq(22).any() and gdf['cog_delta'].abs().max() > 180 else 0
                if ch_score > 0.5: flags.append(22); details['technique_22'] = ch_score; explanations[22] = "Msg 22 + course jumps"

                # 23) Covert: High timing entropy + low var fields
                cov_score = timing_entropy * (1 if gdf['heading'].std() < 1 else 0)  # Non-natural timing
                if cov_score > 1.5: flags.append(23); details['technique_23'] = min(1.0, cov_score / 2); explanations[23] = f"Timing entropy: {timing_entropy:.3f}"

                overall_flags.update(flags)

            # Aggregate
            flagged = list(overall_flags)
            impacts = [PDF_IMPACTS.get(t, 'Unknown') for t in flagged]
            detail_scores = np.array(list(details.values()))
            overall_score = float(np.mean(detail_scores) * kin_anomalies * (1 + cluster_score))  # Weighted hybrid

            # Persist baselines after updates
            with open('baselines.pkl', 'wb') as f:
                pickle.dump(dict(self.baselines), f)

            return json.dumps({
                'score': overall_score,
                'techniques': flagged,
                'impacts': impacts,
                'details': details,
                'explanations': explanations
            })
        except Exception as e:
            with open('baselines.pkl', 'wb') as f:
                pickle.dump(dict(self.baselines), f)
            return json.dumps({'score': 0.0, 'techniques': [], 'impacts': [], 'details': {f'technique_{i}': 0.0 for i in range(1,24)}, 'error': str(e)})
