from cog import BasePredictor, Input
from typing import Dict, List
import json
import math
from datetime import datetime, timedelta
from collections import defaultdict

# Technique metadata from PDF (ID: {name, impact snippet for logging})
TECHNIQUES = {
    1: {"name": "Going Dark", "impact": "Loss of MDA, increased enforcement difficulty, greater collision risk"},
    2: {"name": "AIS Spoofing (Self-Spoof)", "impact": "Misattribution of movements, navigational hazards, reduced trust in AIS feeds"},
    3: {"name": "AIS Hijacking (Impersonation)", "impact": "Attribution problems, legal complications, masking of illicit operations"},
    4: {"name": "Location Tampering / GNSS Spoofing", "impact": "Navigation errors, collision risk, loss of trust in GNSS-dependent systems"},
    5: {"name": "Replay / Relay (Message Re-injection)", "impact": "False situational pictures, wasted enforcement effort, contaminated archives"},
    6: {"name": "Availability Disruption (Flooding / Jamming / DoS)", "impact": "Regional loss of MDA, increased maritime safety risk, concealment opportunities"},
    7: {"name": "Zombie Vessels", "impact": "Hinders vetting, enforcement, and provenance checks; complicates legal accountability"},
    8: {"name": "AIS Handshakes / Identity Swap", "impact": "Obscures enforcement trails, confuses historical logs, complicates seizure"},
    9: {"name": "Hybrid & Coordinated Attacks", "impact": "High-severity strategic deception with amplified operational and geopolitical consequences"},
    10: {"name": "Virtual AtoN (V-AtoN) Spoofing", "impact": "Safety hazards (groundings, collisions), disruption of routing, malicious entrapment"},
    11: {"name": "Aggregator / Provider Injection", "impact": "Widespread misinformation, reputational damage, misdirected operational responses"},
    12: {"name": "Antenna / Transmitter Cloning & Base-Station Emulation", "impact": "Misleading DF efforts, erroneous localization, greater difficulty proving origin"},
    13: {"name": "Firmware / Supply-Chain Compromise", "impact": "Deep, hard-to-detect persistence that undermines device trust"},
    14: {"name": "Satellite-AIS (S-AIS) Specific Attacks", "impact": "Massive dissemination of false data, eroding confidence in satellite-derived AIS"},
    15: {"name": "Frequency-Based Tactics (Hopping / Multi-Channel Flooding)", "impact": "Wider blind areas, more complex attribution, difficulty maintaining coverage"},
    16: {"name": "Radar / Sensor Corruption (Cross-Sensor Deception)", "impact": "Weakens sensor fusion, increases false positives/negatives, complicates resolution"},
    17: {"name": "Social-Engineering / Operator-Level Deception", "impact": "Scales deception via trusted channels, undermines procedural controls"},
    18: {"name": "Replay / Store-and-Forward Variants (Surgical Re-Injection)", "impact": "Tactical confusion, wasted responses, contamination of decision-making"},
    19: {"name": "Coordinated Disinformation / Political Use", "impact": "Diplomatic tension, misinformed policy decisions, potential unintended escalation"},
    20: {"name": "Distress & SAR Message Spoofing", "impact": "Dangerous diversion of rescue assets, potential ambush scenarios, erosion of trust"},
    21: {"name": "Binary / Safety Message & Area-Notice Manipulation (Msg 8/12/14)", "impact": "Operational disruption, navigational hazards, potential economic/safety consequences"},
    22: {"name": "Channel-Management Abuse (Msg 22 / DSC)", "impact": "Fragmented MDA, selective monitoring blind spots, opportunity for focused illicit activity"},
    23: {"name": "Covert Messaging", "impact": "Enables clandestine coordination while preserving plausible deniability; degrades AIS confidence"}
}

class Predictor(BasePredictor):
    def setup(self):
        self.last_seen = defaultdict(lambda: None)  # Track per-MMSI timestamps for statefulness
        self.known_hashes = defaultdict(set)  # For replay detection

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        if not (lat1 and lon1 and lat2 and lon2):
            return 0
        R = 6371  # km
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def predict(self, ais_json: str = Input(description="Batched AIS JSON as string (list of dicts with MMSI, Latitude, Longitude, SpeedOverGround, Timestamp)")) -> Dict:
        try:
            data = json.loads(ais_json)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            return {"score": 0.0, "techniques": [], "impact": "Invalid JSON input"}

        flagged_techniques = set()
        total_score = 0.0
        mmsis = [msg.get('MMSI') for msg in data if msg.get('MMSI')]

        for msg in data:
            mmsi = msg.get('MMSI')
            if not mmsi:
                continue
            try:
                ts_str = msg.get('Timestamp', datetime.now().isoformat())
                timestamp = datetime.fromisoformat(ts_str.replace('Z', '+00:00')) if 'Z' in ts_str else datetime.fromisoformat(ts_str)
            except:
                timestamp = datetime.now()

            # Rule-based detections from PDF manifests
            # 1: Going Dark (track disappears >30min)
            if self.last_seen[mmsi]:
                gap = timestamp - self.last_seen[mmsi]
                if gap > timedelta(minutes=30):
                    flagged_techniques.add(1)
                    total_score += 0.8
            self.last_seen[mmsi] = timestamp

            # 2: AIS Spoofing (teleportation/impossible kinematics >10km jump)
            if len(data) > 1:
                for prev_msg in data[:-1]:
                    if prev_msg.get('MMSI') == mmsi:
                        dist = self.haversine_distance(msg.get('Latitude'), msg.get('Longitude'),
                                                       prev_msg.get('Latitude'), prev_msg.get('Longitude'))
                        if dist > 10:  # km
                            flagged_techniques.add(2)
                            total_score += 0.7
                        break

            # 3: Hijacking (same MMSI in distant locations/duplicates)
            if mmsis.count(mmsi) > 1:
                flagged_techniques.add(3)
                total_score += 0.6

            # 4: GNSS Spoofing (coordinated jumps/impossible speeds >50 knots)
            sog = msg.get('SpeedOverGround', 0)
            if sog > 50:
                flagged_techniques.add(4)
                total_score += 0.7

            # 5: Replay (bit-identical payloads reappear)
            msg_str = json.dumps(msg, sort_keys=True)
            msg_hash = hash(msg_str)
            if msg_hash in self.known_hashes[mmsi]:
                flagged_techniques.add(5)
                total_score += 0.9
            self.known_hashes[mmsi].add(msg_hash)

            # 6: Flooding (high message volume/malformed >50 in batch)
            if len(data) > 50:
                flagged_techniques.add(6)
                total_score += 0.5

            # 7: Zombie (MMSI of scrapped vessels; stub: invalid range <100 or >999999999)
            if not (100 <= mmsi <= 999999999):
                flagged_techniques.add(7)
                total_score += 0.8

            # 8: Identity Swap (near-simultaneous static changes; stub: rapid MMSI/ID changes in batch)
            if any('Name' in msg and msg['Name'] != data[0].get('Name', '') for msg in data[1:]):
                flagged_techniques.add(8)
                total_score += 0.7

            # 9: Coordinated (multiple vessels similar anomalies; stub: >3 vessels same lat/lon offset)
            lats = [m.get('Latitude') for m in data]
            if max(lats) - min(lats) < 0.01 and len(set(lats)) < len(data)/3:
                flagged_techniques.add(9)
                total_score += 0.9

            # 10: V-AtoN (new AtoNs without physical; stub: detect AtoN message type if present)
            if msg.get('MessageType') in [21, 123]:  # AIS AtoN types
                flagged_techniques.add(10)
                total_score += 0.6

            # 11: Aggregator Injection (inconsistent with local; stub: flag if no 'source' or anomalous timestamp)
            if 'source' not in msg and abs((timestamp - datetime.now()).days) > 1:
                flagged_techniques.add(11)
                total_score += 0.7

            # 12: Cloning (RF anomalies; stub: duplicate full msg)
            if data.count(msg) > 1:
                flagged_techniques.add(12)
                total_score += 0.8

            # 13: Firmware Compromise (unexpected static edits; stub: varying name/MMSI in sequence)
            names = [m.get('Name', '') for m in data if m.get('MMSI') == mmsi]
            if len(set(names)) > 1:
                flagged_techniques.add(13)
                total_score += 0.75

            # 14: S-AIS Attacks (ghosts during passes; stub: high volume in short time)
            if len(data) > 20 and (timestamp - data[0]['Timestamp']) < timedelta(minutes=5):
                flagged_techniques.add(14)
                total_score += 0.8

            # 15: Frequency Tactics (channel shifts; stub: varying 'Channel' if present)
            channels = [m.get('Channel', 87) for m in data]
            if len(set(channels)) > 1:
                flagged_techniques.add(15)
                total_score += 0.6

            # 16: Sensor Corruption (mismatched kinematics; stub: speed=0 but position changes)
            if sog == 0 and dist > 1:  # From earlier jump calc
                flagged_techniques.add(16)
                total_score += 0.7

            # 17: Social-Engineering (manual edits; stub: unusual access patterns - flag if 'UserEdited' key)
            if msg.get('UserEdited', False):
                flagged_techniques.add(17)
                total_score += 0.5

            # 18: Surgical Replay (exact short sequences; stub: substring match in batch)
            for prev in data[:-1]:
                if json.dumps(prev)[:50] == json.dumps(msg)[:50]:
                    flagged_techniques.add(18)
                    total_score += 0.85

            # 19: Disinformation (theater anomalies timed to events; stub: high flags in batch)
            if len(flagged_techniques) > 5:
                flagged_techniques.add(19)
                total_score += 0.9

            # 20: Distress Spoofing (improbable distress msgs; stub: if 'Distress' key without lat)
            if msg.get('MessageType') == 14 and not msg.get('Latitude'):  # Safety msg
                flagged_techniques.add(20)
                total_score += 0.8

            # 21: Binary Manipulation (unusual binary payloads; stub: if binary data present)
            if 'BinaryData' in msg:
                flagged_techniques.add(21)
                total_score += 0.6

            # 22: Channel Abuse (switch instructions; stub: Msg 22 type)
            if msg.get('MessageType') == 22:
                flagged_techniques.add(22)
                total_score += 0.7

            # 23: Covert Messaging (non-natural patterns; stub: structured timing variance <1s)
            times = [datetime.fromisoformat(m.get('Timestamp', ts_str)) for m in data]
            if times and all((times[i+1] - times[i]).total_seconds() < 1 for i in range(len(times)-1)):
                flagged_techniques.add(23)
                total_score += 0.75

        # Aggregate
        unique_techs = list(flagged_techniques)
        score = min(total_score / max(len(data), 1), 1.0)  # Normalize
        impacts = [TECHNIQUES[t]["impact"] for t in unique_techs]
        impact = "; ".join(impacts) if impacts else "No anomalies detected"

        return {
            "score": float(score),
            "techniques": unique_techs,
            "impact": impact
        }
