#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple rule engine for HTP/DAP + emotions + caption.
Inputs:
  - detections: list of {label, score, bbox}
  - top_emotion: str (e.g., 'happy', 'sad', 'angry', 'natural')
  - caption: str
Output:
  - dict with 'indicators', 'risk_flags', 'summary'
"""

import re
from collections import Counter

# Basic keyword bins for caption-based hints (extend freely)
KEYWORD_BUCKETS = {
    "isolation": ["alone", "empty", "no one", "lonely", "isolated", "far away"],
    "conflict": ["angry", "fight", "storm", "dark", "broken"],
    "warmth": ["family", "together", "home", "sun", "smile", "happy"],
    "uncertainty": ["confused", "unclear", "shaky", "messy", "hesitant"],
    "control": ["orderly", "organized", "neat", "aligned"],
}

# HTP/DAP rule hints (purely heuristic, non-diagnostic)
def rulebook(dets: list, top_emotion: str, caption: str):
    labels = [d["label"] for d in dets]
    conf = [float(d.get("score", 0.0)) for d in dets]
    counts = Counter(labels)
    text = caption.lower()

    # Helper checkers
    def has(label, min_score=0.25): 
        return any(d["label"] == label and float(d.get("score",0)) >= min_score for d in dets)

    def kw_hits(bucket):
        return any(k in text for k in KEYWORD_BUCKETS.get(bucket, []))

    indicators = []
    risk_flags = []

    # --- House-related cues ---
    if has("house"):
        indicators.append("Presence of a house (interest in security/home base).")
        if has("door") and not has("window"):
            indicators.append("Door present without windows (possible guardedness/privacy).")
        if has("window") and not has("door"):
            indicators.append("Windows without door (observing without approaching).")
        if has("chimney"):
            indicators.append("Chimney (warmth/comfort themes).")
        if has("fence"):
            indicators.append("Fence (boundaries/need for protection).")
        if has("path"):
            indicators.append("Path to house (approachability/connection).")

    # --- Tree cues ---
    if has("tree"):
        indicators.append("Tree present (growth, vitality themes).")
        if has("roots"):
            indicators.append("Roots visible (connection to origins/grounding).")
        if has("fruit"):
            indicators.append("Fruit (productivity/reward).")
        if has("branches") and not has("leaves"):
            indicators.append("Branches without leaves (possible depletion/fatigue).")

    # --- Person cues ---
    if has("person"):
        indicators.append("Human figure present (self/relationships focus).")
        # body parts: missing or present
        body_parts = ["head","eyes","mouth","nose","ears","neck","arms","hands","fingers","legs","feet","shoulders"]
        missing = [bp for bp in body_parts if not has(bp)]
        if len(missing) >= 6:
            indicators.append("Many body details omitted (possible avoidance/anxiety).")
        if has("hands") and not has("fingers"):
            indicators.append("Hands present but fingers omitted (fine-control/agency not emphasized).")
        if has("mouth") and not has("nose"):
            indicators.append("Mouth emphasized over nose (communication/affect over sensing).")

    # --- Emotion cue ---
    if top_emotion in ["sad","angry"]:
        risk_flags.append(f"Dominant negative affect: {top_emotion}.")
    elif top_emotion == "happy":
        indicators.append("Positive affect detected (happy).")
    else:
        indicators.append("Affect appears neutral/natural.")

    # --- Caption keyword buckets ---
    if kw_hits("isolation"):  risk_flags.append("Caption hints at isolation.")
    if kw_hits("conflict"):   risk_flags.append("Caption hints at conflict/tension.")
    if kw_hits("warmth"):     indicators.append("Caption hints at warmth/bonding.")
    if kw_hits("uncertainty"):risk_flags.append("Caption hints at uncertainty/hesitation.")
    if kw_hits("control"):    indicators.append("Caption hints at order/control.")

    # Minimal prioritization
    headline = []
    if risk_flags:
        headline.append("Risk cues: " + "; ".join(sorted(set(risk_flags))))
    if indicators:
        headline.append("Indicators: " + "; ".join(sorted(set(indicators))))

    summary = " | ".join(headline) if headline else "No salient cues detected."
    return {
        "indicators": sorted(set(indicators)),
        "risk_flags": sorted(set(risk_flags)),
        "summary": summary
    }

def interpret(detections, top_emotion, caption):
    return rulebook(detections or [], (top_emotion or "natural").lower(), caption or "")
