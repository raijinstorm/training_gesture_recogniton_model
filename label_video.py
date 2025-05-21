#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app2.py  —  Gesture recognition on a pre-recorded MP4 (or any video file)

• Loads a pre-trained SVM saved with joblib (same as app.py).
• Reads frames from --input, overlays the predicted gesture label,
  and shows them live.
• If --output is supplied, writes the annotated video to that file.
"""

import argparse
import csv
import copy
from pathlib import Path
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import joblib


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gesture recognition on a video file")
    parser.add_argument("--input", required=True, help="Path to an input video (e.g. .mp4)")
    parser.add_argument("--output", default="", help="Optional path to save annotated video")
    parser.add_argument(
        "--model_path",
        default="model/keypoint_classifier/svm_model.joblib",
        help="Path to SVM joblib",
    )
    parser.add_argument(
        "--label_path",
        default="model/keypoint_classifier/keypoint_classifier_label.csv",
        help="Path to label CSV (one label per line)",
    )
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--mirror", action="store_true", help="Mirror each frame (like webcam)")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Classifier wrapper
# --------------------------------------------------------------------------- #
class KeyPointClassifier:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def __call__(self, features: list[float]) -> int:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        return int(self.model.predict(x)[0])


# --------------------------------------------------------------------------- #
# Geometry + preprocessing
# --------------------------------------------------------------------------- #
def calc_brect(image, landmarks):
    h, w = image.shape[:2]
    pts = np.asarray([(int(l.x * w), int(l.y * h)) for l in landmarks.landmark], np.int32)
    x, y, bw, bh = cv.boundingRect(pts)
    return [x, y, x + bw, y + bh]


def calc_landmarks(image, landmarks):
    h, w = image.shape[:2]
    return [[int(l.x * w), int(l.y * h)] for l in landmarks.landmark]


def pre_process_landmarks(lm_list):
    # translate so wrist is origin, flatten, normalise
    base_x, base_y = lm_list[0]
    rel = [[x - base_x, y - base_y] for x, y in lm_list]
    flat = np.asarray(rel, np.float32).flatten()
    max_val = np.max(np.abs(flat))
    return list(flat / max_val) if max_val else list(flat)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def draw_landmarks(image, points):
    for x, y in points:
        cv.circle(image, (x, y), 3, (255, 255, 255), -1)
        cv.circle(image, (x, y), 3, (0, 0, 0), 1)
    return image


def draw_brect(image, brect):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
    return image


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = get_args()

    # video in
    cap = cv.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    src_w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    # optional video out
    if args.output:
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        writer = cv.VideoWriter(args.output, fourcc, src_fps, (src_w, src_h))
        if not writer.isOpened():
            raise IOError("Could not open VideoWriter")
    else:
        writer = None

    # mediapipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # model + labels
    classifier = KeyPointClassifier(args.model_path)
    with open(args.label_path, encoding="utf-8-sig") as f:
        labels = [row[0] for row in csv.reader(f)]

    # FPS display
    ticks = deque(maxlen=10)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.mirror:
            frame = cv.flip(frame, 1)  # optional mirror to mimic webcam

        debug = copy.deepcopy(frame)

        # inference
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
                brect = calc_brect(debug, lm)
                lm_list = calc_landmarks(debug, lm)
                feat = pre_process_landmarks(lm_list)
                idx = classifier(feat)
                label = labels[idx] if idx < len(labels) else str(idx)

                draw_brect(debug, brect)
                draw_landmarks(debug, lm_list)
                cv.rectangle(debug, (brect[0], brect[1] - 25), (brect[2], brect[1]), (0, 0, 0), -1)
                cv.putText(
                    debug,
                    f"{handed.classification[0].label}: {label}",
                    (brect[0] + 5, brect[1] - 7),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv.LINE_AA,
                )

        # FPS
        ticks.append(cv.getTickCount())
        if len(ticks) >= 2:
            fps = cv.getTickFrequency() / (ticks[-1] - ticks[-2])
            cv.putText(
                debug,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                3,
                cv.LINE_AA,
            )
            cv.putText(
                debug,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                1,
                cv.LINE_AA,
            )

        # show + save
        cv.imshow("Hand Gesture Recognition (video)", debug)
        if writer:
            writer.write(debug)

        if cv.waitKey(1) & 0xFF == 27:  # ESC
            break

    # tidy up
    cap.release()
    if writer:
        writer.release()
    cv.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
