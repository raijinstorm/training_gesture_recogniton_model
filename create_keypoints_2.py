#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import csv
import copy
import itertools
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

# Define the list of gestures.
# To extend this list, simply add another dictionary with keys 'emoji' and 'name'.
gestures = [
    {'emoji': '🖐', 'name': 'open_hand'},
    {'emoji': '👌', 'name': 'ok'},
    {'emoji': '🤏', 'name': 'pinch'},
    {'emoji': '✌️', 'name': 'victory'},
    {'emoji': '🤟', 'name': 'love_you'},
    {'emoji': '👍', 'name': 'thumbs_up'},
    {'emoji': '👎', 'name': 'thumbs_down'},
    {'emoji': '👈', 'name': 'left_point'},
    {'emoji': '✊', 'name': 'fist'},
    {'emoji': '🫶', 'name': 'heart_hands'},
    {'emoji': '🙏', 'name': 'pray'},
    {'emoji': '🤞', 'name': 'fingers_crossed'},
    {'emoji': '💪', 'name': 'flex'},
    {'emoji': '🤌', 'name': 'pinched_fingers'}
]

def get_label_index_from_filename(filename, gestures):
    """
    Extracts the gesture label index from the beginning of the filename.
    The comparison is case-insensitive.
    Returns the index of the gesture in the gestures list or None if no match is found.
    """
    basename = os.path.basename(filename).lower()
    for index, gesture in enumerate(gestures):
        if basename.startswith(gesture['name'].lower()):
            return index
    return None

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_list = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([landmark_x, landmark_y])
    return landmark_list

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    # Use the first landmark (e.g., the wrist) as the base coordinate.
    base_x, base_y = temp_landmark_list[0]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    # Flatten the list into one dimension.
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    # Normalize the values.
    max_value = max(list(map(abs, temp_landmark_list)))
    normalized = [n / max_value for n in temp_landmark_list]
    return normalized

def logging_csv(label_index, landmark_list):
    csv_path = 'keypoint2.csv'
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label_index, *landmark_list])

def main():
    parser = argparse.ArgumentParser(
        description="Create CSV dataset from a folder of photos using MediaPipe hands (Keypoint mode only). "
                    "Each image file should start with one of the defined gesture labels."
    )
    parser.add_argument('--folder', required=True, help='Path to folder containing images')
    parser.add_argument('--static_image_mode', action='store_true',
                        help='Run MediaPipe in static image mode (recommended for photos)')
    parser.add_argument("--min_detection_confidence", type=float, default=0.7,
                        help='Minimum detection confidence for MediaPipe')
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                        help='Minimum tracking confidence for MediaPipe')
    args = parser.parse_args()

    folder_path = args.folder

    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=args.static_image_mode,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    # Find all image files in the folder.
    image_paths = glob.glob(os.path.join(folder_path, '*.*'))
    if not image_paths:
        print("No images found in the folder.")
        return

    for image_path in image_paths:
        # Get the gesture label index from the filename.
        label_index = get_label_index_from_filename(image_path, gestures)
        if label_index is None:
            print(f"Skipping {image_path}: no valid gesture label found.")
            continue

        image = cv.imread(image_path)
        if image is None:
            print(f"Unable to read image: {image_path}")
            continue

        # Process the image with MediaPipe.
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_processed_landmarks = pre_process_landmark(landmark_list)
                logging_csv(label_index, pre_processed_landmarks)
                print(f"Processed {image_path} with label index {label_index}")
        else:
            print(f"No hand detected in image: {image_path}")

    print("Dataset creation complete.")

if __name__ == '__main__':
    main()
