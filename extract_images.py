import cv2
import numpy as np
import mediapipe as mp
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt


def is_blurry(image, threshold):
    """Return True if the image is considered blurry based on the given threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_too_dark(image, threshold):
    """Return True if the image is too dark based on the given threshold."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) < threshold

def has_hand(image):
    """Return True if a hand is detected in the image using MediaPipe Hands."""
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        return results.multi_hand_landmarks is not None

def check_image(image, blurry_threshold=15, dark_threshold=60, check_hand_flag=True):
    """
    Check if an image meets the criteria:
      - Not blurry if blurry_threshold > 0.
      - Not too dark if dark_threshold > 0.
      - Contains a hand if check_hand_flag is True.
    """
    if blurry_threshold > 0 and is_blurry(image, threshold=blurry_threshold):
        return False
    if dark_threshold > 0 and is_too_dark(image, threshold=dark_threshold):
        return False
    if check_hand_flag and not has_hand(image):
        return False
    return True

def extract_and_filter_images(source_folder, dest_folder, blurry_threshold=15, dark_threshold=60, check_hand_flag=True, write_images=True):
    """
    Process each video in the source_folder:
      - Extract 15 evenly spaced images.
      - Check each image with check_image using provided thresholds and hand check flag.
      - Save the image to dest_folder if it passes the checks and write_images is True.
    
    Args:
        source_folder (str): Folder with video files.
        dest_folder (str): Folder to save accepted images.
        blurry_threshold (int): Threshold for blurriness; if 0, skip blur check.
        dark_threshold (int): Threshold for darkness; if 0, skip dark check.
        check_hand_flag (bool): Whether to require a hand in the image.
        write_images (bool): Whether to write the passing images to disk.
    
    Returns:
        tuple: (total_images_extracted, total_good_images)
    """
    os.makedirs(dest_folder, exist_ok=True)
    total_images_extracted = 0
    total_good_images = 0

    for video_file in os.listdir(source_folder):
        if not video_file.lower().endswith(('.mp4', '.webm', '.avi', '.mov')):
            continue

        video_path = os.path.join(source_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Could not open video: {video_file}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration_sec = frame_count / fps if fps else 0

        # Generate 15 evenly spaced timestamps (in seconds)
        timestamps = np.linspace(0, duration_sec, num=15, endpoint=False)

        for idx, t in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at {t:.2f} sec in {video_file}")
                continue

            total_images_extracted += 1

            if check_image(frame, blurry_threshold, dark_threshold, check_hand_flag):
                total_good_images += 1
                if write_images:
                    base_name, _ = os.path.splitext(video_file)
                    image_filename = f"{base_name}_{idx}.jpg"
                    save_path = os.path.join(dest_folder, image_filename)
                    cv2.imwrite(save_path, frame)
            else:
                print(f"Rejected frame from video: {video_file}")

        cap.release()

    print(f"Total images extracted: {total_images_extracted}")
    print(f"Total good images: {total_good_images}")
    return total_images_extracted, total_good_images