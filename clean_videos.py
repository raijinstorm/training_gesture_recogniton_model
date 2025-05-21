import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

def convert_videos_to_mp4(source_folder, dest_folder):
    #convert to mp4
    pass

def show_videos_stats(dataset_path, dest_folder):
    """
    Display statistics from dataset
    
    Args: 
        dataset_path (str): path to dataset file
        
    Return: void
    """
    os.makedirs(dest_folder, exist_ok=True)
    
    df = pd.read_csv(dataset_path)
    print(df.describe())
    for col in df.columns:
        if (col == "filename"):
            continue
        
        plt.figure(figsize=(6, 3))
        
        # First subplot: Box Plot (only for numeric columns)
        plt.subplot(1, 2, 1)
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.boxplot(y=df[col])
            plt.title(f'Box Plot of {col}')
        
        # Second subplot: Another plot
        plt.subplot(1, 2, 2)
        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, use a histogram with KDE overlay
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
        
        plt.tight_layout()
        # Save the combined plots for this column to a file
        plt.savefig(f'{dest_folder}/{col}_plots.png')
        plt.close()
        
        # 1. Extract gesture label (prefix before first underscore)
    gestures = df['filename'].dropna().apply(lambda x: x.split('_')[0])

    # 2. Count and sort
    counts = gestures.value_counts().sort_values(ascending=False)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Gesture Distribution by Class')
    plt.xlabel('Gesture Class')
    plt.ylabel('Count')
    plt.tight_layout()

    # 4. Save to disk
    plt.savefig(os.path.join(dest_folder, 'gesture_distribution.png'))
    plt.close()

def get_cleaned_csv(input_path, output_path, q1, q3):
    import pandas as pd
    
    # Load the dataset
    df = pd.read_csv(input_path)

    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(q1)
        Q3 = df[col].quantile(q3)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

    # Save the cleaned dataset
    df.to_csv(output_path, index=False)
       
def create_csv_from_vids(source_folder,output_csv ):
    video_data = []
    
    for filename in os.listdir(source_folder):
        if filename.lower().endswith('.mp4'):
            video_path = os.path.join(source_folder, filename)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error opening {filename}")
                continue
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            duration = frame_count / fps if fps else 0
            
            video_data.append({
                "filename": filename,
                "fps": fps,
                "frame_count": frame_count,
                "width": int(width),
                "height": int(height),
                "duration_sec": duration
            })
            
            cap.release()
    
    df = pd.DataFrame(video_data)
    df.to_csv(output_csv, index=False)
    
    print(f"Metadata for {len(df)} videos saved to {output_csv}")

def save_accepted_videos(source_folder ,dest_folder, path_to_csv):
    df = pd.read_csv(path_to_csv)
    # Create destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Iterate over each filename in the dataset
    for fname in df["filename"]:
        src_path = os.path.join(source_folder, fname)
        dst_path = os.path.join(dest_folder, fname)
        
        # Check if the file exists in the source folder
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"File not found: {fname}")
        
def get_cleaned_videos(source_folder, dest_folder, q1 = 0.3, q3 = 0.97, is_test_run = False):
    """
    Function is made to get insighits from video specs, then If needed filter
    videos according to thresholds 
    
    Args: 
        source_folder (str): Path to the folder with video files in MP4 format
        dest_folder (str): Path to the folder to save good videos.
        
    Return: void
    """
    
    os.makedirs(dest_folder, exist_ok=True)
    
    #create dataset
    vids_csv = dest_folder + "\\" + "vids.csv"
    create_csv_from_vids(source_folder, vids_csv)
    
    #temporary
    #vids_csv = os.path.join(dest_folder, "video_metadata.csv")
    
    #show stats
    show_videos_stats(vids_csv, os.path.join(dest_folder, "stats_before_filtering"))
    
    #filter dataset
    clean_csv = os.path.join(dest_folder, "clean.csv")
    get_cleaned_csv(vids_csv, clean_csv, q1, q3)
    
    
    #show stats
    show_videos_stats(clean_csv, os.path.join(dest_folder, "stats_after_filtering"))
    
    #copy videos that pass filtering into new folder
    if not is_test_run:
        save_accepted_videos(source_folder, os.path.join(dest_folder, "cleaned_videos"), clean_csv)
        print("vids are copied") 


