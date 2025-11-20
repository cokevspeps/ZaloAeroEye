import cv2
import os
from pathlib import Path

def extract_frames(base_data_path, output_base_path):
    """
    Extracts all frames from videos based on the directory structure.

    The script iterates through all subfolders in:
    base_data_path/samples/
    
    It looks for 'drone_video.mp4' inside each subfolder, e.g.:
    .../samples/Backpack_0/drone_video.mp4
    .../samples/Jacket_1/drone_video.mp4
    
    Frames will be saved to a matching structure in:
    output_base_path/Backpack_0/frame_00000.jpg ...
    output_base_path/Jacket_1/frame_00000.jpg ...
    """
    
    # Define the path to the samples directory
    samples_dir = Path(base_data_path) / "samples"
    
    # Define the top-level directory to save extracted frames
    frames_output_dir = Path(output_base_path)
    
    # Create the base output directory if it doesn't exist
    frames_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Looking for video samples in: {samples_dir}")
    print(f"Saving extracted frames to: {frames_output_dir}")
    print("---")

    # Iterate through each item in the 'samples' directory
    # This will loop over 'Backpack_0', 'Backpack_1', 'Jacket_0', etc.
    for video_folder in samples_dir.iterdir():
        if video_folder.is_dir():
            video_name = video_folder.name  # e.g., "Backpack_0"
            video_file_path = video_folder / "drone_video.mp4"
            
            if not video_file_path.exists():
                print(f"Warning: No 'drone_video.mp4' found in {video_folder}. Skipping.")
                continue

            # Create a specific output folder for this video's frames
            video_specific_output_dir = frames_output_dir / video_name
            video_specific_output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Processing video: {video_file_path}...")
            
            # --- Start OpenCV video processing ---
            cap = cv2.VideoCapture(str(video_file_path))
            
            if not cap.isOpened():
                print(f"Error: Could not open video file {video_file_path}")
                continue
                
            frame_count = 0
            while True:
                # Read one frame from the video
                success, frame = cap.read()
                
                # If 'success' is False, we've reached the end of the video
                if not success:
                    break
                
                # Construct the output filename
                # Using 6-digit padding (e.g., 000001) is good for sorting
                frame_filename = f"frame_{frame_count:06d}.jpg"
                output_frame_path = video_specific_output_dir / frame_filename
                
                # Save the frame as a JPG image
                cv2.imwrite(str(output_frame_path), frame)
                
                frame_count += 1
            
            # Release the video capture object
            cap.release()
            print(f"Successfully extracted {frame_count} frames to {video_specific_output_dir}")
            # --- End OpenCV video processing ---

    print("---")
    print("All videos processed.")

# --- --- --- --- --- --- --- --- --- --- --- ---
# --- RUN THE SCRIPT ---
# --- --- --- --- --- --- --- --- --- --- --- ---

# Define your main 'train' folder path
# This is the folder that contains 'samples' and 'annotations'
train_folder_path = "train" # <-- CHANGED THIS LINE

# We'll save the frames in a new folder at the same level
output_folder = "extracted_frames" # <-- CHANGED THIS LINE

# Call the function
extract_frames(train_folder_path, output_folder)