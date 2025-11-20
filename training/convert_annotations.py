import json
import cv2
import os
from pathlib import Path

CLASS_MAP = {
    "Backpack": 0,
    "Jacket": 1,
    "Laptop": 2,
    "Lifering": 3,
    "MobilePhone": 4,
    "Person1": 5,
    "WaterBottle": 6
}

def convert_to_yolo(bbox, frame_width, frame_height, class_id):
    """
    Converts a [x, y, w, h] bounding box to the
    YOLO [class_id, x_center, y_center, width, height] format.
    
    All coordinates are normalized from 0.0 to 1.0.
    """
    x, y, w, h = bbox
    
    x_center = x + w / 2
    y_center = y + h / 2
    
    x_center_norm = x_center / frame_width
    y_center_norm = y_center / frame_height
    width_norm = w / frame_width
    height_norm = h / frame_height
    
    # class_id giờ được truyền vào
    return f"{class_id} {x_center_norm} {y_center_norm} {width_norm} {height_norm}\n"

def process_annotations(base_dir):
    """
    Loads annotations.json, finds corresponding videos to get dimensions,
    and writes YOLO-formatted .txt files to extracted_labels.
    """
    base_path = Path(base_dir)
    annotations_file = base_path / "train" / "annotations" / "annotations.json"
    samples_dir = base_path / "train" / "samples"
    labels_output_dir = base_path / "extracted_labels"
    
    labels_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading annotations from: {annotations_file}")
    with open(annotations_file, 'r') as f:
        all_annotations = json.load(f)
    print("Annotations loaded successfully.")
    print("---")
    
    if isinstance(all_annotations, dict):
        iterator = all_annotations.items()
        is_list_format = False
    elif isinstance(all_annotations, list):
        iterator = ((entry.get('video_id'), entry) for entry in all_annotations if entry.get('video_id'))
        is_list_format = True
    else:
        print("Unknown annotations format. Expected dict or list. Exiting.")
        return

    for video_name, video_data in iterator:
        print(f"Processing video: {video_name}...")
        
        category_name = video_name.split('_')[0]
        
        # Lấy class_id từ MAP
        if category_name not in CLASS_MAP:
            print(f"  Warning: Không tìm thấy category '{category_name}' trong CLASS_MAP. Bỏ qua video {video_name}.")
            continue # Bỏ qua video này
        
        class_id = CLASS_MAP[category_name]

        video_file_path = samples_dir / video_name / "drone_video.mp4"

        if not video_file_path.exists():
            print(f"  Warning: Video file not found at {video_file_path}. Skipping.")
            continue

        cap = cv2.VideoCapture(str(video_file_path))
        if not cap.isOpened():
            print(f"  Error: Could not open video {video_file_path}. Skipping.")
            continue

        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if frame_width == 0 or frame_height == 0:
            print(f"  Error: Could not read dimensions from {video_file_path}. Skipping.")
            continue

        video_label_output_dir = labels_output_dir / video_name
        video_label_output_dir.mkdir(parents=True, exist_ok=True)

        annotated_frames_count = 0

        if not is_list_format:
            # original format
            for frame_str, annotation in video_data.items():
                try:
                    frame_num = int(frame_str)

                    if "bbox" not in annotation:
                        continue

                    bbox = annotation["bbox"]
                    if not bbox or len(bbox) == 0:
                        continue

                    yolo_data = convert_to_yolo(bbox, frame_width, frame_height, class_id)

                    label_filename = f"frame_{frame_num:06d}.txt"
                    label_file_path = video_label_output_dir / label_filename

                    with open(label_file_path, 'w') as label_file:
                        label_file.write(yolo_data)

                    annotated_frames_count += 1

                except Exception as e:
                    print(f"  Error processing frame {frame_str} in {video_name}: {e}")

        else:
            # new list-based format
            for ann_block in video_data.get('annotations', []):
                for bbox_obj in ann_block.get('bboxes', []):
                    try:
                        frame_num = int(bbox_obj.get('frame'))
                        x1 = int(bbox_obj.get('x1'))
                        y1 = int(bbox_obj.get('y1'))
                        x2 = int(bbox_obj.get('x2'))
                        y2 = int(bbox_obj.get('y2'))

                        w = x2 - x1
                        h = y2 - y1

                        yolo_data = convert_to_yolo([x1, y1, w, h], frame_width, frame_height, class_id)

                        label_filename = f"frame_{frame_num:06d}.txt"
                        label_file_path = video_label_output_dir / label_filename
                        
                        with open(label_file_path, 'a') as label_file:
                            label_file.write(yolo_data)

                        annotated_frames_count += 1

                    except Exception as e:
                        print(f"  Error processing bbox for frame {bbox_obj.get('frame')} in {video_name}: {e}")

        print(f"  Successfully created {annotated_frames_count} label files for {video_name}.")
        
    print("---")
    print("All annotations processed.")

base_folder = "." 
process_annotations(base_folder)