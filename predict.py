import sys
import os
import csv
import time
import argparse
import cv2
import torch
import timm
import numpy as np
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from torchvision import transforms

try:
    from sahi.models.ultralytics import Yolov8DetectionModel
except ImportError:
    try:
        from sahi.model import Yolov8DetectionModel
    except ImportError:
        from sahi import AutoDetectionModel
        Yolov8DetectionModel = None 
from sahi.predict import get_sliced_prediction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SLICE_SIZE = 640
OVERLAP_RATIO = 0.2
SIMILARITY_THRESHOLD = 0.25
CLASS_MAPPING = {
    'Backpack': 0, 'Jacket': 1, 'Laptop': 2, 'Lifering': 3,
    'MobilePhone': 4, 'Person1': 5, 'WaterBottle': 6
}

def setup_reid_model(model_path):
    model = timm.create_model('resnet50', pretrained=False, num_classes=0).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.eval()
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config)
    return model, transform

def get_embedding(cv2_image, model, transform):
    if cv2_image is None or cv2_image.size == 0: return None
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        embedding = model(img_tensor)
    return F.normalize(embedding, p=2, dim=1)

def process_video(video_id, video_dir, detection_model, reid_model, reid_transform):
    start_time = time.time()
    
    target_class_name = video_id.split('_')[0]
    if target_class_name not in CLASS_MAPPING:
        return [], 0
    target_class_id = CLASS_MAPPING[target_class_name]

    ref_dir = os.path.join(video_dir, 'object_images')
    ref_embs = []
    if os.path.exists(ref_dir):
        for img_name in os.listdir(ref_dir):
            if img_name.endswith(('.jpg', '.png')):
                img = cv2.imread(os.path.join(ref_dir, img_name))
                emb = get_embedding(img, reid_model, reid_transform)
                if emb is not None: ref_embs.append(emb)
    
    if not ref_embs:
        return [], (time.time() - start_time) * 1000
        
    master_ref_emb = torch.mean(torch.cat(ref_embs), dim=0, keepdim=True)

    video_path = os.path.join(video_dir, 'drone_video.mp4')
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    
    answers = []
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        
        result = get_sliced_prediction(
            frame, detection_model,
            slice_height=SLICE_SIZE, slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO, overlap_width_ratio=OVERLAP_RATIO,
            verbose=0
        )

        frame_answers = []
        for pred in result.object_prediction_list:
            if pred.category.id == target_class_id:
                bbox = pred.bbox
                x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                
                emb = get_embedding(crop, reid_model, reid_transform)
                if emb is not None:
                    sim = F.cosine_similarity(master_ref_emb, emb).item()
                    if sim >= SIMILARITY_THRESHOLD:
                        w_box = x2 - x1
                        h_box = y2 - y1
                        answers.append([x1, y1, w_box, h_box])

    cap.release()
    
    elapsed_ms = (time.time() - start_time) * 1000
    return answers, elapsed_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to public_test folder')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save submission.csv')
    args = parser.parse_args()

    print("Loading models...")
    yolo_path = os.path.join("model", "yolov8m.pt")
    if 'Yolov8DetectionModel' in globals() and Yolov8DetectionModel:
        detection_model = Yolov8DetectionModel(model_path=yolo_path, confidence_threshold=0.2, device=str(DEVICE))
    else:
        from sahi import AutoDetectionModel
        detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', model_path=yolo_path, confidence_threshold=0.2, device=str(DEVICE))

    reid_path = os.path.join("model", "reid_finetuned.pth")
    reid_model, reid_transform = setup_reid_model(reid_path)

    print(f"Processing from {args.input_path}...")
    with open(args.output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer', 'time'])

        if not os.path.exists(args.input_path):
            print("Input path not found!")
            exit()

        video_folders = sorted([d for d in os.listdir(args.input_path) if os.path.isdir(os.path.join(args.input_path, d))])
        
        for video_id in video_folders:
            print(f"Processing {video_id}...")
            video_dir = os.path.join(args.input_path, video_id)
            answers, time_ms = process_video(video_id, video_dir, detection_model, reid_model, reid_transform)
            writer.writerow([video_id, str(answers), int(time_ms)])

    print(f"Done! Submission saved to {args.output_path}")