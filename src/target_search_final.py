import cv2
import torch
import timm
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from pathlib import Path

# --- IMPORT SAHI (Dùng cách import an toàn như đã sửa) ---
try:
    from sahi.models.ultralytics import Yolov8DetectionModel
except ImportError:
    try:
        from sahi.model import Yolov8DetectionModel
    except ImportError:
        from sahi import AutoDetectionModel # Fallback
        Yolov8DetectionModel = None 

from sahi.predict import get_sliced_prediction

# --- --- --- --- --- --- --- --- --- --- ---
# --- 1. CẤU HÌNH HỆ THỐNG ---
# --- --- --- --- --- --- --- --- --- --- ---

MODEL_PATH = r'D:\code\observing\results\kaggle\working\runs\detect\zalo_aero_run_2_medium\weights\best.pt' 
BASE_SAMPLES_DIR = 'train/samples'
VIDEO_TO_TEST = 'Lifering_1' # Thử với video khó
TARGET_CLASS_ID = 3          # ID: Lifering=3
SIMILARITY_THRESHOLD = 0.25  # Ngưỡng giống nhau

# Cấu hình SAHI (Cắt ảnh)
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_RATIO = 0.2

# --- CLASS TRACKER ĐƠN GIẢN (Tích hợp sẵn) ---
class SimpleTracker:
    def __init__(self, max_lost_frames=5, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = [] # List các track: {'id': 1, 'bbox': [x1,y1,x2,y2], 'lost': 0, 'score': 0.9}
        self.max_lost = max_lost_frames
        self.iou_thresh = iou_threshold

    def update(self, detections):
        # detections: list of {'bbox': [x1,y1,x2,y2], 'score': float}
        updated_tracks = []
        
        # 1. Cố gắng ghép detection mới với track cũ (Dựa trên IoU)
        for det in detections:
            best_iou = 0
            best_track_idx = -1
            
            for i, track in enumerate(self.tracks):
                iou = self.calculate_iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = i
            
            if best_iou > self.iou_thresh:
                # Tìm thấy cặp khớp -> Cập nhật track cũ
                track = self.tracks.pop(best_track_idx)
                track['bbox'] = det['bbox']
                track['score'] = det['score'] # Cập nhật điểm ReID mới nhất
                track['lost'] = 0
                updated_tracks.append(track)
            else:
                # Không khớp ai -> Tạo track mới (ID mới)
                updated_tracks.append({
                    'id': self.next_id,
                    'bbox': det['bbox'],
                    'score': det['score'],
                    'lost': 0
                })
                self.next_id += 1
        
        # 2. Xử lý các track bị mất dấu (không tìm thấy trong frame này)
        for track in self.tracks:
            track['lost'] += 1
            if track['lost'] <= self.max_lost:
                updated_tracks.append(track) # Giữ lại thêm vài frame nữa
                
        self.tracks = updated_tracks
        return self.tracks

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# --- --- --- --- --- --- --- --- --- ---
# --- 2. CÁC HÀM HỖ TRỢ (ReID) ---
# --- --- --- --- --- --- --- --- --- ---

def setup_reid_model(device):
    print("Đang tải model Re-ID (Fine-tuned)...")
    # Tạo khung model
    model = timm.create_model('resnet50', pretrained=False, num_classes=0).to(device)
    
    # --- LOAD WEIGHTS CỦA BẠN VÀO ĐÂY ---
    # Đảm bảo file .pth nằm cùng thư mục hoặc trỏ đúng đường dẫn
    checkpoint = torch.load("reid_finetuned.pth", map_location=device)
    model.load_state_dict(checkpoint)
    # ------------------------------------
    
    model.eval()
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config)
    
    return model, transform

def get_embedding(cv2_image, reid_model, transform, device):
    if cv2_image is None or cv2_image.size == 0: return None
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = reid_model(img_tensor)
    return F.normalize(embedding, p=2, dim=1)

# --- --- --- --- --- --- --- --- --- ---
# --- 3. CHƯƠNG TRÌNH CHÍNH ---
# --- --- --- --- --- --- --- --- --- ---

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Thiết bị: {device}")

    # 3.1. Tải YOLO (SAHI)
    print(f"Tải YOLO từ: {MODEL_PATH}")
    try:
        if 'Yolov8DetectionModel' in globals() and Yolov8DetectionModel:
            detection_model = Yolov8DetectionModel(
                model_path=MODEL_PATH, confidence_threshold=0.2, device=str(device)
            )
        else:
            from sahi import AutoDetectionModel
            detection_model = AutoDetectionModel.from_pretrained(
                model_type='yolov8', model_path=MODEL_PATH, confidence_threshold=0.2, device=str(device)
            )
    except Exception as e:
        print(f"Lỗi tải SAHI: {e}")
        exit()

    # 3.2. Tải Re-ID & Tracker
    reid_model, reid_transform = setup_reid_model(device)
    tracker = SimpleTracker(max_lost_frames=10) # Giữ box thêm 10 frames nếu bị mất dấu

    # 3.3. Tạo Reference Embedding
    print(f"Tạo mẫu tham chiếu cho: {VIDEO_TO_TEST}")
    ref_dir = Path(BASE_SAMPLES_DIR) / VIDEO_TO_TEST / 'object_images'
    ref_embs = []
    for p in ref_dir.glob('*.jpg'):
        img = cv2.imread(str(p))
        if img is not None:
            emb = get_embedding(img, reid_model, reid_transform, device)
            if emb is not None: ref_embs.append(emb)

    if not ref_embs:
        print("Lỗi: Không có ảnh tham chiếu.")
        exit()
    master_ref_emb = torch.mean(torch.cat(ref_embs), dim=0, keepdim=True)

    # 3.4. Xử lý Video
    vid_path = Path(BASE_SAMPLES_DIR) / VIDEO_TO_TEST / 'drone_video.mp4'
    out_path = f"{VIDEO_TO_TEST}_final_result.mp4"
    cap = cv2.VideoCapture(str(vid_path))
    
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(5))
    total = int(cap.get(7))
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"Bắt đầu xử lý {total} frames...")
    
    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret: break

        # --- BƯỚC 1: SAHI DETECT ---
        result = get_sliced_prediction(
            frame, detection_model,
            slice_height=SLICE_HEIGHT, slice_width=SLICE_WIDTH,
            overlap_height_ratio=OVERLAP_RATIO, overlap_width_ratio=OVERLAP_RATIO,
            verbose=0
        )

        # --- BƯỚC 2: RE-ID VERIFY ---
        valid_detections = []
        for pred in result.object_prediction_list:
            if pred.category.id == TARGET_CLASS_ID:
                bbox = pred.bbox
                x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
                
                # Crop & ReID
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = frame[y1:y2, x1:x2]
                
                emb = get_embedding(crop, reid_model, reid_transform, device)
                if emb is not None:
                    sim = F.cosine_similarity(master_ref_emb, emb).item()
                    if sim >= SIMILARITY_THRESHOLD:
                        valid_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'score': sim
                        })

        # --- BƯỚC 3: TRACKING UPDATE ---
        # Cập nhật vị trí dựa trên các detection đã qua vòng Re-ID
        tracked_objects = tracker.update(valid_detections)

        # Vẽ kết quả
        for obj in tracked_objects:
            x1, y1, x2, y2 = obj['bbox']
            # Nếu vật thể đang bị "mất dấu" tạm thời, vẽ màu đỏ nhạt, còn không vẽ màu xanh
            color = (0, 255, 0) if obj['lost'] == 0 else (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"ID:{obj['id']} ({obj['score']:.2f})", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Hoàn tất! Video lưu tại: {out_path}")