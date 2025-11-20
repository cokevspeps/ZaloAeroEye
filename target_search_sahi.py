import cv2
import torch
import timm
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

# --- NHẬP THƯ VIỆN SAHI ---
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# --- --- --- --- --- --- --- --- --- --- ---
# --- 1. THIẾT LẬP CÁC ĐƯỜNG DẪN & THAM SỐ ---
# --- --- --- --- --- --- --- --- --- --- ---

MODEL_PATH = r'D:\code\observing\results\kaggle\working\runs\detect\zalo_aero_run_2_medium\weights\best.pt' 
BASE_SAMPLES_DIR = 'train/samples'
VIDEO_TO_TEST = 'Lifering_1'  # Thử ngay với ca khó Lifering!
TARGET_CLASS_ID = 3           # ID của Lifering là 3
SIMILARITY_THRESHOLD = 0.25   # Ngưỡng Re-ID (Nên để thấp để bắt đầu)

# --- CẤU HÌNH SAHI (QUAN TRỌNG) ---
SLICE_HEIGHT = 640
SLICE_WIDTH = 640
OVERLAP_HEIGHT_RATIO = 0.2 # Chồng lấn 20% để không cắt đôi vật thể
OVERLAP_WIDTH_RATIO = 0.2

# Tên các class để hiển thị
CLASS_NAMES = {
    0: 'Backpack', 1: 'Jacket', 2: 'Laptop', 3: 'Lifering',
    4: 'MobilePhone', 5: 'Person', 6: 'WaterBottle'
}

# --- --- --- --- --- --- --- --- --- ---
# --- 2. CÁC HÀM RE-ID (GIỮ NGUYÊN) ---
# --- --- --- --- --- --- --- --- --- ---

def setup_reid_model(device):
    print("Đang tải model Re-ID (ResNet-50)...")
    model = timm.create_model('resnet50', pretrained=True, num_classes=0).to(device)
    model.eval()
    # SỬA LỖI TIMM: Dùng cấu hình rỗng nhưng lấy model làm chuẩn
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

print("Bắt đầu chương trình tìm kiếm (SAHI + Re-ID)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Thiết bị: {device}")

# 3.1. Tải Model YOLO thông qua SAHI
print(f"Đang tải model YOLO-SAHI từ: {MODEL_PATH}")
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=0.2, # Độ tự tin tối thiểu để YOLO bắt vật thể
    device=str(device) # SAHI cần device dạng string 'cuda:0' hoặc 'cpu'
)

# 3.2. Tải Model Re-ID
reid_model, reid_transform = setup_reid_model(device)

# 3.3. Tạo dấu vân tay tham chiếu (Reference)
print(f"Đang tạo Reference Embedding cho '{VIDEO_TO_TEST}'...")
reference_dir = Path(BASE_SAMPLES_DIR) / VIDEO_TO_TEST / 'object_images'
reference_embeddings = []

if not reference_dir.exists():
    print(f"LỖI: Không tìm thấy: {reference_dir}")
    exit()

for img_path in reference_dir.glob('*.jpg'):
    ref_img = cv2.imread(str(img_path))
    if ref_img is None: continue
    emb = get_embedding(ref_img, reid_model, reid_transform, device)
    if emb is not None: reference_embeddings.append(emb)

if not reference_embeddings:
    print("LỖI: Không tạo được embedding tham chiếu.")
    exit()

master_reference_embedding = torch.mean(torch.cat(reference_embeddings), dim=0, keepdim=True)
print("Đã tạo xong Reference Embedding.")

# 3.4. Xử lý Video
video_path = Path(BASE_SAMPLES_DIR) / VIDEO_TO_TEST / 'drone_video.mp4'
output_video_path = f"{VIDEO_TO_TEST}_sahi_result.mp4"

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"LỖI: Không mở được video {video_path}")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

print(f"Đang xử lý {total_frames} frames với SAHI (Sẽ chậm hơn bình thường)...")

# Loop xử lý
for i in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret: break
    
    # --- GIAI ĐOẠN 1: LỌC BẰNG SAHI ---
    # Thay vì model.predict(), ta dùng get_sliced_prediction
    # Nó tự động cắt ảnh, detect, và gộp kết quả
    result = get_sliced_prediction(
        frame,
        detection_model,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        verbose=0 # Tắt log in ra màn hình cho đỡ rối
    )
    
    final_boxes = []
    
    # Duyệt qua các object mà SAHI tìm thấy
    for object_prediction in result.object_prediction_list:
        # Kiểm tra class ID
        # Lưu ý: SAHI trả về category.id
        if object_prediction.category.id == TARGET_CLASS_ID:
            
            # Lấy bbox (SAHI trả về box dạng [minx, miny, maxx, maxy])
            bbox = object_prediction.bbox
            x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            
            # --- GIAI ĐOẠN 2: XÁC MINH (RE-ID) ---
            # Cắt ảnh và kiểm tra độ giống
            # Cần clamp tọa độ để không cắt ra ngoài ảnh
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            candidate_crop = frame[y1:y2, x1:x2]
            
            emb_candidate = get_embedding(candidate_crop, reid_model, reid_transform, device)
            
            if emb_candidate is None: continue
            
            similarity = F.cosine_similarity(master_reference_embedding, emb_candidate).item()
            
            # Debug nhẹ để xem SAHI tìm được gì
            # print(f"Frame {i}: Found candidate (Sim: {similarity:.2f})")

            if similarity >= SIMILARITY_THRESHOLD:
                final_boxes.append((x1, y1, x2, y2, similarity))

    # Vẽ kết quả
    for (x1, y1, x2, y2, score) in final_boxes:
        label_name = CLASS_NAMES.get(TARGET_CLASS_ID, "Target")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nXong! Video lưu tại: {output_video_path}")