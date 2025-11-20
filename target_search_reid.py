import cv2
import torch
import timm
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

CLASS_NAMES = {
    0: 'Backpack',
    1: 'Jacket',
    2: 'Laptop',
    3: 'Lifering',
    4: 'MobilePhone',
    5: 'Person1',
    6: 'WaterBottle'
}
# --- --- --- --- --- --- --- --- --- --- ---
# --- 1. THIẾT LẬP CÁC ĐƯỜNG DẪN & THAM SỐ ---
# --- --- --- --- --- --- --- --- --- --- ---

MODEL_PATH = r'D:\code\observing\results\kaggle\working\runs\detect\zalo_aero_run_2_medium\weights\best.pt' 

BASE_SAMPLES_DIR = 'train/samples'

# Tên video muốn test ('Backpack_0', 'Person1_1')
VIDEO_TO_TEST = 'Person1_1'

# ID của class bạn muốn tìm (tra trong file data.yaml)
# 0: 'Backpack', 1: 'Jacket', 2: 'Laptop', ...
TARGET_CLASS_ID = 5

# Ngưỡng Re-ID: Cần "giống" bao nhiêu % để coi là khớp?
# Đây là "nút vặn" mới của bạn thay cho MIN_MATCH_COUNT
# (Bắt đầu ở 0.4, tăng lên nếu bị sai, giảm xuống nếu bị sót)
SIMILARITY_THRESHOLD = 0.3

# --- --- --- --- --- --- --- --- --- ---
# --- 2. CÁC HÀM XỬ LÝ RE-ID (GIAI ĐOẠN 2) ---
# --- --- --- --- --- --- --- --- --- ---

def setup_reid_model(device):
    """Tải model AI (ResNet-50) để tạo 'dấu vân tay' (embedding)."""
    print("Đang tải model Re-ID (ResNet-50)...")
    # Tải ResNet-50, đã pre-train, và bỏ đi lớp cuối (lớp phân loại)
    # để nó trở thành một 'bộ trích xuất đặc trưng' (feature extractor)
    model = timm.create_model('resnet50', pretrained=True, num_classes=0).to(device)
    model.eval() # Chuyển sang chế độ "dự đoán"
    
    # Lấy thông tin config của model để tạo bộ tiền xử lý ảnh
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config)
    
    print("Tải model Re-ID thành công.")
    return model, transform

def get_embedding(cv2_image, reid_model, transform, device):
    """Biến một ảnh (dạng CV2) thành một 'dấu vân tay' (embedding vector)."""
    if cv2_image.size == 0:
        return None
        
    # Model AI cần ảnh dạng PIL và đúng chuẩn (RGB)
    img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    
    # Áp dụng tiền xử lý (resize, normalize, ...)
    img_tensor = transform(img_pil).unsqueeze(0).to(device) # Thêm 1 chiều 'batch'
    
    # Chạy model và tắt 'gradient' (để tiết kiệm bộ nhớ)
    with torch.no_grad():
        embedding = reid_model(img_tensor)
        
    # Chuẩn hóa vector (để so sánh Cosine)
    embedding = F.normalize(embedding, p=2, dim=1)
    
    return embedding

# --- --- --- --- --- --- --- --- --- ---
# --- 3. CHƯƠNG TRÌNH CHÍNH ---
# --- --- --- --- --- --- --- --- --- ---

print("Bắt đầu chương trình tìm kiếm mục tiêu (YOLO + Re-ID)...")

# 3.1. Thiết lập thiết bị (Dùng GPU nếu có)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Đang sử dụng thiết bị: {device}")

# 3.2. Tải Model YOLO (Bộ lọc - Giai đoạn 1)
print(f"Đang tải model YOLO từ: {MODEL_PATH}")
yolo_model = YOLO(MODEL_PATH)
yolo_model.fuse() # Tăng tốc

# 3.3. Tải Model Re-ID (Bộ xác minh - Giai đoạn 2)
reid_model, reid_transform = setup_reid_model(device)

# 3.4. Tạo "Dấu vân tay" cho vật thể tham chiếu
print(f"Đang tạo dấu vân tay Re-ID cho '{VIDEO_TO_TEST}'...")
reference_dir = Path(BASE_SAMPLES_DIR) / VIDEO_TO_TEST / 'object_images'
reference_embeddings = [] # Nơi lưu trữ "dấu vân tay"

if not reference_dir.exists():
    print(f"LỖI: Không tìm thấy thư mục tham chiếu: {reference_dir}")
    exit()

for img_path in reference_dir.glob('*.jpg'):
    ref_img = cv2.imread(str(img_path))
    if ref_img is None:
        continue
    emb = get_embedding(ref_img, reid_model, reid_transform, device)
    if emb is not None:
        reference_embeddings.append(emb)

if not reference_embeddings:
    print(f"LỖI: Không trích xuất được Re-ID từ thư mục tham chiếu: {reference_dir}")
    exit()
    
# Gộp các dấu vân tay (nếu có 3 ảnh, ta lấy trung bình)
master_reference_embedding = torch.mean(torch.cat(reference_embeddings), dim=0, keepdim=True)
print(f"Đã tạo xong 'dấu vân tay' tham chiếu.")

# 3.5. Xử lý Video
video_path = Path(BASE_SAMPLES_DIR) / VIDEO_TO_TEST / 'drone_video.mp4'
output_video_path = f"{VIDEO_TO_TEST}_reid_result.mp4"

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"LỖI: Không thể mở video: {video_path}")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print(f"Đang xử lý video... (Tổng cộng {total_frames} frames)")

for _ in tqdm(range(total_frames)):
    ret, frame = cap.read()
    if not ret:
        break
        
    final_boxes = []

    # --- GIAI ĐOẠN 1: LỌC (YOLO) ---
    results = yolo_model.predict(frame, conf=0.25, verbose=False, device=device)
    
    for box in results[0].boxes:
        if int(box.cls) == TARGET_CLASS_ID:
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # --- GIAI ĐOẠN 2: XÁC MINH (RE-ID) ---
            candidate_crop = frame[y1:y2, x1:x2]
            
            emb_candidate = get_embedding(candidate_crop, reid_model, reid_transform, device)
            
            if emb_candidate is None:
                continue
            
            # So sánh dấu vân tay
            similarity = F.cosine_similarity(master_reference_embedding, emb_candidate)
            similarity_score = similarity.item() # Lấy giá trị (ví dụ: 0.55)
            print(f"DEBUG: Tìm thấy 1 ứng viên (ID {int(box.cls)}). Độ giống: {similarity_score:.3f}")
            
            # Nếu điểm giống > ngưỡng
            if similarity_score >= SIMILARITY_THRESHOLD:
                final_boxes.append((x1, y1, x2, y2, similarity_score))

    # 3.6. Vẽ kết quả cuối cùng lên frame
    for (x1, y1, x2, y2, score) in final_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_name = CLASS_NAMES.get(TARGET_CLASS_ID, "Target")
        cv2.putText(frame, f"{label_name} ({score:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
    out.write(frame)

# Dọn dẹp
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"--- HOÀN TẤT! ---")
print(f"Video kết quả đã được lưu tại: {output_video_path}")