from ultralytics import YOLO
from PIL import Image
import os

MODEL_PATH = r'D:\code\observing\results\kaggle\working\runs\detect\zalo_aero_run_2_medium\weights\best.pt'

IMAGE_TO_TEST = r'D:\code\observing\extracted_frames\Person1_1\frame_002703.jpg' 

if not os.path.exists(MODEL_PATH):
    print(f"Lỗi: Không tìm thấy model tại: {MODEL_PATH}")
    exit()
if not os.path.exists(IMAGE_TO_TEST):
    print(f"Lỗi: Không tìm thấy ảnh tại: {IMAGE_TO_TEST}")
    exit()

# 3. Tải model
print(f"Đang tải model từ {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
print("Tải model thành công!")

# 4. Chạy dự đoán
print(f"Đang chạy dự đoán trên {IMAGE_TO_TEST}...")
results = model.predict(
    source=IMAGE_TO_TEST, 
    save=True,
    conf=0.1,
    
    project='runs/detect',
    name='predict',
    exist_ok=True
)
# 5. In kết quả và hiển thị
print("Đã chạy xong!")
if results:
    # Lấy đường dẫn của ảnh đã lưu
    # Nó sẽ được lưu trong thư mục 'runs/detect/predict'
    saved_image_path = results[0].save_path 
    print(f"Kết quả đã được lưu tại: {saved_image_path}")
    
    # Mở ảnh kết quả lên
    Image.open(saved_image_path).show()

print("--- Kết thúc ---")