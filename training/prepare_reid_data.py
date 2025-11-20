import json
import cv2
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
BASE_DIR = Path("train")  # Thư mục gốc chứa samples và annotations
OUTPUT_DIR = Path("reid_dataset") # Nơi lưu dữ liệu Re-ID

def prepare_data():
    # 1. Đọc file annotations
    anno_file = BASE_DIR / "annotations" / "annotations.json"
    with open(anno_file, 'r') as f:
        data = json.load(f)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)

    print("Đang cắt ảnh từ video để làm dữ liệu Re-ID...")

    # 2. Duyệt qua từng video
    # data có thể là dict hoặc list tùy phiên bản, xử lý cả 2
    iterator = data.items() if isinstance(data, dict) else ((x['video_id'], x) for x in data)

    for video_id, video_info in tqdm(iterator):
        # Tạo thư mục cho đối tượng này (ví dụ: reid_dataset/Backpack_0)
        obj_dir = OUTPUT_DIR / video_id
        obj_dir.mkdir(exist_ok=True)

        # A. Copy 3 ảnh tham chiếu (Object Images) vào đây
        ref_src_dir = BASE_DIR / "samples" / video_id / "object_images"
        if ref_src_dir.exists():
            for i, img_file in enumerate(ref_src_dir.glob("*.jpg")):
                shutil.copy(img_file, obj_dir / f"ref_{i}.jpg")
        
        # B. Cắt ảnh từ Video Drone
        video_path = BASE_DIR / "samples" / video_id / "drone_video.mp4"
        cap = cv2.VideoCapture(str(video_path))
        
        # Lấy danh sách frame cần cắt
        # Xử lý cấu trúc JSON (có thể khác nhau chút)
        frames_to_process = {}
        if isinstance(video_info, dict) and 'annotations' not in video_info:
             # Format cũ: "0": {"bbox":...}
             frames_to_process = video_info
        else:
             # Format mới: "annotations": [{"bboxes": ...}]
             # Flatten lại cho dễ xử lý
             if 'annotations' in video_info:
                 for block in video_info['annotations']:
                     for bbox in block['bboxes']:
                         frames_to_process[str(bbox['frame'])] = {'bbox': [bbox['x1'], bbox['y1'], bbox['x2']-bbox['x1'], bbox['y2']-bbox['y1']]}

        count = 0
        # Chỉ lấy mẫu khoảng 100-200 ảnh mỗi video để tránh quá tải
        step = max(1, len(frames_to_process) // 150) 
        
        sorted_frames = sorted([int(k) for k in frames_to_process.keys()])
        
        for i, frame_idx in enumerate(sorted_frames):
            if i % step != 0: continue # Skip bớt
            
            # Nhảy đến frame cần lấy
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break
            
            # Lấy bbox
            key = str(frame_idx)
            if key not in frames_to_process: continue
            
            anno = frames_to_process[key]
            if "bbox" not in anno: continue
            
            x, y, w, h = map(int, anno["bbox"])
            
            # Crop và lưu
            # Mở rộng bbox một chút để lấy ngữ cảnh (padding)
            h_img, w_img = frame.shape[:2]
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_img, x + w + pad)
            y2 = min(h_img, y + h + pad)
            
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                cv2.imwrite(str(obj_dir / f"drone_{frame_idx}.jpg"), crop)
                count += 1
                
        cap.release()

    print(f"\n✅ Dữ liệu đã sẵn sàng tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    prepare_data()