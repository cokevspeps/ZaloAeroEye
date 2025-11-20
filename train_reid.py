import torch
import torch.nn as nn
import torch.optim as optim
import timm
import os
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
DATA_DIR = Path("reid_dataset")
SAVE_PATH = "reid_finetuned.pth"
EPOCHS = 15  # Số vòng train (tăng lên nếu cần)
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. DATASET (CHỌN BỘ BA: ANCHOR - POSITIVE - NEGATIVE) ---
class TripletReIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [d.name for d in root_dir.iterdir() if d.is_dir()]
        self.class_to_images = {}
        
        for cls in self.classes:
            imgs = list((root_dir / cls).glob("*.jpg"))
            # Phân loại: Đâu là ảnh Drone (Anchor), đâu là ảnh Ref (Positive)
            refs = [x for x in imgs if "ref" in x.name]
            drones = [x for x in imgs if "drone" in x.name]
            
            if refs and drones:
                self.class_to_images[cls] = {"refs": refs, "drones": drones}

        # Chỉ giữ lại class nào có đủ cả 2 loại ảnh
        self.classes = list(self.class_to_images.keys())

    def __len__(self):
        # Giả lập độ dài: mỗi epoch lặp qua mỗi class 10 lần
        return len(self.classes) * 10

    def __getitem__(self, idx):
        # 1. Chọn ngẫu nhiên 1 class (VD: Backpack_0)
        pos_class = random.choice(self.classes)
        
        # 2. ANCHOR: Chọn 1 ảnh từ Drone
        anchor_path = random.choice(self.class_to_images[pos_class]["drones"])
        
        # 3. POSITIVE: Chọn 1 ảnh từ Ref (cùng class)
        positive_path = random.choice(self.class_to_images[pos_class]["refs"])
        
        # 4. NEGATIVE: Chọn 1 class KHÁC (VD: Jacket_1)
        neg_class = random.choice([c for c in self.classes if c != pos_class])
        # Negative lấy từ Ref của class kia
        negative_path = random.choice(self.class_to_images[neg_class]["refs"])
        
        # Load và Transform
        anchor = self.transform(Image.open(anchor_path).convert("RGB"))
        positive = self.transform(Image.open(positive_path).convert("RGB"))
        negative = self.transform(Image.open(negative_path).convert("RGB"))
        
        return anchor, positive, negative

# --- 2. HÀM TRAIN ---
def train():
    print(f"Bắt đầu Fine-tune Re-ID trên {DEVICE}...")
    
    # Tiền xử lý ảnh chuẩn của ResNet
    transform = transforms.Compose([
        transforms.Resize((256, 128)), # Kích thước chuẩn ReID (Cao x Rộng)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TripletReIDDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Load Model ResNet50
    model = timm.create_model('resnet50', pretrained=True, num_classes=0).to(DEVICE)
    model.train()
    
    # Loss & Optimizer
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for anchor, positive, negative in pbar:
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward (Chạy 3 lần)
            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)
            
            # Tính Loss: (A gần P) và (A xa N)
            loss = criterion(emb_a, emb_p, emb_n)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / len(dataloader)})
            
    # Lưu model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"Đã train xong! Model lưu tại: {SAVE_PATH}")

if __name__ == "__main__":
    train()