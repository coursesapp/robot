import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from skimage.feature import hog
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# --- 1. Configuration ---
CONFIG = {
    "image_size": (224, 224),
    "grid_levels": [1, 2, 3],  # 1x1, 2x2, 3x3
    "hog_orientations": 8,
    "hog_pixels_per_cell": (16, 16),
    "hog_cells_per_block": (1, 1),
    "batch_size": 32,
    "epochs": 30,
    "lr_confidence": 0.0003,  # Learning rate for confidence head
    "lr_box": 0.0003,         # Learning rate for box head
    "hidden_layers": [2048, 1024 , 500],
    "voc_root": "/kaggle/input/voc0712/VOC_dataset/VOCdevkit/VOC2012",
    "model_path": "multilevel_hog_model.pth",
    "features_cache_dir": "cache_multilevel_hog",
    "lambda_obj": 2.0,
    "lambda_noobj": 1.0,
    "lambda_box": 2.0,
    "assignment_margin": 0.1  # 10% margin for grid assignment (0.0 = strict, 0.2 = very relaxed)
}

# حساب إجمالي عدد الخلايا (1*1 + 2*2 + 3*3 = 14)
TOTAL_CELLS = sum([l**2 for l in CONFIG["grid_levels"]])
os.makedirs(CONFIG["features_cache_dir"], exist_ok=True)

# --- 2. Multi-Level Target Parser ---
def parse_voc_multilevel(xml_path, config):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        
        target = np.zeros((TOTAL_CELLS, 5), dtype=np.float32)
        
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin, ymin = float(bbox.find("xmin").text)/width, float(bbox.find("ymin").text)/height
            xmax, ymax = float(bbox.find("xmax").text)/width, float(bbox.find("ymax").text)/height
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            bw, bh = (xmax - xmin), (ymax - ymin)

            assigned = False
            # نبدأ من المستوى الأصغر (3x3) لتحقيق شرط الأصغر أولاً
            for level in reversed(config["grid_levels"]):
                # تحديد نقطة البداية لهذا المستوى في المصفوفة
                level_start_idx = sum([l**2 for l in config["grid_levels"] if l < level])
                grid_unit = 1.0 / level
                
                for row in range(level):
                    for col in range(level):
                        # حدود الخلية الحالية
                        g_xmin, g_ymin = col * grid_unit, row * grid_unit
                        g_xmax, g_ymax = (col + 1) * grid_unit, (row + 1) * grid_unit
                        
                        # شرط: الكائن داخل الخلية مع هامش مسموح (margin allowance)
                        margin = config["assignment_margin"]
                        if (xmin >= g_xmin - margin and ymin >= g_ymin - margin and 
                            xmax <= g_xmax + margin and ymax <= g_ymax + margin):
                            idx = level_start_idx + (row * level + col)
                            if target[idx, 0] == 0:
                                target[idx, 0] = 1.0
                                target[idx, 1:5] = [cx, cy, bw, bh]
                                assigned = True
                                break
                    if assigned: break
                if assigned: break
        return target.flatten()
    except:
        return np.zeros(TOTAL_CELLS * 5, dtype=np.float32)

# --- 3. Dataset Class ---
class MultiLevelHOGDataset(Dataset):
    def __init__(self, file_names, img_dir, ann_dir, config):
        self.file_names, self.img_dir, self.ann_dir, self.config = file_names, img_dir, ann_dir, config

    def __len__(self): return len(self.file_names)

    def extract_grid_hog(self, image):
        all_hogs = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for level in self.config["grid_levels"]:
            h, w = gray.shape
            sh, sw = h // level, w // level
            for r in range(level):
                for c in range(level):
                    roi = gray[r*sh:(r+1)*sh, c*sw:(c+1)*sw]
                    roi = cv2.resize(roi, (64, 64))
                    feat = hog(roi, orientations=self.config["hog_orientations"],
                               pixels_per_cell=self.config["hog_pixels_per_cell"],
                               cells_per_block=self.config["hog_cells_per_block"], feature_vector=True)
                    all_hogs.append(feat)
        return np.concatenate(all_hogs).astype(np.float32)

    def __getitem__(self, idx):
        f_name = self.file_names[idx]
        cache_path = os.path.join(self.config["features_cache_dir"], f_name + ".npy")
        
        if os.path.exists(cache_path):
            features = np.load(cache_path)
        else:
            img_path = os.path.join(self.img_dir, f_name + ".jpg")
            img = cv2.imread(img_path)
            if img is None: return torch.zeros(1), torch.zeros(TOTAL_CELLS * 5)
            features = self.extract_grid_hog(cv2.resize(img, self.config["image_size"]))
            np.save(cache_path, features)
            
        target = parse_voc_multilevel(os.path.join(self.ann_dir, f_name + ".xml"), self.config)
        return torch.from_numpy(features), torch.from_numpy(target)

# --- 4. Model & Loss ---
class MultiLevelDetectorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, total_cells):
        super().__init__()
        layers, last_dim = [], input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last_dim, h), nn.LeakyReLU(0.1), nn.BatchNorm1d(h), nn.Dropout(0.2)])
            last_dim = h
        self.backbone = nn.Sequential(*layers)
        
        # Multi-Head Architecture
        self.confidence_head = nn.Linear(last_dim, total_cells)  # 14 × 1 (confidence)
        self.box_head = nn.Linear(last_dim, total_cells * 4)     # 14 × 4 (cx, cy, w, h)

    def forward(self, x):
        features = self.backbone(x)
        conf = self.confidence_head(features)  # [batch, 14]
        box = self.box_head(features)          # [batch, 14*4]
        return conf, box

def compute_losses(conf_pred, box_pred, target):
    """
    Separate loss computation for multi-head architecture
    Args:
        conf_pred: [batch, 14] - confidence predictions
        box_pred: [batch, 14*4] - box predictions
        target: [batch, 14*5] - ground truth
    Returns:
        loss_confidence: weighted confidence loss
        loss_box: weighted box loss
    """
    target = target.view(-1, TOTAL_CELLS, 5)
    conf_pred = conf_pred.view(-1, TOTAL_CELLS)           # [batch, 14]
    box_pred = box_pred.view(-1, TOTAL_CELLS, 4)          # [batch, 14, 4]
    
    conf_target = target[:, :, 0]      # [batch, 14]
    box_target = target[:, :, 1:]      # [batch, 14, 4]
    
    obj_mask = conf_target == 1
    noobj_mask = conf_target == 0
    
    # Confidence Loss للخلايا اللي فيها كائنات
    loss_conf_obj = nn.functional.binary_cross_entropy_with_logits(
        conf_pred[obj_mask], conf_target[obj_mask], reduction='sum'
    ) if obj_mask.sum() > 0 else torch.tensor(0.0, device=conf_pred.device)
    
    # Confidence Loss للخلايا الفارغة
    loss_conf_noobj = nn.functional.binary_cross_entropy_with_logits(
        conf_pred[noobj_mask], conf_target[noobj_mask], reduction='sum'
    ) if noobj_mask.sum() > 0 else torch.tensor(0.0, device=conf_pred.device)
    
    # Box Loss (MSE) - فقط للخلايا اللي فيها كائنات
    loss_box_raw = nn.functional.mse_loss(
        torch.sigmoid(box_pred[obj_mask]), 
        box_target[obj_mask], 
        reduction='sum'
    ) if obj_mask.sum() > 0 else torch.tensor(0.0, device=box_pred.device)
    
    # Normalize by batch size
    batch_size = conf_pred.size(0)
    
    # Apply weights and normalize
    loss_confidence = (CONFIG["lambda_obj"] * loss_conf_obj + 
                       CONFIG["lambda_noobj"] * loss_conf_noobj) / batch_size
    
    loss_box = (CONFIG["lambda_box"] * loss_box_raw) / batch_size
    
    return loss_confidence, loss_box

# --- 5. Inference Function ---
def predict_and_show(model, img_path, config, threshold=0.3):
    model.eval()
    device = next(model.parameters()).device
    orig = cv2.imread(img_path)
    if orig is None: return
    h_orig, w_orig = orig.shape[:2]
    
    helper = MultiLevelHOGDataset([], "", "", config)
    feat = helper.extract_grid_hog(cv2.resize(orig, config["image_size"]))
    feat_t = torch.from_numpy(feat).unsqueeze(0).to(device)
    
    with torch.no_grad():
        conf_pred, box_pred = model(feat_t)
        conf_pred = torch.sigmoid(conf_pred).view(TOTAL_CELLS).cpu().numpy()
        box_pred = torch.sigmoid(box_pred).view(TOTAL_CELLS, 4).cpu().numpy()

    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    
    offsets = [0, 1, 5] # لتمييز المستويات 1x1, 2x2, 3x3
    colors = {0: 'red', 1: 'blue', 5: 'lime'}
    
    for i in range(TOTAL_CELLS):
        if conf_pred[i] > threshold:
            level_color = 'white'
            for off in offsets:
                if i >= off: level_color = colors[off]
            
            cx, cy, bw, bh = box_pred[i]
            x, y = (cx - bw/2) * w_orig, (cy - bh/2) * h_orig
            w, h = bw * w_orig, bh * h_orig
            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=level_color, facecolor='none'))
            ax.text(x, y-5, f"{conf_pred[i]:.2f}", color='white', fontsize=8, bbox=dict(facecolor=level_color, alpha=0.5))
    plt.axis('off')
    plt.show()

# --- 7. Main Loop ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_dir = os.path.join(CONFIG["voc_root"], "JPEGImages")
    ann_dir = os.path.join(CONFIG["voc_root"], "Annotations")
    all_files = [f[:-4] for f in os.listdir(ann_dir) if f.endswith(".xml")][:3000]
    
    dataset = MultiLevelHOGDataset(all_files, img_dir, ann_dir, CONFIG)
    train_ds, val_ds = random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    
    sample_feat, _ = dataset[0]
    model = MultiLevelDetectorMLP(sample_feat.shape[0], CONFIG["hidden_layers"], TOTAL_CELLS).to(device)
    
    # Separate optimizers for each head
    optimizer_conf = optim.AdamW(
        list(model.backbone.parameters()) + list(model.confidence_head.parameters()),
        lr=CONFIG["lr_confidence"]
    )
    optimizer_box = optim.AdamW(
        list(model.backbone.parameters()) + list(model.box_head.parameters()),
        lr=CONFIG["lr_box"]
    )
    
    best_val_loss = float('inf')
    best_model_path = "best_" + CONFIG["model_path"]
    
    print(f"Training on {device}... Total Cells: {TOTAL_CELLS}")
    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    for epoch in range(CONFIG["epochs"]):
        # Training phase
        model.train()
        train_conf_loss_sum = 0
        train_box_loss_sum = 0
        for f, t in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            f, t = f.to(device), t.to(device)
            
            # Compute separate losses
            conf_pred, box_pred = model(f)
            loss_conf, loss_box = compute_losses(conf_pred, box_pred, t)
            
            # Backward pass for confidence head
            optimizer_conf.zero_grad()
            loss_conf.backward(retain_graph=True)
            optimizer_conf.step()
            
            # Backward pass for box head
            optimizer_box.zero_grad()
            loss_box.backward()
            optimizer_box.step()
            
            train_conf_loss_sum += loss_conf.item()
            train_box_loss_sum += loss_box.item()
        
        avg_train_conf_loss = train_conf_loss_sum / len(train_loader)
        avg_train_box_loss = train_box_loss_sum / len(train_loader)
        
        # Validation phase
        model.eval()
        val_conf_loss_sum = 0
        val_box_loss_sum = 0
        with torch.no_grad():
            for f, t in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
                f, t = f.to(device), t.to(device)
                conf_pred, box_pred = model(f)
                loss_conf, loss_box = compute_losses(conf_pred, box_pred, t)
                val_conf_loss_sum += loss_conf.item()
                val_box_loss_sum += loss_box.item()
        
        avg_val_conf_loss = val_conf_loss_sum / len(val_loader)
        avg_val_box_loss = val_box_loss_sum / len(val_loader)
        avg_val_loss = avg_val_conf_loss + avg_val_box_loss  # Total for model saving
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"  Train - Conf: {avg_train_conf_loss:.4f}, Box: {avg_train_box_loss:.4f}, Total: {avg_train_conf_loss + avg_train_box_loss:.4f}")
        print(f"  Val   - Conf: {avg_val_conf_loss:.4f}, Box: {avg_val_box_loss:.4f}, Total: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved! (Val Loss: {best_val_loss:.4f})")
        
        # Display 10 images every 10 epochs
        if (epoch+1) % 10 == 0:
            print(f"\n--- Showing 10 sample predictions at epoch {epoch+1} ---")
            for i in range(10):
                idx = random.randint(0, len(val_ds)-1)
                fname = all_files[val_ds.indices[idx]]
                predict_and_show(model, os.path.join(img_dir, fname + ".jpg"), CONFIG)
            print("---\n")
    
    # Save final model
    torch.save(model.state_dict(), CONFIG["model_path"])
    print(f"\nFinal model saved to {CONFIG['model_path']}")
    
    # Load best model and display 20 images
    print(f"\n=== Loading best model (Val Loss: {best_val_loss:.4f}) ===")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    print("Showing 20 final predictions with best model:\n")
    for i in range(20):
        idx = random.randint(0, len(val_ds)-1)
        fname = all_files[val_ds.indices[idx]]
        predict_and_show(model, os.path.join(img_dir, fname + ".jpg"), CONFIG)