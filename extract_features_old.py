# extract_features_correct.py
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse
import sys

from models.dino import dino_model_with_hooks, dino_model
from models.clip import clip_model
from utils.utils import NestedTensor

def extract_dino_features_with_hooks(image_dir, output_dir, enc_output_layer=-1, batch_size=16, device='cuda'):
    """
    DINOv2 with hooks を使った特徴抽出
    オンザフライモードと同じ処理フロー
    """
    
    print(f"🔧 Extracting DINO features (with hooks) from {image_dir}")
    
    # モデルロード（オンザフライと同じ）
    model = dino_model_with_hooks(enc_output_layer=enc_output_layer, 
                                  return_interm_layers=False)
    model = model.to(device)
    model.eval()
    
    # 画像ファイル一覧
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"Found {len(img_files)} images")
    
    # 正規化
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    all_features = []
    patch_size = 14
    
    # バッチ処理で特徴抽出
    for i in tqdm(range(0, len(img_files), batch_size)):
        batch_files = img_files[i:i+batch_size]
        batch_imgs = []
        
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_tensor = normalize(img)
            batch_imgs.append(img_tensor)
        
        # バッチを作成
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        # NestedTensor を作成
        mask = torch.ones(batch_tensor.shape[0], 
                         batch_tensor.shape[2], 
                         batch_tensor.shape[3], 
                         dtype=torch.bool, device=device)
        nested_tensor = NestedTensor(batch_tensor, mask)
        
        # 特徴抽出（models/dino.py と同じ処理）
        with torch.no_grad():
            # バックボーンから中間層を取得
            xs = model.backbone.get_intermediate_layers(nested_tensor.tensors)[0]
            
            # QKV特徴を取得（hook経由）
            feats = model.qkv_feats['qkv_feats']
            
            # Reshape（models/dino.py の66-71行目と同じ）
            nh = 12  # Number of heads
            feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
            q, k, v = feats[0], feats[1], feats[2]
            q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
            
            # CLS トークンを削除（models/dino.py の73行目）
            xs_feats = q[:,1:,:]  # (batch, h*w, 768)
            
            # 形状を (batch, h*w, 768) → (batch, h*w, 768) のままで保存
            # これが datasets/nsd.py の reshape に対応する
            feats_flat = xs_feats.cpu().numpy()
        
        all_features.append(feats_flat)
    
    # すべての特徴を結合
    all_features = np.concatenate(all_features, axis=0)
    print(f"Feature shape: {all_features.shape}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存
    output_path = os.path.join(output_dir, 'train.npy')
    np.save(output_path, all_features)
    print(f"✅ Saved to {output_path}")
    
    return all_features


def extract_dino_features_simple(image_dir, output_dir, enc_output_layer=-1, batch_size=16, device='cuda'):
    """
    通常の DINO を使った特徴抽出（簡潔版）
    """
    
    print(f"🔧 Extracting DINO features (simple) from {image_dir}")
    
    # モデルロード
    model = dino_model(enc_output_layer=enc_output_layer, 
                      return_interm_layers=False)
    model = model.to(device)
    model.eval()
    
    # 画像ファイル一覧
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    print(f"Found {len(img_files)} images")
    
    # 正規化
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    all_features = []
    
    # バッチ処理で特徴抽出
    for i in tqdm(range(0, len(img_files), batch_size)):
        batch_files = img_files[i:i+batch_size]
        batch_imgs = []
        
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_tensor = normalize(img)
            batch_imgs.append(img_tensor)
        
        # バッチを作成
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        # NestedTensor を作成
        mask = torch.ones(batch_tensor.shape[0], 
                         batch_tensor.shape[2], 
                         batch_tensor.shape[3], 
                         dtype=torch.bool, device=device)
        nested_tensor = NestedTensor(batch_tensor, mask)
        
        # 特徴抽出
        with torch.no_grad():
            outputs = model(nested_tensor)
        
        # models/dino.py の115-120行目で reshape されている
        # 出力は {'layer_top': NestedTensor} の形
        feats = outputs['layer_top'].tensors  # (batch, 768, h, w)
        
        # Flat化：(batch, 768, h, w) → (batch, h*w, 768)
        batch_size_actual = feats.shape[0]
        h, w = feats.shape[2], feats.shape[3]
        feats_flat = feats.reshape(batch_size_actual, feats.shape[1], -1)  # (batch, 768, h*w)
        feats_flat = feats_flat.permute(0, 2, 1)  # (batch, h*w, 768)
        feats_flat = feats_flat.cpu().numpy()
        
        all_features.append(feats_flat)
    
    # すべての特徴を結合
    all_features = np.concatenate(all_features, axis=0)
    print(f"Feature shape: {all_features.shape}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存
    output_path = os.path.join(output_dir, 'train.npy')
    np.save(output_path, all_features)
    print(f"✅ Saved to {output_path}")
    
    return all_features


def main():
    parser = argparse.ArgumentParser(description='Extract image features (correct version)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to algonauts data (e.g., /path/to/subj01)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory root')
    parser.add_argument('--subj', type=str, default='01',
                       help='Subject ID')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--enc_layer', type=int, default=-1,
                       help='Encoder layer (-1 = last layer)')
    
    args = parser.parse_args()
    
    # 訓練画像ディレクトリ
    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    
    if not os.path.exists(train_img_dir):
        print(f"❌ Image directory not found: {train_img_dir}")
        return
    
    # DINO 特徴抽出（with hooks - より正確）
    dino_output_dir = os.path.join(args.output_dir, 'dinov2_q_last', args.subj)
    extract_dino_features_with_hooks(train_img_dir, dino_output_dir,
                                     enc_output_layer=args.enc_layer,
                                     batch_size=args.batch_size,
                                     device=args.device)
    
    print("✅ Feature extraction complete!")


if __name__ == '__main__':
    main()
