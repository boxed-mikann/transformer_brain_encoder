# extract_features.py（リポジトリ準拠版）
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse
import sys

# リポジトリのモデルをインポート
from models.dino import dino_model_with_hooks
from models.clip import clip_model
from utils.utils import NestedTensor

def extract_features(image_dir, output_dir, model_type='dino', batch_size=16, device='cuda'):
    """
    リポジトリのコードに準拠した特徴抽出
    datasets/nsd.py の構造に合わせた出力フォーマット
    """
    
    print(f"🔧 Extracting {model_type.upper()} features from {image_dir}")
    
    # モデルロード（リポジトリのコードに準拠）
    if model_type == 'dino':
        model = dino_model_with_hooks(enc_output_layer=-1, return_interm_layers=False)
    elif model_type == 'clip':
        model = clip_model(enc_output_layer=-1, return_interm_layers=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
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
        
        # NestedTensor を作成（datasets/nsd.py で使用されるのと同じ形式）
        mask = torch.ones(batch_tensor.shape[0], 
                         batch_tensor.shape[2], 
                         batch_tensor.shape[3], 
                         dtype=torch.bool, device=device)
        nested_tensor = NestedTensor(batch_tensor, mask)
        
        # 特徴抽出
        with torch.no_grad():
            outputs = model(nested_tensor)
        
        # datasets/nsd.py の62行目に合わせた形式に変換
        # img = torch.reshape(img, (962, 768))  ← これに対応するサイズ
        
        feats = outputs['layer_top'].tensors  # (batch, 768, h, w)
        
        # 形状を調整（datasets/nsd.py での reshape に対応）
        batch_size_actual = feats.shape[0]
        feats_flat = feats.reshape(batch_size_actual, feats.shape[1], -1)  # (batch, 768, h*w)
        feats_flat = feats_flat.permute(0, 2, 1)  # (batch, h*w, 768)
        feats_flat = feats_flat.cpu().numpy()
        
        all_features.append(feats_flat)
    
    # すべての特徴を結合
    all_features = np.concatenate(all_features, axis=0)
    print(f"Feature shape: {all_features.shape}")
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存（datasets/nsd.py の21-22行目に合わせた命名）
    output_path = os.path.join(output_dir, 'train.npy')
    np.save(output_path, all_features)
    print(f"✅ Saved to {output_path}")
    
    return all_features


def main():
    parser = argparse.ArgumentParser(description='Extract image features (compatible with transformer_brain_encoder)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to algonauts data (e.g., /path/to/subj01)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory root (e.g., /path/to/image_features)')
    parser.add_argument('--subj', type=str, default='01',
                       help='Subject ID (e.g., 01, 02, ...)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # 訓練画像ディレクトリ
    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    
    if not os.path.exists(train_img_dir):
        print(f"❌ Image directory not found: {train_img_dir}")
        return
    
    # DINO 特徴抽出
    # datasets/nsd.py の21行目: dino_feat_dir = args.saved_feats_dir + '/dinov2_q_last/'+ args.subj
    dino_output_dir = os.path.join(args.output_dir, 'dinov2_q_last', args.subj)
    extract_features(train_img_dir, dino_output_dir, 
                    model_type='dino',
                    batch_size=args.batch_size,
                    device=args.device)
    
    # CLIP 特徴抽出
    # datasets/nsd.py の22行目: clip_feat_dir = args.saved_feats_dir + '/clip_vit_512/'+ args.subj
    clip_output_dir = os.path.join(args.output_dir, 'clip_vit_512', args.subj)
    extract_features(train_img_dir, clip_output_dir,
                    model_type='clip',
                    batch_size=args.batch_size,
                    device=args.device)
    
    print("✅ Feature extraction complete!")


if __name__ == '__main__':
    main()
