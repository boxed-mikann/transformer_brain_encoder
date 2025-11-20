"""
Feature Extraction Script for Transformer Brain Encoder
========================================================

このスクリプトは、画像から特徴量を事前に抽出し、保存します。
This script pre-extracts image features and saves them for later use.

使用方法 (Usage):
    python extract_features.py \
        --data_dir /path/to/algonauts_data/subj01 \
        --output_dir /path/to/save/features \
        --subj 01 \
        --backbone dinov2_q \
        --batch_size 16

サポートするバックボーン (Supported backbones):
    - dinov2_q: DINOv2 with QKV hooks (推奨/recommended)
    - dinov2: DINOv2 standard
    - clip: CLIP ViT-L-14

出力形式 (Output format):
    saved_feats_dir/
    ├── dinov2_q_last/
    │   └── {subj}/
    │       ├── train.npy    # [N_train, num_patches+1, 768]
    │       └── synt.npy     # [N_test, num_patches+1, 768]
    └── clip_vit_512/
        └── {subj}/
            ├── train.npy    # [N_train, num_patches+1, 512]
            └── synt.npy     # [N_test, num_patches+1, 512]
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import argparse

from models.dino import dino_model_with_hooks, dino_model
from models.clip import clip_model
from utils.utils import NestedTensor


def extract_dino_features_with_hooks(image_dir, output_path, enc_output_layer=-1, batch_size=16, device='cuda'):
    """
    DINOv2 with hooks を使った特徴抽出（メモリ最適化版）
    QKV特徴量を抽出し、datasets/nsd.pyと互換性のある形式で保存
    
    メモリ最適化:
    - numpy.memmapを使用して特徴量を段階的に書き込む
    - バッチ処理後にtorch.cuda.empty_cache()を呼び出してGPUメモリを解放
    
    Args:
        image_dir: 画像ディレクトリパス
        output_path: 出力ファイルパス (.npy)
        enc_output_layer: エンコーダー層の指定 (-1=最終層)
        batch_size: バッチサイズ
        device: デバイス ('cuda' or 'cpu')
    
    Returns:
        all_features: numpy array [num_images, num_patches+1, 768]
    """
    
    print(f"🔧 Extracting DINO features (with hooks) from {image_dir}")
    
    # モデルロード
    model = dino_model_with_hooks(enc_output_layer=enc_output_layer, 
                                  return_interm_layers=False,
                                  return_cls=False)
    model = model.to(device)
    model.eval()
    
    # 画像ファイル一覧
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    num_images = len(img_files)
    print(f"Found {num_images} images")
    
    # 正規化 (datasets/nsd.pyと同じ)
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    patch_size = 14
    
    # 特徴量の形状を決定 (DINOv2: 962パッチ + 768次元)
    num_patches = 962  # 31*31 + 1 CLS token
    feature_dim = 768
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # メモリマップ配列を作成（メモリ最適化）
    memmap_features = np.memmap(output_path + '.tmp', dtype='float32', mode='w+', 
                                shape=(num_images, num_patches, feature_dim))
    
    # バッチ処理で特徴抽出
    current_idx = 0
    for i in tqdm(range(0, len(img_files), batch_size), desc="Extracting features"):
        batch_files = img_files[i:i+batch_size]
        batch_imgs = []
        
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_tensor = normalize(img)
            
            # DINOv2用のパディング (datasets/nsd.pyと同じ処理)
            size_im = (
                img_tensor.shape[0],
                int(np.ceil(img_tensor.shape[1] / patch_size) * patch_size),
                int(np.ceil(img_tensor.shape[2] / patch_size) * patch_size),
            )
            paded = torch.zeros(size_im)
            paded[:, :img_tensor.shape[1], :img_tensor.shape[2]] = img_tensor
            batch_imgs.append(paded)
        
        # バッチを作成
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        # NestedTensor を作成
        mask = torch.ones(batch_tensor.shape[0], 
                         batch_tensor.shape[2], 
                         batch_tensor.shape[3], 
                         dtype=torch.bool, device=device)
        nested_tensor = NestedTensor(batch_tensor, mask)
        
        # 特徴抽出 (models/dino.pyのforward処理を再現)
        with torch.no_grad():
            xs = nested_tensor.tensors
            h, w = int(xs.shape[2]/14), int(xs.shape[3]/14)
            
            # バックボーンから中間層を取得
            xs = model.backbone.get_intermediate_layers(xs)[0]
            
            # QKV特徴を取得 (hook経由)
            feats = model.qkv_feats['qkv_feats']
            
            # Reshape (models/dino.py の58-62行目と同じ)
            nh = 12  # Number of heads
            feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
            q, k, v = feats[0], feats[1], feats[2]
            q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
            
            # xs = q[:,1:,:] としてCLSトークンを除外
            xs_feats = q[:,1:,:]  # [B, 961, 768] (31x31パッチ)
            
            # ただし、datasets/nsd.pyでは img[1:,:] を使用してCLSトークンを除外するため
            # ここではCLSトークンを含めたまま保存する
            # つまり q 全体を保存: [B, 962, 768] (961パッチ + 1 CLS)
            feats_with_cls = q  # [B, 962, 768]
            
            feats_np = feats_with_cls.cpu().numpy()
        
        # メモリマップに直接書き込み（メモリ最適化）
        batch_size_actual = len(batch_files)
        memmap_features[current_idx:current_idx+batch_size_actual] = feats_np
        current_idx += batch_size_actual
        
        # GPUメモリを解放（メモリ最適化）
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # メモリマップをフラッシュ
    memmap_features.flush()
    
    print(f"✅ Feature shape: {memmap_features.shape}")
    
    # 一時ファイルを最終ファイルに移動
    import shutil
    shutil.move(output_path + '.tmp', output_path)
    print(f"✅ Saved to {output_path}")
    
    # 保存された特徴量を返す（互換性のため）
    all_features = np.load(output_path, mmap_mode='r')
    return all_features


def extract_dino_features_simple(image_dir, output_path, enc_output_layer=-1, batch_size=16, device='cuda'):
    """
    通常の DINO を使った特徴抽出（メモリ最適化版）
    
    メモリ最適化:
    - numpy.memmapを使用して特徴量を段階的に書き込む
    - バッチ処理後にtorch.cuda.empty_cache()を呼び出してGPUメモリを解放
    
    Args:
        image_dir: 画像ディレクトリパス
        output_path: 出力ファイルパス (.npy)
        enc_output_layer: エンコーダー層の指定 (-1=最終層)
        batch_size: バッチサイズ
        device: デバイス ('cuda' or 'cpu')
    
    Returns:
        all_features: numpy array [num_images, num_patches+1, 768]
    """
    
    print(f"🔧 Extracting DINO features (simple) from {image_dir}")
    
    # モデルロード
    model = dino_model(enc_output_layer=enc_output_layer, 
                      return_interm_layers=False,
                      return_cls=False)
    model = model.to(device)
    model.eval()
    
    # 画像ファイル一覧
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    num_images = len(img_files)
    print(f"Found {num_images} images")
    
    # 正規化
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    patch_size = 14
    
    # 特徴量の形状を決定 (DINOv2: 962パッチ + 768次元)
    num_patches = 962  # 31*31 + 1 CLS token
    feature_dim = 768
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # メモリマップ配列を作成（メモリ最適化）
    memmap_features = np.memmap(output_path + '.tmp', dtype='float32', mode='w+', 
                                shape=(num_images, num_patches, feature_dim))
    
    # バッチ処理で特徴抽出
    current_idx = 0
    for i in tqdm(range(0, len(img_files), batch_size), desc="Extracting features"):
        batch_files = img_files[i:i+batch_size]
        batch_imgs = []
        
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((224, 224))
            img_tensor = normalize(img)
            
            # パディング
            size_im = (
                img_tensor.shape[0],
                int(np.ceil(img_tensor.shape[1] / patch_size) * patch_size),
                int(np.ceil(img_tensor.shape[2] / patch_size) * patch_size),
            )
            paded = torch.zeros(size_im)
            paded[:, :img_tensor.shape[1], :img_tensor.shape[2]] = img_tensor
            batch_imgs.append(paded)
        
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
            xs = nested_tensor.tensors
            patch_size = 14
            w_p = int(xs.shape[2] / patch_size)
            h_p = int(xs.shape[3] / patch_size)
            
            xs = model.backbone.get_intermediate_layers(xs, n=12)
            xs_layer = xs[enc_output_layer]  # [B, num_patches+1, 768]
            
            # CLSトークンを含む形式で保存
            # datasets/nsd.pyがreshapeに使用する
            feats_np = xs_layer.cpu().numpy()
        
        # メモリマップに直接書き込み（メモリ最適化）
        batch_size_actual = len(batch_files)
        memmap_features[current_idx:current_idx+batch_size_actual] = feats_np
        current_idx += batch_size_actual
        
        # GPUメモリを解放（メモリ最適化）
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # メモリマップをフラッシュ
    memmap_features.flush()
    
    print(f"✅ Feature shape: {memmap_features.shape}")
    
    # 一時ファイルを最終ファイルに移動
    import shutil
    shutil.move(output_path + '.tmp', output_path)
    print(f"✅ Saved to {output_path}")
    
    # 保存された特徴量を返す（互換性のため）
    all_features = np.load(output_path, mmap_mode='r')
    return all_features


def extract_clip_features(image_dir, output_path, enc_output_layer=-1, batch_size=16, device='cuda'):
    """
    CLIP を使った特徴抽出（メモリ最適化版）
    
    メモリ最適化:
    - numpy.memmapを使用して特徴量を段階的に書き込む
    - バッチ処理後にtorch.cuda.empty_cache()を呼び出してGPUメモリを解放
    
    Args:
        image_dir: 画像ディレクトリパス
        output_path: 出力ファイルパス (.npy)
        enc_output_layer: エンコーダー層の指定 (CLIPでは未使用)
        batch_size: バッチサイズ
        device: デバイス ('cuda' or 'cpu')
    
    Returns:
        all_features: numpy array [num_images, num_patches+1, 768]
    """
    
    print(f"🔧 Extracting CLIP features from {image_dir}")
    
    # モデルロード
    model = clip_model(enc_output_layer=enc_output_layer, 
                      return_interm_layers=False,
                      return_cls=False)
    model = model.to(device)
    model.eval()
    
    # 画像ファイル一覧
    img_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    num_images = len(img_files)
    print(f"Found {num_images} images")
    
    # 正規化
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 特徴量の形状を決定 (CLIP: 257パッチ + 768次元)
    num_patches = 257  # 16*16 + 1 CLS token
    feature_dim = 768
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # メモリマップ配列を作成（メモリ最適化）
    memmap_features = np.memmap(output_path + '.tmp', dtype='float32', mode='w+', 
                                shape=(num_images, num_patches, feature_dim))
    
    # バッチ処理で特徴抽出
    current_idx = 0
    for i in tqdm(range(0, len(img_files), batch_size), desc="Extracting features"):
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
        
        # 特徴抽出 (models/clip.pyのforward処理を再現)
        with torch.no_grad():
            # CLIP visual encoder
            cls_token, patch_tokens = model.backbone.visual(batch_tensor)
            
            # Project patch tokens from 1024 → 768
            proj = model.backbone.visual.proj  # shape: (1024, 768)
            patch_tokens_proj = patch_tokens @ proj  # (B, 256, 768)
            
            # CLSトークンも含める形式で保存
            cls_token_reshaped = cls_token.unsqueeze(1)  # (B, 1, 768)
            full_tokens = torch.cat([cls_token_reshaped, patch_tokens_proj], dim=1)  # (B, 257, 768)
            
            feats_np = full_tokens.cpu().numpy()
        
        # メモリマップに直接書き込み（メモリ最適化）
        batch_size_actual = len(batch_files)
        memmap_features[current_idx:current_idx+batch_size_actual] = feats_np
        current_idx += batch_size_actual
        
        # GPUメモリを解放（メモリ最適化）
        if device == 'cuda':
            torch.cuda.empty_cache()
    
    # メモリマップをフラッシュ
    memmap_features.flush()
    
    print(f"✅ Feature shape: {memmap_features.shape}")
    
    # 一時ファイルを最終ファイルに移動
    import shutil
    shutil.move(output_path + '.tmp', output_path)
    print(f"✅ Saved to {output_path}")
    
    # 保存された特徴量を返す（互換性のため）
    all_features = np.load(output_path, mmap_mode='r')
    return all_features


def main():
    parser = argparse.ArgumentParser(
        description='Extract image features for Transformer Brain Encoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to algonauts data directory (e.g., /path/to/subj01)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory root for saving features')
    parser.add_argument('--subj', type=str, default='01',
                       help='Subject ID (e.g., 01, 02, ...)')
    parser.add_argument('--backbone', type=str, default='dinov2_q',
                       choices=['dinov2_q', 'dinov2', 'clip'],
                       help='Backbone model to use for feature extraction')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for feature extraction')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--enc_layer', type=int, default=-1,
                       help='Encoder layer to extract features from (-1 = last layer)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Feature Extraction Configuration")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Subject: {args.subj}")
    print(f"Backbone: {args.backbone}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Encoder layer: {args.enc_layer}")
    print("="*60)
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    # ディレクトリ設定
    train_img_dir = os.path.join(args.data_dir, 'training_split', 'training_images')
    test_img_dir = os.path.join(args.data_dir, 'test_split', 'test_images')
    
    if not os.path.exists(train_img_dir):
        print(f"❌ Training image directory not found: {train_img_dir}")
        return
    
    if not os.path.exists(test_img_dir):
        print(f"⚠️  Test image directory not found: {test_img_dir}")
        print("   Skipping test feature extraction")
        test_img_dir = None
    
    # 出力パス設定
    if args.backbone == 'dinov2_q':
        output_subdir = 'dinov2_q_last'
    elif args.backbone == 'dinov2':
        output_subdir = 'dinov2_last'
    elif args.backbone == 'clip':
        output_subdir = 'clip_vit_512'
    
    output_subject_dir = os.path.join(args.output_dir, output_subdir, args.subj)
    train_output_path = os.path.join(output_subject_dir, 'train.npy')
    test_output_path = os.path.join(output_subject_dir, 'synt.npy')
    
    # 特徴抽出関数の選択
    if args.backbone == 'dinov2_q':
        extract_fn = extract_dino_features_with_hooks
    elif args.backbone == 'dinov2':
        extract_fn = extract_dino_features_simple
    elif args.backbone == 'clip':
        extract_fn = extract_clip_features
    
    # 訓練データの特徴抽出
    print("\n" + "="*60)
    print("Extracting TRAINING features")
    print("="*60)
    train_features = extract_fn(
        train_img_dir, 
        train_output_path,
        enc_output_layer=args.enc_layer,
        batch_size=args.batch_size,
        device=args.device
    )
    
    # テストデータの特徴抽出
    if test_img_dir:
        print("\n" + "="*60)
        print("Extracting TEST features")
        print("="*60)
        test_features = extract_fn(
            test_img_dir,
            test_output_path,
            enc_output_layer=args.enc_layer,
            batch_size=args.batch_size,
            device=args.device
        )
    
    print("\n" + "="*60)
    print("✅ Feature extraction completed successfully!")
    print("="*60)
    print(f"\nSaved features to:")
    print(f"  Training: {train_output_path}")
    if test_img_dir:
        print(f"  Test:     {test_output_path}")
    
    print(f"\nTo use these features in training, run:")
    print(f"  python main.py \\")
    print(f"    --subj {int(args.subj)} \\")
    if args.backbone == 'dinov2_q':
        print(f"    --saved_feats dinov2q \\")
    elif args.backbone == 'clip':
        print(f"    --saved_feats clip \\")
    print(f"    --saved_feats_dir {args.output_dir} \\")
    print(f"    --encoder_arch transformer \\")
    print(f"    --readout_res rois_all")
    print("="*60)


if __name__ == '__main__':
    main()
