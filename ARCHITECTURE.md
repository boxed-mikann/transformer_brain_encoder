# Transformer Brain Encoder - Architecture Documentation

## 概要 (Overview)

このプロジェクトは、視覚刺激に対する脳のfMRI反応を予測する深層学習モデルです。画像特徴量を脳活動パターンにマッピングするためにTransformerアーキテクチャを使用しています。

This project implements a deep learning model that predicts brain fMRI responses to visual stimuli. It uses a Transformer architecture to map image features to brain activity patterns.

## システムアーキテクチャ (System Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Training Flow                        │
│                          (main.py)                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │   Data Loading (datasets/nsd.py)      │
        │   - algonauts_dataset class            │
        │   - fetch_dataloaders()                │
        └───────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
                ▼                               ▼
    ┌────────────────────┐         ┌────────────────────────┐
    │  Feature Mode:     │         │  Pre-extracted Mode:   │
    │  Online Extraction │         │  --saved_feats         │
    │  (from images)     │         │  (from .npy files)     │
    └────────────────────┘         └────────────────────────┘
                │                               │
                │                               │
                ▼                               ▼
    ┌────────────────────┐         ┌────────────────────────┐
    │  Backbone Models   │         │  Load Features from    │
    │  (models/)         │         │  saved_feats_dir/      │
    │  - dino.py         │         │  - dinov2_q_last/      │
    │  - clip.py         │         │  - clip_vit_512/       │
    │  - resnet.py       │         │                        │
    └────────────────────┘         └────────────────────────┘
                │                               │
                └───────────────┬───────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │   Brain Encoder                       │
        │   (models/brain_encoder.py)           │
        │   - Transformer decoder                │
        │   - Linear readout                     │
        └───────────────────────────────────────┘
                                │
                                ▼
        ┌───────────────────────────────────────┐
        │   Training & Evaluation               │
        │   (engine.py)                          │
        │   - train_one_epoch()                  │
        │   - evaluate()                         │
        │   - test()                             │
        └───────────────────────────────────────┘
```

## 主要コンポーネント (Main Components)

### 1. main.py
**役割**: メインエントリーポイント、訓練ループの制御

**主要関数**:
- `get_args_parser()`: コマンドライン引数の定義
- `main(rank, world_size, args)`: メイン訓練ループ
- `SetCriterion`: 損失関数の定義

**データフロー**:
```
args = parse_args()
  ↓
データローダーの作成 (fetch_dataloaders)
  ↓
モデルの構築 (brain_encoder)
  ↓
訓練ループ (train_one_epoch, evaluate)
```

### 2. datasets/nsd.py
**役割**: データセットの読み込みと前処理

**主要クラス・関数**:
- `algonauts_dataset`: Dataset実装
  - `__init__`: データセット初期化
    - `saved_feats`がNoneの場合: 画像パスを保存
    - `saved_feats`が指定されている場合: 事前抽出された特徴量を読み込み
  - `__getitem__`: サンプル取得
    - オンザフライモード: 画像を読み込み、transforms適用
    - 事前抽出モード: .npyファイルから特徴量を読み込み
- `fetch_dataloaders()`: DataLoaderの作成

**データ形式**:
```python
# オンザフライモード
img: Tensor [C, H, W]  # 正規化済み画像
fmri_data: dict {
    'lh_f': [lh_fmri],  # 左半球fMRI
    'rh_f': [rh_fmri]   # 右半球fMRI
}

# 事前抽出モード
img: Tensor [962, 768] または [31, 31, C]  # 特徴量
fmri_data: dict (同上)
```

### 3. models/backbone.py
**役割**: バックボーンモデルの構築

**主要関数**:
- `build_backbone(args)`: 引数に基づいてバックボーンを選択
  - 'resnet': ResNetモデル
  - 'dinov2': DINOv2モデル (通常版)
  - 'dinov2_q': DINOv2モデル (QKV特徴量版)
  - 'clip': CLIPモデル

**出力形式**:
- Joinerを通して位置エンコーディングと組み合わせ
- NestedTensor形式で特徴量とマスクを返す

### 4. models/dino.py
**役割**: DINOv2バックボーンの実装

**クラス**:

#### `dino_model_with_hooks`
QKV特徴量を抽出するバージョン (推奨)

**処理フロー**:
```python
入力画像 [B, 3, 224, 224]
  ↓
DINOv2 backbone (get_intermediate_layers)
  ↓
Hook経由でQKV特徴量取得
  ↓
reshape & permute
  ↓
Q特徴量 [B, 962, 768]
  ↓
CLSトークン除去 → [B, 961, 768]
  ↓
reshape → [B, 768, 31, 31]
  ↓
NestedTensor(特徴量, マスク)
```

**重要なポイント**:
- `enc_output_layer`: どの層から特徴量を取得するか (-1 = 最終層)
- フック登録: `backbone.blocks[layer].attn.qkv.register_forward_hook()`
- パッチ数: 224/14 = 16 → 16x16 = 256パッチ (+1 CLSトークン = 257)
- 出力形状: (31, 31) = 961パッチ (パディング済み)

#### `dino_model`
通常の特徴量を抽出するシンプル版

**処理フロー**:
```python
入力画像 [B, 3, 224, 224]
  ↓
DINOv2 backbone (get_intermediate_layers)
  ↓
指定層の出力取得
  ↓
reshape → [B, 768, w_p, h_p]
  ↓
NestedTensor(特徴量, マスク)
```

### 5. models/clip.py
**役割**: CLIPバックボーンの実装

**処理フロー**:
```python
入力画像 [B, 3, 224, 224]
  ↓
CLIP visual encoder
  ↓
cls_token [B, 768], patch_tokens [B, 256, 1024]
  ↓
projection: patch_tokens @ proj → [B, 256, 768]
  ↓
reshape → [B, 768, 16, 16]
  ↓
NestedTensor(特徴量, マスク)
```

### 6. models/brain_encoder.py
**役割**: 画像特徴量から脳活動への変換

**処理フロー**:
```python
NestedTensor(特徴量, マスク)
  ↓
[encoder_arch = 'transformer']
  ↓
Transformer Decoder
  - Query embeddings: [num_queries, hidden_dim]
  - Cross-attention: queries × image features
  ↓
output_tokens [B, num_queries, hidden_dim]
  ↓
Linear projection (lh_embed, rh_embed)
  ↓
lh_f_pred, rh_f_pred
```

**重要なパラメータ**:
- `encoder_arch`: 'transformer', 'linear', 'custom_transformer', 'spatial_feature'
- `readout_res`: 'voxels', 'rois_all', 'streams_inc', 'hemis'
- `num_queries`: ROI数に基づくクエリ数

### 7. engine.py
**役割**: 訓練・評価ループの実装

**主要関数**:
- `train_one_epoch()`: 1エポックの訓練
  ```python
  for imgs, targets in data_loader:
      imgs = nested_tensor_from_tensor_list(imgs)
      outputs = model(imgs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
  ```

- `evaluate()`: 検証データでの評価
  ```python
  for samples, targets in data_loader:
      outputs = model(samples)
      # 予測値と真値を収集
      lh_f_pred, rh_f_pred = outputs[...]
      lh_fmri_val, rh_fmri_val = targets[...]
  return 予測値, 真値, 損失
  ```

- `test()`: テストデータでの予測
  ```python
  for samples in data_loader:
      outputs = model(samples)
      lh_f_pred, rh_f_pred = outputs[...]
  return 予測値
  ```

## 特徴量の事前抽出 (Feature Pre-extraction)

### 目的
訓練時に毎回バックボーンを通す代わりに、特徴量を事前に抽出して保存することで:
- 訓練時間を短縮
- メモリ使用量を削減
- 異なるエンコーダーで同じ特徴量を再利用

### 使用方法

#### 1. 特徴量の抽出
```bash
python extract_features.py \
    --data_dir /path/to/algonauts_data/subj01 \
    --output_dir /path/to/save/features \
    --subj 01 \
    --backbone dinov2_q  # または dinov2, clip
```

#### 2. 事前抽出された特徴量での訓練
```bash
python main.py \
    --subj 1 \
    --saved_feats dinov2q \
    --saved_feats_dir /path/to/save/features \
    --encoder_arch transformer \
    --readout_res rois_all
```

### ディレクトリ構造
```
saved_feats_dir/
├── dinov2_q_last/
│   └── 01/
│       ├── train.npy    # [N_train, 962, 768]
│       └── synt.npy     # [N_test, 962, 768]
└── clip_vit_512/
    └── 01/
        ├── train.npy    # [N_train, 257, 512]
        └── synt.npy     # [N_test, 257, 512]
```

### 特徴量の形状
```python
# DINOv2_q (with hooks)
# 形状: [num_images, 962, 768]
# 962 = 31*31 + 1 (パッチ数 + CLSトークン)
# 768 = 特徴次元

# CLIP
# 形状: [num_images, 257, 512]
# 257 = 16*16 + 1 (パッチ数 + CLSトークン)
# 512 = CLIP特徴次元
```

## データセットでの特徴量の扱い (Feature Handling in Dataset)

### datasets/nsd.py の処理

**saved_feats=Noneの場合 (オンザフライ)**:
```python
# __getitem__で画像を読み込み
img = Image.open(img_path).convert('RGB')
img = transform(img)  # 正規化
# パディング処理 (DINOv2の場合)
if 'dinov2' in backbone_arch:
    paded = torch.zeros(size_im)
    paded[:, :img.shape[1], :img.shape[2]] = img
    img = paded
return img, fmri_data
```

**saved_feats指定時 (事前抽出)**:
```python
# __init__で特徴量をロード
self.fts_subj_train = np.load(dino_feat_dir + '/train.npy')
self.clip_subj_train = np.load(clip_feat_dir + '/train.npy')

# __getitem__で特徴量を取得
img = torch.tensor(self.fts_subj_train[idx])
img = torch.reshape(img, (962, 768))

# CLIPと結合する場合
if self.cat_clip:
    clip_fts = torch.tensor(self.clip_subj_train[idx])
    clip_fts = torch.tile(clip_fts[None, :], (img.shape[0], 1))
    img = torch.cat((img, clip_fts), dim=1)
    # reshape: [962, 768+512] → [31, 31, 1280] → [1280, 31, 31]
    img = torch.reshape(img[1:,:], (31, 31, 512+768)).permute(2, 0, 1)
else:
    # reshape: [962, 768] → [31, 31, 768] → [768, 31, 31]
    img = torch.reshape(img[1:,:], (31, 31, 768)).permute(2, 0, 1)

return img, fmri_data
```

**重要な注意点**:
1. `img[1:,:]` でCLSトークンを除外 (962→961パッチ)
2. 961 = 31×31 になるようにreshape
3. permute(2,0,1)でチャネルを最初の次元に移動: [C, H, W]
4. この形状がBrain Encoderの入力として期待される

## 関数呼び出し関係図 (Function Call Relationships)

### 訓練時のフロー
```
main()
  │
  ├─ get_args_parser() → args
  │
  ├─ roi_maps(data_dir) → ROI情報
  │
  ├─ fetch_dataloaders(args, train='train')
  │   │
  │   └─ algonauts_dataset.__init__()
  │       │
  │       ├─ [saved_feats=None] → 画像パス保存
  │       └─ [saved_feats指定] → np.load(特徴量)
  │
  ├─ brain_encoder(args)
  │   │
  │   ├─ build_backbone(args)
  │   │   │
  │   │   ├─ [dinov2_q] → dino_model_with_hooks()
  │   │   ├─ [dinov2] → dino_model()
  │   │   ├─ [clip] → clip_model()
  │   │   └─ [resnet] → resnet_model()
  │   │
  │   └─ build_transformer(args) / build_custom_transformer(args)
  │
  ├─ SetCriterion(lh_challenge_rois, rh_challenge_rois)
  │
  └─ for epoch in range(epochs):
      │
      ├─ train_one_epoch(model, criterion, train_loader, ...)
      │   │
      │   └─ for imgs, targets in train_loader:
      │       │
      │       ├─ [saved_feats=None]
      │       │   → algonauts_dataset.__getitem__()
      │       │   → Image.open() → transform() → パディング
      │       │
      │       ├─ [saved_feats指定]
      │       │   → algonauts_dataset.__getitem__()
      │       │   → torch.tensor(事前抽出特徴量) → reshape
      │       │
      │       ├─ nested_tensor_from_tensor_list(imgs)
      │       │
      │       ├─ model(imgs)
      │       │   │
      │       │   ├─ [saved_feats=None]
      │       │   │   → backbone_model(samples)
      │       │   │       → dino_model_with_hooks.forward()
      │       │   │           → backbone.get_intermediate_layers()
      │       │   │           → hook取得: qkv_feats
      │       │   │           → reshape & permute
      │       │   │           → NestedTensor(x, mask)
      │       │   │
      │       │   ├─ [saved_feats指定]
      │       │   │   → samples = NestedTensor(事前整形済み特徴量, mask)
      │       │   │   → backboneをスキップ
      │       │   │
      │       │   └─ transformer(features, mask, query_embed, pos_embed)
      │       │       → lh_embed(output_tokens)
      │       │       → rh_embed(output_tokens)
      │       │
      │       ├─ criterion(outputs, targets) → loss
      │       │
      │       └─ loss.backward() → optimizer.step()
      │
      └─ evaluate(model, criterion, val_loader, ...)
          │
          └─ 同様のフロー (勾配計算なし)
```

### 特徴量抽出時のフロー (extract_features.py)
```
main()
  │
  ├─ argparse → data_dir, output_dir, backbone
  │
  ├─ 画像リスト取得
  │   → train_imgs = [img1.png, img2.png, ...]
  │   → test_imgs = [img1.png, img2.png, ...]
  │
  ├─ [backbone='dinov2_q']
  │   └─ extract_dino_features_with_hooks()
  │       │
  │       ├─ dino_model_with_hooks(enc_output_layer=-1)
  │       │
  │       └─ for batch in batches:
  │           │
  │           ├─ Image.open() → resize(224,224) → normalize
  │           ├─ batch_tensor = torch.stack(imgs)
  │           ├─ nested_tensor = NestedTensor(batch_tensor, mask)
  │           │
  │           ├─ model.backbone.get_intermediate_layers(nested_tensor.tensors)
  │           ├─ feats = model.qkv_feats['qkv_feats']
  │           ├─ reshape: [B, 257, 3, 12, 64] → [3, B, 12, 257, 64]
  │           ├─ q = feats[0] → [B, 257, 768]
  │           ├─ xs_feats = q[:,1:,:] → [B, 256, 768] (CLSトークン除去)
  │           │
  │           └─ all_features.append(xs_feats.cpu().numpy())
  │
  ├─ [backbone='clip']
  │   └─ extract_clip_features()
  │       │
  │       ├─ clip_model(enc_output_layer=-1)
  │       │
  │       └─ for batch in batches:
  │           │
  │           ├─ Image.open() → resize → normalize
  │           ├─ cls_token, patch_tokens = model.backbone.visual(xs)
  │           ├─ patch_tokens_proj = patch_tokens @ proj
  │           │
  │           └─ all_features.append(patch_tokens_proj.cpu().numpy())
  │
  └─ np.save(output_dir + '/train.npy', all_features)
     np.save(output_dir + '/synt.npy', test_features)
```

## まとめ (Summary)

このシステムは、2つのモードで動作します:

### オンザフライモード (Online Mode)
- 毎回画像を読み込み、バックボーンで特徴量抽出
- メリット: シンプル、ディスク容量不要
- デメリット: 訓練が遅い、毎回同じ計算を繰り返す

### 事前抽出モード (Pre-extracted Mode)
- extract_features.pyで事前に特徴量を抽出・保存
- main.pyでは保存済み特徴量を読み込むだけ
- メリット: 訓練が高速、異なるエンコーダーで再利用可能
- デメリット: ディスク容量が必要

両モードは`datasets/nsd.py`の`algonauts_dataset`クラスで透過的に切り替わり、後続の処理は同じパイプラインで動作します。
