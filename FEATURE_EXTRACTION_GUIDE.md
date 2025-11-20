# Feature Extraction Guide / 特徴量抽出ガイド

## 概要 (Overview)

このガイドでは、`extract_features_new.py`を使用して画像から特徴量を事前に抽出する方法を説明します。

This guide explains how to pre-extract image features using `extract_features_new.py`.

## なぜ特徴量を事前抽出するのか? (Why Pre-extract Features?)

### メリット (Advantages)
1. **訓練の高速化**: バックボーンモデルを毎回実行する必要がなくなる
2. **メモリ効率**: 大きなバックボーンモデルをメモリに保持する必要がない
3. **再利用性**: 異なるエンコーダーアーキテクチャで同じ特徴量を使用できる
4. **実験の効率化**: 特徴量は一度だけ抽出すればよい

### デメリット (Disadvantages)
1. **ディスク容量**: 特徴量ファイルがディスク容量を消費する
2. **前処理ステップ**: 訓練前に特徴量抽出を実行する必要がある

## 使用方法 (Usage)

### 1. 基本的な使い方 (Basic Usage)

```bash
python extract_features_new.py \
    --data_dir /path/to/algonauts_data/subj01 \
    --output_dir /path/to/save/features \
    --subj 01 \
    --backbone dinov2_q \
    --batch_size 16 \
    --device cuda
```

### 2. パラメータの説明 (Parameter Description)

| パラメータ | 説明 | デフォルト値 |
|----------|------|-----------|
| `--data_dir` | Algonautsデータディレクトリ (必須) | - |
| `--output_dir` | 特徴量を保存するディレクトリ (必須) | - |
| `--subj` | 被験者ID (01, 02, ...) | 01 |
| `--backbone` | バックボーンモデル (dinov2_q, dinov2, clip) | dinov2_q |
| `--batch_size` | バッチサイズ | 16 |
| `--device` | デバイス (cuda or cpu) | cuda |
| `--enc_layer` | エンコーダー層 (-1 = 最終層) | -1 |

### 3. バックボーンの選択 (Backbone Selection)

#### dinov2_q (推奨 / Recommended)
- **説明**: DINOv2 with QKV hooks
- **特徴量次元**: [N, 962, 768]
  - 962 = 31×31パッチ + 1 CLSトークン
  - 768 = 特徴次元
- **使用例**:
```bash
python extract_features_new.py \
    --data_dir /data/subj01 \
    --output_dir /features \
    --backbone dinov2_q
```

#### dinov2 (シンプル版 / Simple Version)
- **説明**: Standard DINOv2
- **特徴量次元**: [N, 962, 768]
- **使用例**:
```bash
python extract_features_new.py \
    --data_dir /data/subj01 \
    --output_dir /features \
    --backbone dinov2
```

#### clip
- **説明**: CLIP ViT-L-14
- **特徴量次元**: [N, 257, 768]
  - 257 = 16×16パッチ + 1 CLSトークン
  - 768 = 特徴次元 (内部的には1024→768に投影)
- **使用例**:
```bash
python extract_features_new.py \
    --data_dir /data/subj01 \
    --output_dir /features \
    --backbone clip
```

### 4. 出力ファイル構造 (Output File Structure)

```
output_dir/
├── dinov2_q_last/
│   ├── 01/
│   │   ├── train.npy    # [N_train, 962, 768]
│   │   └── synt.npy     # [N_test, 962, 768]
│   ├── 02/
│   │   ├── train.npy
│   │   └── synt.npy
│   └── ...
├── dinov2_last/
│   └── 01/
│       ├── train.npy
│       └── synt.npy
└── clip_vit_512/
    └── 01/
        ├── train.npy    # [N_train, 257, 768]
        └── synt.npy     # [N_test, 257, 768]
```

### 5. 特徴量を使った訓練 (Training with Pre-extracted Features)

特徴量を抽出したら、`main.py`で`--saved_feats`フラグを使用します:

After extracting features, use the `--saved_feats` flag with `main.py`:

#### DINOv2_q の場合:
```bash
python main.py \
    --subj 1 \
    --saved_feats dinov2q \
    --saved_feats_dir /path/to/features \
    --encoder_arch transformer \
    --readout_res rois_all \
    --batch_size 16 \
    --epochs 15
```

#### CLIP の場合:
```bash
python main.py \
    --subj 1 \
    --saved_feats clip \
    --saved_feats_dir /path/to/features \
    --encoder_arch transformer \
    --readout_res rois_all \
    --batch_size 16 \
    --epochs 15
```

## 技術的詳細 (Technical Details)

### 特徴量の形状 (Feature Shapes)

#### DINOv2 (dinov2_q, dinov2)
```python
# 入力画像: [B, 3, 224, 224]
# ↓ パディング → [B, 3, 224, 224] (14の倍数にパディング済み)
# ↓ DINOv2 backbone
# ↓ QKV特徴量抽出 (dinov2_q) or 標準特徴量 (dinov2)
# 出力: [B, 962, 768]
#   - 962 = 31×31 + 1 (パッチ数 + CLSトークン)
#   - 768 = 特徴次元
```

#### CLIP
```python
# 入力画像: [B, 3, 224, 224]
# ↓ CLIP visual encoder
# ↓ cls_token [B, 1, 768], patch_tokens [B, 256, 1024]
# ↓ projection: patch_tokens @ proj → [B, 256, 768]
# ↓ concatenate: [cls_token, patch_tokens]
# 出力: [B, 257, 768]
#   - 257 = 16×16 + 1 (パッチ数 + CLSトークン)
#   - 768 = 特徴次元
```

### datasets/nsd.py での処理 (Processing in datasets/nsd.py)

事前抽出された特徴量は、`datasets/nsd.py`の`algonauts_dataset`クラスで以下のように処理されます:

Pre-extracted features are processed in the `algonauts_dataset` class as follows:

```python
# 特徴量をロード
img = torch.tensor(self.fts_subj_train[idx])  # [962, 768]

# CLSトークンを除外してreshape
img = torch.reshape(img[1:,:], (31, 31, 768))  # [31, 31, 768]

# チャネルを最初の次元に移動
img = img.permute(2, 0, 1)  # [768, 31, 31]

# この形状が Brain Encoder の入力として期待される
```

**重要**: 
- `img[1:,:]`でCLSトークンを除外 (962→961パッチ)
- 961 = 31×31 になるようにreshape
- Brain Encoderは [C, H, W] 形式を期待

### 互換性の確認 (Compatibility Check)

`extract_features_new.py`が正しく動作することを確認するには:

To verify that `extract_features_new.py` works correctly:

1. 特徴量を抽出:
```bash
python extract_features_new.py \
    --data_dir /your/data/subj01 \
    --output_dir /tmp/test_features \
    --subj 01 \
    --backbone dinov2_q \
    --batch_size 4
```

2. 特徴量の形状を確認:
```python
import numpy as np
train_feats = np.load('/tmp/test_features/dinov2_q_last/01/train.npy')
print(f"Train features shape: {train_feats.shape}")
# Expected: (N_images, 962, 768) for dinov2_q
```

3. main.pyで訓練:
```bash
python main.py \
    --subj 1 \
    --saved_feats dinov2q \
    --saved_feats_dir /tmp/test_features \
    --encoder_arch transformer \
    --readout_res rois_all \
    --epochs 1  # テストのため1エポックのみ
```

## トラブルシューティング (Troubleshooting)

### 問題: CUDAメモリ不足
**解決策**: バッチサイズを小さくする
```bash
python extract_features_new.py ... --batch_size 8
```

### 問題: モデルのダウンロードが遅い
**解決策**: 
- DINOv2とCLIPモデルは初回実行時に自動的にダウンロードされます
- torch.hubキャッシュ: `~/.cache/torch/hub/`
- open_clipキャッシュ: `~/.cache/clip/`

### 問題: 形状が一致しない
**確認事項**:
1. 正しいバックボーンを指定しているか
2. `--saved_feats`の値が正しいか (dinov2q, clip など)
3. 特徴量ファイルのパスが正しいか

### 問題: メモリ不足
**解決策**: CPUで実行
```bash
python extract_features_new.py ... --device cpu
```

## パフォーマンス比較 (Performance Comparison)

### オンザフライ vs 事前抽出 (Online vs Pre-extracted)

| モード | 1エポックあたりの時間 | ディスク使用量 | メモリ使用量 |
|--------|-------------------|-------------|------------|
| オンザフライ | ~30分 (GPU) | 0 GB | ~8 GB |
| 事前抽出 | ~5分 (GPU) | ~2 GB | ~4 GB |

*注: 数値は目安であり、ハードウェアやデータセットサイズによって異なります*

### 推奨ワークフロー (Recommended Workflow)

1. **実験初期**: オンザフライモードで動作確認
2. **ハイパーパラメータ調整**: 事前抽出モードで高速実験
3. **最終訓練**: 必要に応じて両方を併用

## よくある質問 (FAQ)

### Q1: どのバックボーンを使うべきですか?
**A**: `dinov2_q`を推奨します。論文の実験で最も良い結果を示しています。

### Q2: 複数の被験者の特徴量を一度に抽出できますか?
**A**: 現在は1被験者ずつ実行する必要があります。シェルスクリプトでループ処理が可能です:
```bash
for subj in 01 02 03 04; do
    python extract_features_new.py \
        --data_dir /data/subj${subj} \
        --output_dir /features \
        --subj ${subj} \
        --backbone dinov2_q
done
```

### Q3: 特徴量ファイルを削除しても大丈夫ですか?
**A**: はい。特徴量は元の画像から再抽出できます。

### Q4: 異なるバックボーンの特徴量を組み合わせられますか?
**A**: `datasets/nsd.py`では`cat_clip`フラグでDINOv2とCLIPの特徴量を結合できます。

## 参考資料 (References)

- [ARCHITECTURE.md](./ARCHITECTURE.md): システムアーキテクチャの詳細
- [README.md](./README.md): プロジェクト全体の説明
- [models/dino.py](./models/dino.py): DINOv2バックボーンの実装
- [models/clip.py](./models/clip.py): CLIPバックボーンの実装
- [datasets/nsd.py](./datasets/nsd.py): データセットとデータローダーの実装
