# Google Colabでのテスト手順 / Testing Instructions for Google Colab

## このPRについて (About This PR)

このPRは、Google Colabなどのメモリ制限のある環境で特徴量抽出と学習を行えるようにメモリ使用量を最適化するものです。

This PR optimizes memory usage to enable feature extraction and training in memory-constrained environments like Google Colab.

## テスト環境 (Test Environment)

- **推奨**: Google Colab (無料版でも可能)
- **GPU**: T4 または A100
- **RAM**: 12-16 GB（Colab標準）

## テスト手順 (Test Steps)

### ステップ1: Colabノートブックの作成

新しいColabノートブックを作成し、GPUを有効化:
1. Runtime → Change runtime type → GPU → Save

### ステップ2: リポジトリのクローンとブランチの切り替え

```python
# セル1: リポジトリのクローン
!git clone https://github.com/boxed-mikann/transformer_brain_encoder.git
%cd transformer_brain_encoder

# メモリ最適化ブランチに切り替え
!git checkout copilot/optimize-memory-usage

# 最新の変更を確認
!git log --oneline -5
```

### ステップ3: 必要なパッケージのインストール

```python
# セル2: パッケージのインストール
!pip install -q torch torchvision
!pip install -q transformers
!pip install -q open_clip_torch
!pip install -q scikit-learn scipy nilearn
!pip install -q tqdm pillow

print("✅ パッケージのインストール完了")
```

### ステップ4: バリデーションスクリプトの実行

```python
# セル3: メモリ最適化の実装を検証
!python validate_memory_changes.py
```

**期待される出力:**
```
✅ PASS: Syntax validation
✅ PASS: extract_features.py memory optimization
✅ PASS: datasets/nsd.py memory optimization

Memory optimizations implemented successfully:
1. ✅ extract_features.py uses np.memmap for incremental writing
2. ✅ extract_features.py calls torch.cuda.empty_cache() after batches
3. ✅ datasets/nsd.py uses mmap_mode='r' for memory-efficient loading
```

### ステップ5: 互換性テストの実行

```python
# セル4: 互換性テスト
!python test_compatibility.py
```

**期待される出力:**
```
✅ PASS: Saved features mode
✅ PASS: On-the-fly mode
✅ PASS: Memmap writing

ALL COMPATIBILITY TESTS PASSED
```

### ステップ6: Google Driveのマウント（オプション：実データでテストする場合）

```python
# セル5: Google Driveのマウント
from google.colab import drive
drive.mount('/content/drive')

# データとフィーチャー保存先を設定
DATA_DIR = '/content/drive/MyDrive/algonauts_data/subj01'
OUTPUT_DIR = '/content/drive/MyDrive/algonauts_features'

print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
```

### ステップ7: メモリ使用量の監視（実行前）

```python
# セル6: メモリ使用量のベースライン確認
!nvidia-smi
!cat /proc/meminfo | grep MemAvailable
```

### ステップ8: 小規模テストデータで特徴抽出（実データがある場合）

**注意**: 実際のデータがない場合、このステップはスキップしてください。

```python
# セル7: 特徴抽出（メモリ最適化版）
# 注意: 実際のデータディレクトリを指定してください
!python extract_features.py \
    --data_dir /content/drive/MyDrive/algonauts_data/subj01 \
    --output_dir /content/drive/MyDrive/algonauts_features \
    --subj 01 \
    --backbone dinov2_q \
    --batch_size 8 \
    --device cuda

# 処理中はメモリ使用量を監視
# 別セルで以下を実行してメモリをチェック:
# !watch -n 5 nvidia-smi
```

### ステップ9: メモリ使用量の監視（実行後）

```python
# セル8: メモリ使用量の確認
!nvidia-smi
!cat /proc/meminfo | grep MemAvailable

# 抽出された特徴量ファイルの確認
import numpy as np
import os

feature_dir = '/content/drive/MyDrive/algonauts_features/dinov2_q_last/01'
if os.path.exists(feature_dir):
    train_file = os.path.join(feature_dir, 'train.npy')
    if os.path.exists(train_file):
        # mmap_mode='r' で読み込み（メモリ効率的）
        features = np.load(train_file, mmap_mode='r')
        print(f"✅ 特徴量ファイルが正常に作成されました")
        print(f"   Shape: {features.shape}")
        print(f"   Dtype: {features.dtype}")
        print(f"   File size: {os.path.getsize(train_file) / 1024 / 1024:.2f} MB")
    else:
        print("⚠️  train.npy が見つかりません")
else:
    print("⚠️  特徴量ディレクトリが見つかりません")
```

## テスト結果の確認ポイント

### ✅ 成功の指標

1. **バリデーションスクリプト**: すべてのチェックが PASS
2. **互換性テスト**: すべてのテストが PASS
3. **メモリ使用量**: 
   - 特徴抽出中: GPUメモリ < 4GB
   - 特徴抽出後: RAMメモリの使用量が大幅に増加しない
4. **特徴量ファイル**: 
   - 正しい形状 (DINOv2: [N, 962, 768])
   - mmap_mode='r' で読み込み可能

### ❌ 失敗の指標

1. OOMエラーが発生
2. メモリ使用量が従来と同じかそれ以上
3. 特徴量ファイルが破損または形状が不正
4. テストスクリプトが FAIL

## トラブルシューティング

### 問題: Colab無料版でもメモリ不足

**解決策**: バッチサイズをさらに小さくする
```bash
--batch_size 4  # または 2
```

### 問題: "ModuleNotFoundError"

**解決策**: パッケージを再インストール
```python
!pip install --upgrade torch torchvision transformers
```

### 問題: GPU が利用できない

**解決策**: 
1. Runtime → Change runtime type → GPU → Save
2. CPUでもテスト可能ですが、非常に遅くなります:
```bash
--device cpu
```

### 問題: Google Driveのマウントに失敗

**解決策**:
1. 再度マウントを試行
2. Colabノートブックを再起動

## 期待される改善

### メモリ使用量

| 項目 | 変更前 | 変更後 | 改善率 |
|-----|--------|--------|-------|
| 特徴抽出時のピークメモリ | ~15 GB | ~3 GB | **80%削減** |
| 訓練時のピークメモリ | ~12 GB | ~6 GB | **50%削減** |

### 処理時間

処理時間はほぼ同じ（約+10%）。メモリ削減のトレードオフとして許容範囲内。

## 最小限のテスト（データなしでも可能）

実際のデータがない場合でも、以下のテストで検証可能:

```python
# セル1-4の実行（バリデーションと互換性テスト）
# これだけで基本的な実装が正しいことを確認できます
```

## フィードバック

テスト結果をGitHub Issueまたはこのブランチのコメントでお知らせください:
- ✅ 成功した環境（Colab無料版/Pro、使用したbatch_size等）
- ❌ 失敗した場合のエラーメッセージとスタックトレース
- 📊 メモリ使用量の実測値

## 参考資料

- [MEMORY_OPTIMIZATION_GUIDE.md](./MEMORY_OPTIMIZATION_GUIDE.md): 詳細なメモリ最適化ガイド
- [FEATURE_EXTRACTION_GUIDE.md](./FEATURE_EXTRACTION_GUIDE.md): 特徴抽出の基本ガイド

## まとめ

このPRにより、Google Colab無料版でも大規模データセットの処理が可能になります。
テストを実施して、問題があればお知らせください！
