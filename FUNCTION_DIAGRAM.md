# Function Call Diagram / 関数呼び出し図

## 完全な処理フロー図 (Complete Processing Flow)

```
┌────────────────────────────────────────────────────────────────────────┐
│                           USER WORKFLOW                                 │
│                         ユーザーワークフロー                              │
└────────────────────────────────────────────────────────────────────────┘

Option 1: オンザフライモード (Online Mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    python main.py --subj 1 --backbone_arch dinov2_q
           │
           ├─ main()
           │   ├─ データローダー作成: fetch_dataloaders()
           │   │   └─ algonauts_dataset.__getitem__()
           │   │       └─ Image.open() → transform() → Tensor
           │   │
           │   ├─ モデル作成: brain_encoder()
           │   │   ├─ build_backbone() → dino_model_with_hooks()
           │   │   └─ build_transformer()
           │   │
           │   └─ 訓練ループ
           │       ├─ for img, fmri in train_loader:
           │       │   ├─ backbone(img) → features
           │       │   ├─ transformer(features) → output_tokens
           │       │   ├─ lh_embed, rh_embed → predictions
           │       │   └─ loss.backward() → update
           │       │
           │       └─ evaluate() → 検証


Option 2: 事前抽出モード (Pre-extracted Mode)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    STEP 1: Feature Extraction
    ──────────────────────────
    python extract_features.py --data_dir ... --output_dir ...
           │
           ├─ main()
           │   ├─ 画像リスト取得
           │   │   └─ os.listdir(training_images/) → [img1.png, ...]
           │   │
           │   ├─ モデルロード
           │   │   ├─ [dinov2_q] → dino_model_with_hooks()
           │   │   ├─ [dinov2]   → dino_model()
           │   │   └─ [clip]     → clip_model()
           │   │
           │   ├─ バッチ処理ループ
           │   │   └─ for batch in batches:
           │   │       ├─ Image.open() → resize → normalize
           │   │       ├─ model(images) → features
           │   │       └─ features.append()
           │   │
           │   └─ np.save(output_dir/train.npy)
           │       np.save(output_dir/synt.npy)
           │
           └─ 保存された特徴量:
               - dinov2_q_last/01/train.npy  [N, 962, 768]
               - dinov2_q_last/01/synt.npy   [N, 962, 768]

    STEP 2: Training with Pre-extracted Features
    ─────────────────────────────────────────────
    python main.py --saved_feats dinov2q --saved_feats_dir ...
           │
           ├─ main()
           │   ├─ データローダー作成: fetch_dataloaders()
           │   │   └─ algonauts_dataset.__init__()
           │   │       ├─ np.load(dinov2_q_last/01/train.npy)
           │   │       └─ self.fts_subj_train = features
           │   │
           │   ├─ モデル作成: brain_encoder()
           │   │   ├─ build_backbone() [スキップ or CLSモード]
           │   │   └─ build_transformer()
           │   │
           │   └─ 訓練ループ
           │       └─ for features, fmri in train_loader:
           │           ├─ [backboneスキップ]
           │           ├─ transformer(features) → output_tokens
           │           ├─ lh_embed, rh_embed → predictions
           │           └─ loss.backward() → update
```

## 詳細な関数呼び出しツリー (Detailed Function Call Tree)

### 1. メイン訓練フロー (Main Training Flow)

```
main.py::main(rank, world_size, args)
│
├─ argparse.parse_args() → args
│
├─ datasets/nsd_utils.py::roi_maps(data_dir)
│   └─ → roi_name_maps, lh_challenge_rois, rh_challenge_rois
│
├─ datasets/nsd_utils.py::roi_masks(readout_res, ...)
│   └─ → lh_challenge_rois_s, rh_challenge_rois_s, num_queries
│
├─ datasets/nsd.py::fetch_dataloaders(args, train='train')
│   │
│   ├─ algonauts_dataset(args, 'train', train_imgs_paths, idxs_train)
│   │   │
│   │   ├─ __init__(self, args, is_train, imgs_paths, idxs, transform)
│   │   │   │
│   │   │   ├─ [saved_feats = None]
│   │   │   │   └─ self.imgs_paths = imgs_paths[idxs]
│   │   │   │
│   │   │   └─ [saved_feats 指定]
│   │   │       ├─ np.load(dino_feat_dir + '/train.npy')
│   │   │       ├─ np.load(clip_feat_dir + '/train.npy')
│   │   │       └─ self.fts_subj_train = fts[idxs]
│   │   │
│   │   └─ __getitem__(self, idx)
│   │       │
│   │       ├─ [saved_feats = None]
│   │       │   ├─ Image.open(imgs_paths[idx])
│   │       │   ├─ transform(img)
│   │       │   └─ [dinov2] パディング処理
│   │       │
│   │       └─ [saved_feats 指定]
│   │           ├─ img = torch.tensor(fts_subj_train[idx])
│   │           ├─ reshape: [962, 768] → [31, 31, 768]
│   │           ├─ permute: [31, 31, 768] → [768, 31, 31]
│   │           └─ [cat_clip] CLIP特徴量と結合
│   │
│   └─ DataLoader(algonauts_dataset, ...)
│
├─ models/brain_encoder.py::brain_encoder(args)
│   │
│   ├─ models/backbone.py::build_backbone(args)
│   │   │
│   │   ├─ [resnet*]
│   │   │   └─ models/resnet.py::resnet_model(...)
│   │   │
│   │   ├─ [dinov2_q*]
│   │   │   └─ models/dino.py::dino_model_with_hooks(enc_output_layer, ...)
│   │   │       ├─ torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
│   │   │       ├─ backbone.blocks[layer].attn.qkv.register_forward_hook(...)
│   │   │       └─ forward(tensor_list: NestedTensor)
│   │   │           ├─ backbone.get_intermediate_layers(xs)[0]
│   │   │           ├─ hook取得: qkv_feats
│   │   │           ├─ reshape & permute
│   │   │           └─ NestedTensor(features, mask)
│   │   │
│   │   ├─ [dinov2*]
│   │   │   └─ models/dino.py::dino_model(enc_output_layer, ...)
│   │   │       ├─ torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
│   │   │       └─ forward(tensor_list: NestedTensor)
│   │   │           ├─ backbone.get_intermediate_layers(xs, n=12)
│   │   │           ├─ xs[enc_output_layer]
│   │   │           └─ NestedTensor(features, mask)
│   │   │
│   │   └─ [clip*]
│   │       └─ models/clip.py::clip_model(enc_output_layer, ...)
│   │           ├─ open_clip.create_model_and_transforms('ViT-L-14', ...)
│   │           └─ forward(tensor_list: NestedTensor)
│   │               ├─ backbone.visual(xs) → cls_token, patch_tokens
│   │               ├─ patch_tokens @ proj
│   │               └─ NestedTensor(features, mask)
│   │
│   ├─ [encoder_arch = 'transformer']
│   │   └─ models/transformer.py::build_transformer(args)
│   │       └─ Transformer(d_model, nhead, num_encoder_layers, ...)
│   │
│   ├─ [encoder_arch = 'custom_transformer']
│   │   └─ models/custom_transformer.py::build_custom_transformer(args)
│   │
│   └─ nn.Embedding(num_queries, hidden_dim) → query_embed
│
├─ main.py::SetCriterion(lh_challenge_rois_s, rh_challenge_rois_s)
│   │
│   └─ forward(outputs, targets)
│       ├─ [encoder_arch != 'linear'] ROI集約処理
│       ├─ MSELoss(lh_f_pred, lh_f)
│       ├─ MSELoss(rh_f_pred, rh_f)
│       └─ [encoder_arch = 'linear'] L2正則化追加
│
├─ torch.optim.AdamW(model.parameters(), lr=args.lr, ...)
├─ torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, ...)
│
└─ for epoch in range(start_epoch, epochs):
    │
    ├─ engine.py::train_one_epoch(model, criterion, train_loader, ...)
    │   │
    │   └─ for imgs, targets in data_loader:
    │       │
    │       ├─ utils/utils.py::nested_tensor_from_tensor_list(imgs)
    │       │   └─ NestedTensor(tensors, mask)
    │       │
    │       ├─ model(imgs)
    │       │   │
    │       │   ├─ [saved_feats = None]
    │       │   │   ├─ backbone_model(samples)
    │       │   │   │   └─ dino/clip/resnet forward()
    │       │   │   │       └─ NestedTensor(features, mask), pos
    │       │   │   │
    │       │   │   └─ features, pos = backbone_model(samples)
    │       │   │
    │       │   ├─ [saved_feats 指定]
    │       │   │   └─ samples = NestedTensor(事前整形済み特徴量, mask)
    │       │   │       [backboneをスキップ]
    │       │   │
    │       │   ├─ features.decompose() → input_proj_src, mask
    │       │   │
    │       │   ├─ [encoder_arch = 'transformer']
    │       │   │   ├─ transformer(input_proj_src, mask, query_embed, pos)
    │       │   │   │   ├─ encoder(src, ...)
    │       │   │   │   └─ decoder(tgt, memory, ...)
    │       │   │   │       └─ output_tokens [B, num_queries, hidden_dim]
    │       │   │   │
    │       │   │   ├─ lh_embed(output_tokens[:, :num_queries//2, :])
    │       │   │   │   └─ lh_f_pred [B, lh_vs, num_queries//2]
    │       │   │   │
    │       │   │   └─ rh_embed(output_tokens[:, num_queries//2:, :])
    │       │   │       └─ rh_f_pred [B, rh_vs, num_queries//2]
    │       │   │
    │       │   └─ [encoder_arch = 'linear']
    │       │       ├─ input_proj(features) → flattened
    │       │       ├─ lh_embed(flattened) → lh_f_pred
    │       │       └─ rh_embed(flattened) → rh_f_pred
    │       │
    │       ├─ criterion(outputs, targets)
    │       │   └─ loss_lh + loss_rh (+ L2 reg)
    │       │
    │       ├─ loss.backward()
    │       ├─ clip_grad_norm_(model.parameters(), max_norm)
    │       └─ optimizer.step()
    │
    ├─ lr_scheduler.step()
    │
    └─ engine.py::evaluate(model, criterion, val_loader, ...)
        │
        └─ for samples, targets in data_loader:
            ├─ model(samples) [勾配なし]
            ├─ [encoder_arch != 'linear'] ROI集約
            ├─ 予測値と真値を収集
            └─ correlation計算
                └─ scipy.stats.pearsonr(pred, true)
```

### 2. 特徴量抽出フロー (Feature Extraction Flow)

```
extract_features.py::main()
│
├─ argparse.parse_args() → args
│
├─ os.listdir(train_img_dir) → train_img_list
├─ os.listdir(test_img_dir) → test_img_list
│
├─ [backbone = 'dinov2_q']
│   └─ extract_dino_features_with_hooks(image_dir, output_path, ...)
│       │
│       ├─ models/dino.py::dino_model_with_hooks(enc_output_layer=-1, ...)
│       │   └─ torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
│       │
│       ├─ for i in range(0, len(img_files), batch_size):
│       │   │
│       │   ├─ for img_file in batch_files:
│       │   │   ├─ Image.open(img_path).convert('RGB')
│       │   │   ├─ resize(224, 224)
│       │   │   ├─ transforms.ToTensor()
│       │   │   ├─ transforms.Normalize([0.485, 0.456, 0.406], ...)
│       │   │   └─ パディング (14の倍数)
│       │   │
│       │   ├─ batch_tensor = torch.stack(batch_imgs)
│       │   ├─ nested_tensor = NestedTensor(batch_tensor, mask)
│       │   │
│       │   ├─ model.backbone.get_intermediate_layers(nested_tensor.tensors)
│       │   │   └─ xs [B, 257, 768]  # 16×16+1 patches
│       │   │
│       │   ├─ feats = model.qkv_feats['qkv_feats']
│       │   │   └─ [B, 257, 3×12×64]  # 3=qkv, 12=heads, 64=dim
│       │   │
│       │   ├─ reshape & permute
│       │   │   ├─ [B, 257, 3, 12, 64] → [3, B, 12, 257, 64]
│       │   │   ├─ q = feats[0] → [B, 12, 257, 64]
│       │   │   ├─ transpose(1,2) → [B, 257, 12, 64]
│       │   │   └─ reshape → [B, 257, 768]
│       │   │
│       │   ├─ xs_feats = q[:,1:,:]  # CLSトークン除外 → [B, 256, 768]
│       │   │   ※実際は q 全体を保存: [B, 257, 768]
│       │   │   ※datasets/nsd.pyで img[1:,:] を使用
│       │   │
│       │   └─ all_features.append(feats_np)
│       │
│       ├─ all_features = np.concatenate(all_features, axis=0)
│       │   └─ [N, 257, 768]
│       │
│       └─ np.save(output_path, all_features)
│
├─ [backbone = 'dinov2']
│   └─ extract_dino_features_simple(image_dir, output_path, ...)
│       │
│       ├─ models/dino.py::dino_model(enc_output_layer=-1, ...)
│       │
│       ├─ for batch in batches:
│       │   ├─ 画像処理 (同上)
│       │   ├─ model.backbone.get_intermediate_layers(xs, n=12)
│       │   ├─ xs_layer = xs[enc_output_layer]
│       │   └─ all_features.append(xs_layer.cpu().numpy())
│       │
│       └─ np.save(output_path, all_features)
│
└─ [backbone = 'clip']
    └─ extract_clip_features(image_dir, output_path, ...)
        │
        ├─ models/clip.py::clip_model(enc_output_layer=-1, ...)
        │   └─ open_clip.create_model_and_transforms('ViT-L-14', ...)
        │
        ├─ for batch in batches:
        │   ├─ 画像処理 (パディングなし)
        │   ├─ cls_token, patch_tokens = model.backbone.visual(batch_tensor)
        │   │   └─ cls_token [B, 768], patch_tokens [B, 256, 1024]
        │   │
        │   ├─ proj = model.backbone.visual.proj  # [1024, 768]
        │   ├─ patch_tokens_proj = patch_tokens @ proj  # [B, 256, 768]
        │   │
        │   ├─ cls_token_reshaped = cls_token.unsqueeze(1)  # [B, 1, 768]
        │   ├─ full_tokens = torch.cat([cls_token_reshaped, patch_tokens_proj], dim=1)
        │   │   └─ [B, 257, 768]
        │   │
        │   └─ all_features.append(full_tokens.cpu().numpy())
        │
        └─ np.save(output_path, all_features)
```

### 3. データフロー詳細 (Detailed Data Flow)

#### オンザフライモード (Online Mode)
```
画像ファイル (.png)
  │
  ├─ datasets/nsd.py::algonauts_dataset.__getitem__()
  │   ├─ Image.open() → PIL Image [H, W, 3]
  │   ├─ transforms.ToTensor() → Tensor [3, H, W]
  │   ├─ transforms.Normalize() → 正規化済み Tensor [3, H, W]
  │   └─ [dinov2] パディング → [3, 224, 224]
  │
  ├─ DataLoader → バッチ [B, 3, 224, 224]
  │
  ├─ utils/utils.py::nested_tensor_from_tensor_list()
  │   └─ NestedTensor(tensors [B, 3, 224, 224], mask [B, 224, 224])
  │
  ├─ models/backbone.py::build_backbone()
  │   └─ models/dino.py::dino_model_with_hooks.forward()
  │       ├─ backbone.get_intermediate_layers() → [B, 257, 768]
  │       ├─ hook取得: qkv_feats
  │       ├─ reshape & extract Q → [B, 257, 768]
  │       ├─ q[:,1:,:] → [B, 256, 768]  # CLSトークン除外
  │       ├─ reshape → [B, 768, 16, 16]
  │       └─ NestedTensor(features [B, 768, 16, 16], mask [B, 16, 16])
  │
  ├─ models/transformer.py::Transformer.forward()
  │   ├─ encoder(src [B, 256, 768], ...)
  │   └─ decoder(tgt [num_queries, 768], memory [B, 256, 768], ...)
  │       └─ output_tokens [B, num_queries, 768]
  │
  ├─ models/brain_encoder.py::lh_embed()
  │   └─ lh_f_pred [B, lh_vs]
  │
  └─ models/brain_encoder.py::rh_embed()
      └─ rh_f_pred [B, rh_vs]
```

#### 事前抽出モード (Pre-extracted Mode)
```
特徴量ファイル (.npy)
  │
  ├─ datasets/nsd.py::algonauts_dataset.__init__()
  │   └─ np.load(dinov2_q_last/01/train.npy) → [N, 962, 768]
  │
  ├─ datasets/nsd.py::algonauts_dataset.__getitem__()
  │   ├─ features[idx] → [962, 768]
  │   ├─ features[1:,:] → [961, 768]  # CLSトークン除外
  │   ├─ reshape → [31, 31, 768]
  │   └─ permute(2,0,1) → [768, 31, 31]
  │
  ├─ DataLoader → バッチ [B, 768, 31, 31]
  │
  ├─ utils/utils.py::nested_tensor_from_tensor_list()
  │   └─ NestedTensor(tensors [B, 768, 31, 31], mask [B, 31, 31])
  │
  ├─ [backboneスキップ]
  │
  ├─ models/transformer.py::Transformer.forward()
  │   ├─ flatten: [B, 768, 31, 31] → [B, 768, 961]
  │   ├─ permute: [B, 768, 961] → [B, 961, 768]
  │   ├─ encoder(src [B, 961, 768], ...)
  │   └─ decoder(tgt [num_queries, 768], memory [B, 961, 768], ...)
  │       └─ output_tokens [B, num_queries, 768]
  │
  ├─ models/brain_encoder.py::lh_embed()
  │   └─ lh_f_pred [B, lh_vs]
  │
  └─ models/brain_encoder.py::rh_embed()
      └─ rh_f_pred [B, rh_vs]
```

## 重要な注意点 (Important Notes)

### 1. 特徴量の形状の互換性
```python
# extract_features.pyで保存: [N, 962, 768]
# datasets/nsd.pyで読み込み:
features[idx]      # [962, 768]
features[idx][1:]  # [961, 768]  ← CLSトークン除外
reshape(31, 31, 768)  # [31, 31, 768]
permute(2, 0, 1)      # [768, 31, 31]
```

### 2. パッチ数の計算
```python
# DINOv2:
# 画像サイズ: 224×224
# パッチサイズ: 14×14
# パッチ数: (224/14) × (224/14) = 16×16 = 256
# + CLSトークン = 257
# パディング後: 31×31 = 961 (+ CLS = 962)

# CLIP:
# 画像サイズ: 224×224
# パッチサイズ: 14×14
# パッチ数: 16×16 = 256
# + CLSトークン = 257
```

### 3. モジュール間の依存関係
```
main.py
  ├─ depends on: datasets/nsd.py
  ├─ depends on: models/brain_encoder.py
  ├─ depends on: engine.py
  └─ depends on: utils/utils.py

models/brain_encoder.py
  ├─ depends on: models/backbone.py
  ├─ depends on: models/transformer.py
  └─ depends on: utils/utils.py

models/backbone.py
  ├─ depends on: models/dino.py
  ├─ depends on: models/clip.py
  ├─ depends on: models/resnet.py
  └─ depends on: models/position_encoding.py

extract_features.py
  ├─ depends on: models/dino.py
  ├─ depends on: models/clip.py
  └─ depends on: utils/utils.py
```

## まとめ (Summary)

この図は、Transformer Brain Encoderの完全な処理フローを示しています:

1. **オンザフライモード**: 画像 → バックボーン → Transformer → 予測
2. **事前抽出モード**: 画像 → 特徴量ファイル → Transformer → 予測

両モードとも同じTransformerと予測層を使用しますが、特徴量の抽出タイミングが異なります。
事前抽出モードは訓練を高速化し、複数の実験で特徴量を再利用できます。
