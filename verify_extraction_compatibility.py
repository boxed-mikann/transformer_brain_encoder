"""
Verification script to check that extract_features.py produces
compatible features for use with main.py --saved_feats mode.

このスクリプトは、extract_features.pyが生成する特徴量が
main.py --saved_featsモードで使用可能な形式であることを検証します。
"""

import numpy as np
import torch
import sys
import os

def verify_feature_shape(features, expected_patches, expected_dim, backbone_name):
    """
    特徴量の形状を検証
    
    Args:
        features: numpy array of features
        expected_patches: 期待されるパッチ数 (CLSトークン含む)
        expected_dim: 期待される特徴次元
        backbone_name: バックボーン名 (エラーメッセージ用)
    """
    print(f"\n{'='*60}")
    print(f"Verifying {backbone_name} features")
    print(f"{'='*60}")
    
    print(f"Feature shape: {features.shape}")
    print(f"Expected shape: [N, {expected_patches}, {expected_dim}]")
    
    # Check dimensions
    if len(features.shape) != 3:
        print(f"❌ ERROR: Expected 3D array, got {len(features.shape)}D")
        return False
    
    if features.shape[1] != expected_patches:
        print(f"❌ ERROR: Expected {expected_patches} patches, got {features.shape[1]}")
        return False
    
    if features.shape[2] != expected_dim:
        print(f"❌ ERROR: Expected feature dim {expected_dim}, got {features.shape[2]}")
        return False
    
    print(f"✅ Shape verification passed!")
    
    # Test reshape operation (as done in datasets/nsd.py)
    print(f"\nTesting datasets/nsd.py reshape operation...")
    try:
        # Simulate what datasets/nsd.py does
        sample = torch.tensor(features[0])  # [patches+1, dim]
        sample_no_cls = sample[1:, :]  # Remove CLS token
        
        # Calculate grid size
        num_patches = sample_no_cls.shape[0]
        grid_size = int(np.sqrt(num_patches))
        
        if grid_size * grid_size != num_patches:
            print(f"⚠️  WARNING: Number of patches ({num_patches}) is not a perfect square")
            print(f"   Attempting to reshape to closest square: {grid_size}x{grid_size}")
        
        # Reshape
        reshaped = torch.reshape(sample_no_cls, (grid_size, grid_size, expected_dim))
        print(f"After removing CLS: {sample_no_cls.shape}")
        print(f"After reshape: {reshaped.shape}")
        
        # Permute to [C, H, W] format
        permuted = reshaped.permute(2, 0, 1)
        print(f"After permute (C, H, W): {permuted.shape}")
        
        print(f"✅ Reshape operation successful!")
        return True
        
    except Exception as e:
        print(f"❌ ERROR during reshape: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*60)
    print("Feature Compatibility Verification")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python verify_extraction_compatibility.py <feature_file.npy>")
        print("\nExample:")
        print("  python verify_extraction_compatibility.py /features/dinov2_q_last/01/train.npy")
        print("  python verify_extraction_compatibility.py /features/clip_vit_512/01/train.npy")
        sys.exit(1)
    
    feature_path = sys.argv[1]
    
    if not os.path.exists(feature_path):
        print(f"\n❌ ERROR: Feature file not found: {feature_path}")
        sys.exit(1)
    
    # Determine backbone type from path
    if 'dinov2_q' in feature_path or 'dinov2' in feature_path:
        backbone = 'dinov2'
        expected_patches = 962  # 31*31 + 1
        expected_dim = 768
    elif 'clip' in feature_path:
        backbone = 'clip'
        expected_patches = 257  # 16*16 + 1
        expected_dim = 768
    else:
        print(f"\n❌ ERROR: Cannot determine backbone type from path: {feature_path}")
        print("   Expected path to contain 'dinov2_q', 'dinov2', or 'clip'")
        sys.exit(1)
    
    # Load features
    print(f"\nLoading features from: {feature_path}")
    try:
        features = np.load(feature_path)
        print(f"✅ Successfully loaded features")
    except Exception as e:
        print(f"❌ ERROR loading features: {str(e)}")
        sys.exit(1)
    
    # Verify shape
    success = verify_feature_shape(features, expected_patches, expected_dim, backbone)
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("✅ VERIFICATION PASSED")
        print(f"\nThese features are compatible with main.py --saved_feats mode")
        print(f"\nTo use these features, run:")
        if backbone == 'dinov2':
            saved_feats_arg = 'dinov2q'
        else:
            saved_feats_arg = 'clip'
        
        # Extract subj from path
        parts = feature_path.split('/')
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) == 2:
                subj = part
                break
        else:
            subj = '01'
        
        # Extract base directory
        output_dir = feature_path.split(backbone)[0].rstrip('/')
        
        print(f"  python main.py \\")
        print(f"    --subj {int(subj)} \\")
        print(f"    --saved_feats {saved_feats_arg} \\")
        print(f"    --saved_feats_dir {output_dir} \\")
        print(f"    --encoder_arch transformer \\")
        print(f"    --readout_res rois_all")
    else:
        print("❌ VERIFICATION FAILED")
        print(f"\nThese features may not be compatible with main.py --saved_feats mode")
        print(f"Please regenerate features using extract_features.py")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
