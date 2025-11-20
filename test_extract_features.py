"""
Test script for extract_features_new.py
Creates synthetic test data and verifies feature extraction works correctly
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil

def create_test_images(output_dir, num_images=10):
    """Create synthetic test images"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(num_images):
        # Create random RGB image
        img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img_pil = Image.fromarray(img)
        img_path = os.path.join(output_dir, f'test_img_{i:05d}.png')
        img_pil.save(img_path)
    
    print(f"✅ Created {num_images} test images in {output_dir}")


def test_feature_extraction():
    """Test feature extraction with synthetic data"""
    
    print("="*60)
    print("Testing Feature Extraction Script")
    print("="*60)
    
    # Create temporary directories
    temp_root = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_root}")
    
    try:
        # Create directory structure
        data_dir = os.path.join(temp_root, 'test_data', 'subj01')
        train_img_dir = os.path.join(data_dir, 'training_split', 'training_images')
        test_img_dir = os.path.join(data_dir, 'test_split', 'test_images')
        output_dir = os.path.join(temp_root, 'features')
        
        # Create test images
        print("\n1. Creating test images...")
        create_test_images(train_img_dir, num_images=5)
        create_test_images(test_img_dir, num_images=3)
        
        # Import the extraction script
        print("\n2. Importing extraction module...")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import extract_features_new as extract_features
        
        # Test each backbone
        backbones = ['dinov2_q', 'dinov2', 'clip']
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        results = {}
        
        for backbone in backbones:
            print(f"\n{'='*60}")
            print(f"3. Testing {backbone} backbone")
            print(f"{'='*60}")
            
            try:
                # Set output paths
                if backbone == 'dinov2_q':
                    output_subdir = 'dinov2_q_last'
                elif backbone == 'dinov2':
                    output_subdir = 'dinov2_last'
                elif backbone == 'clip':
                    output_subdir = 'clip_vit_512'
                
                output_subject_dir = os.path.join(output_dir, output_subdir, '01')
                train_output_path = os.path.join(output_subject_dir, 'train.npy')
                test_output_path = os.path.join(output_subject_dir, 'synt.npy')
                
                # Select extraction function
                if backbone == 'dinov2_q':
                    extract_fn = extract_features.extract_dino_features_with_hooks
                elif backbone == 'dinov2':
                    extract_fn = extract_features.extract_dino_features_simple
                elif backbone == 'clip':
                    extract_fn = extract_features.extract_clip_features
                
                # Extract features
                print(f"\n   Extracting training features...")
                train_features = extract_fn(
                    train_img_dir,
                    train_output_path,
                    enc_output_layer=-1,
                    batch_size=2,
                    device=device
                )
                
                print(f"\n   Extracting test features...")
                test_features = extract_fn(
                    test_img_dir,
                    test_output_path,
                    enc_output_layer=-1,
                    batch_size=2,
                    device=device
                )
                
                # Verify shapes
                print(f"\n   Verifying output shapes...")
                print(f"   Train features: {train_features.shape}")
                print(f"   Test features: {test_features.shape}")
                
                # Check expected shapes
                expected_train_samples = 5
                expected_test_samples = 3
                
                assert train_features.shape[0] == expected_train_samples, \
                    f"Expected {expected_train_samples} train samples, got {train_features.shape[0]}"
                assert test_features.shape[0] == expected_test_samples, \
                    f"Expected {expected_test_samples} test samples, got {test_features.shape[0]}"
                
                # Check feature dimensions
                if backbone in ['dinov2_q', 'dinov2']:
                    # DINOv2: [N, 962, 768] (961 patches + 1 CLS)
                    expected_dim = 768
                    expected_patches = 962  # 31*31 + 1 = 962
                elif backbone == 'clip':
                    # CLIP: [N, 257, 768] (256 patches + 1 CLS)
                    expected_dim = 768
                    expected_patches = 257  # 16*16 + 1 = 257
                
                assert train_features.shape[2] == expected_dim, \
                    f"Expected feature dim {expected_dim}, got {train_features.shape[2]}"
                assert train_features.shape[1] == expected_patches, \
                    f"Expected {expected_patches} patches, got {train_features.shape[1]}"
                
                print(f"   ✅ {backbone} extraction successful!")
                results[backbone] = 'PASS'
                
            except Exception as e:
                print(f"   ❌ {backbone} extraction failed: {str(e)}")
                results[backbone] = f'FAIL: {str(e)}'
                import traceback
                traceback.print_exc()
        
        # Print summary
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        for backbone, result in results.items():
            status = "✅" if result == "PASS" else "❌"
            print(f"{status} {backbone}: {result}")
        
        # Overall result
        all_passed = all(r == "PASS" for r in results.values())
        print(f"\n{'='*60}")
        if all_passed:
            print("✅ ALL TESTS PASSED!")
        else:
            print("❌ SOME TESTS FAILED")
        print(f"{'='*60}")
        
        return all_passed
        
    finally:
        # Cleanup
        print(f"\nCleaning up temporary directory: {temp_root}")
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == '__main__':
    success = test_feature_extraction()
    sys.exit(0 if success else 1)
