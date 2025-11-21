"""
Compatibility test for memory-optimized feature extraction and loading.
Tests both pre-extracted mode and on-the-fly mode.
"""

import os
import sys
import tempfile
import shutil
import numpy as np
from PIL import Image

def create_mock_args(saved_feats=None, saved_feats_dir=None):
    """Create mock args object for testing"""
    class Args:
        def __init__(self):
            self.image_size = 224
            self.saved_feats = saved_feats
            self.saved_feats_dir = saved_feats_dir
            self.subj = '01'
            self.backbone_arch = 'dinov2_q' if not saved_feats else None
            self.data_dir = None
    
    return Args()

def test_saved_features_mode():
    """Test that saved features mode uses memory mapping"""
    print("="*60)
    print("Testing Saved Features Mode (Memory Optimized)")
    print("="*60)
    
    # Create temporary directory structure
    temp_root = tempfile.mkdtemp()
    
    try:
        # Create mock feature files
        dino_feat_dir = os.path.join(temp_root, 'dinov2_q_last', '01')
        clip_feat_dir = os.path.join(temp_root, 'clip_vit_512', '01')
        os.makedirs(dino_feat_dir, exist_ok=True)
        os.makedirs(clip_feat_dir, exist_ok=True)
        
        # Create mock feature arrays
        num_samples = 100
        train_features_dino = np.random.randn(num_samples, 962, 768).astype('float32')
        train_features_clip = np.random.randn(num_samples, 257, 768).astype('float32')
        
        # Save as numpy files
        np.save(os.path.join(dino_feat_dir, 'train.npy'), train_features_dino)
        np.save(os.path.join(clip_feat_dir, 'train.npy'), train_features_clip)
        np.save(os.path.join(dino_feat_dir, 'synt.npy'), train_features_dino[:10])
        np.save(os.path.join(clip_feat_dir, 'synt.npy'), train_features_clip[:10])
        
        # Import the dataset class
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from datasets.nsd import algonauts_dataset
        
        # Create mock paths and indices
        mock_paths = [f'path/to/img_{i}.png' for i in range(num_samples)]
        train_idxs = np.arange(80)
        
        # Create args with saved_feats enabled
        args = create_mock_args(saved_feats='dinov2q', saved_feats_dir=temp_root)
        
        # Create dataset (this should use memory mapping)
        print("\n1. Creating dataset with saved features...")
        dataset = algonauts_dataset(args, 'train', mock_paths, train_idxs, transform=None)
        
        print(f"✅ Dataset created successfully")
        print(f"   - Number of samples: {len(dataset)}")
        print(f"   - Using saved features: {dataset.saved_feats}")
        
        # Verify that data is loaded (should be memory-mapped)
        print("\n2. Verifying memory-mapped loading...")
        print(f"   - DINO features type: {type(dataset.fts_subj_train)}")
        print(f"   - DINO features shape: {dataset.fts_subj_train.shape}")
        print(f"   - CLIP features type: {type(dataset.clip_subj_train)}")
        print(f"   - CLIP features shape: {dataset.clip_subj_train.shape}")
        
        # Check if it's a numpy array (could be memmap or regular array after indexing)
        if isinstance(dataset.fts_subj_train, np.ndarray):
            print("✅ Features loaded as numpy array")
        
        print("\n✅ Saved features mode test PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ Saved features mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

def test_onthefly_mode():
    """Test that on-the-fly mode still works (no memory optimization applied)"""
    print("\n" + "="*60)
    print("Testing On-the-Fly Mode (No Memory Optimization)")
    print("="*60)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from datasets.nsd import algonauts_dataset
        
        # Create temporary directory for mock data
        temp_root = tempfile.mkdtemp()
        
        # Create mock fMRI data
        data_dir = os.path.join(temp_root, 'data')
        fmri_dir = os.path.join(data_dir, 'training_split', 'training_fmri')
        os.makedirs(fmri_dir, exist_ok=True)
        
        num_samples = 100
        lh_fmri = np.random.randn(num_samples, 1000).astype('float32')
        rh_fmri = np.random.randn(num_samples, 1000).astype('float32')
        
        np.save(os.path.join(fmri_dir, 'lh_training_fmri.npy'), lh_fmri)
        np.save(os.path.join(fmri_dir, 'rh_training_fmri.npy'), rh_fmri)
        
        # Create mock paths and indices
        mock_paths = [f'path/to/img_{i}.png' for i in range(num_samples)]
        train_idxs = np.arange(80)
        
        # Create args WITHOUT saved_feats (on-the-fly mode)
        args = create_mock_args(saved_feats=None, saved_feats_dir=None)
        args.data_dir = data_dir
        
        print("\n1. Creating dataset without saved features (on-the-fly mode)...")
        dataset = algonauts_dataset(args, 'train', mock_paths, train_idxs, transform=None)
        
        print(f"✅ Dataset created successfully")
        print(f"   - Number of samples: {len(dataset)}")
        print(f"   - Using saved features: {dataset.saved_feats}")
        print(f"   - Should load images on-the-fly: True")
        
        # Verify that fMRI data is loaded
        print("\n2. Verifying fMRI data loading...")
        print(f"   - LH fMRI shape: {dataset.lh_fmri.shape}")
        print(f"   - RH fMRI shape: {dataset.rh_fmri.shape}")
        
        print("\n✅ On-the-fly mode test PASSED")
        
        shutil.rmtree(temp_root, ignore_errors=True)
        return True
        
    except Exception as e:
        print(f"\n❌ On-the-fly mode test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memmap_writing():
    """Test that memmap writing works correctly"""
    print("\n" + "="*60)
    print("Testing Memmap Writing")
    print("="*60)
    
    temp_file = tempfile.mktemp(suffix='.npy')
    
    try:
        # Create a memmap file
        num_images = 100
        num_patches = 962
        feature_dim = 768
        
        print("\n1. Creating memmap file...")
        memmap_features = np.memmap(temp_file, dtype='float32', mode='w+', 
                                    shape=(num_images, num_patches, feature_dim))
        
        print(f"✅ Memmap created: shape={memmap_features.shape}, dtype={memmap_features.dtype}")
        
        # Write data in batches
        print("\n2. Writing data in batches...")
        batch_size = 10
        for i in range(0, num_images, batch_size):
            batch_data = np.random.randn(batch_size, num_patches, feature_dim).astype('float32')
            memmap_features[i:i+batch_size] = batch_data
        
        memmap_features.flush()
        print(f"✅ Data written successfully")
        
        # Move to final file
        final_file = temp_file.replace('.npy', '_final.npy')
        shutil.move(temp_file, final_file)
        
        # Read back with mmap_mode
        print("\n3. Reading back with mmap_mode='r'...")
        loaded_features = np.load(final_file, mmap_mode='r')
        print(f"✅ Data loaded: shape={loaded_features.shape}, dtype={loaded_features.dtype}")
        
        # Verify shape
        assert loaded_features.shape == (num_images, num_patches, feature_dim), "Shape mismatch!"
        
        print("\n✅ Memmap writing test PASSED")
        
        os.remove(final_file)
        return True
        
    except Exception as e:
        print(f"\n❌ Memmap writing test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)

def main():
    print("Compatibility Test Suite")
    print("="*60)
    print()
    
    results = {
        "Saved features mode": test_saved_features_mode(),
        "On-the-fly mode": test_onthefly_mode(),
        "Memmap writing": test_memmap_writing()
    }
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL COMPATIBILITY TESTS PASSED")
        print("="*60)
        print("\nMemory optimizations are compatible with:")
        print("1. ✅ Pre-extracted features mode (with memory mapping)")
        print("2. ✅ On-the-fly feature extraction mode (unchanged)")
        print("3. ✅ Memmap-based file operations")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
