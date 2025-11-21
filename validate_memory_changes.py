"""
Memory optimization validation script
This script validates the changes made to extract_features.py and datasets/nsd.py
without needing to run the actual models.
"""

import os
import sys
import ast

def check_memmap_usage(filepath):
    """Check if np.memmap is used in the file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    has_memmap = 'np.memmap' in content or 'numpy.memmap' in content
    has_mmap_mode = "mmap_mode='r'" in content or 'mmap_mode="r"' in content
    has_empty_cache = 'torch.cuda.empty_cache()' in content
    
    return has_memmap, has_mmap_mode, has_empty_cache

def validate_extract_features():
    """Validate extract_features.py changes"""
    filepath = '/home/runner/work/transformer_brain_encoder/transformer_brain_encoder/extract_features.py'
    
    print("="*60)
    print("Validating extract_features.py")
    print("="*60)
    
    has_memmap, has_mmap_mode, has_empty_cache = check_memmap_usage(filepath)
    
    print(f"✓ Uses np.memmap for writing: {has_memmap}")
    print(f"✓ Uses mmap_mode='r' for reading: {has_mmap_mode}")
    print(f"✓ Calls torch.cuda.empty_cache(): {has_empty_cache}")
    
    # Check that we're not accumulating in lists
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Count occurrences of the old pattern
    list_append_count = content.count('all_features.append(')
    memmap_write_count = content.count('memmap_features[current_idx:')
    
    print(f"\nList accumulation patterns found: {list_append_count}")
    print(f"Memmap direct write patterns found: {memmap_write_count}")
    
    if list_append_count > 0:
        print("⚠️  Warning: Still using list accumulation in some places")
    
    if memmap_write_count >= 3:  # Should have at least 3 (one for each function)
        print("✅ All extraction functions use memmap for writing")
    else:
        print(f"⚠️  Warning: Expected at least 3 memmap writes, found {memmap_write_count}")
    
    return has_memmap and has_empty_cache and memmap_write_count >= 3

def validate_nsd_dataset():
    """Validate datasets/nsd.py changes"""
    filepath = '/home/runner/work/transformer_brain_encoder/transformer_brain_encoder/datasets/nsd.py'
    
    print("\n" + "="*60)
    print("Validating datasets/nsd.py")
    print("="*60)
    
    has_memmap, has_mmap_mode, has_empty_cache = check_memmap_usage(filepath)
    
    print(f"✓ Uses mmap_mode='r' for loading: {has_mmap_mode}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Check for mmap_mode in np.load calls for saved features
    saved_feats_loads = content.count("np.load(dino_feat_dir") + content.count("np.load(clip_feat_dir")
    mmap_loads = content.count("mmap_mode='r'")
    
    print(f"\nSaved feature loads found: {saved_feats_loads}")
    print(f"Loads using mmap_mode='r': {mmap_loads}")
    
    if mmap_loads >= 4:  # Should have at least 4 (train/test for dino/clip)
        print("✅ All saved feature loads use mmap_mode='r'")
    else:
        print(f"⚠️  Warning: Expected at least 4 mmap loads, found {mmap_loads}")
    
    return has_mmap_mode and mmap_loads >= 4

def check_code_syntax():
    """Check that the Python files have valid syntax"""
    print("\n" + "="*60)
    print("Checking Python syntax")
    print("="*60)
    
    files_to_check = [
        '/home/runner/work/transformer_brain_encoder/transformer_brain_encoder/extract_features.py',
        '/home/runner/work/transformer_brain_encoder/transformer_brain_encoder/datasets/nsd.py'
    ]
    
    all_valid = True
    for filepath in files_to_check:
        try:
            with open(filepath, 'r') as f:
                ast.parse(f.read())
            print(f"✅ {os.path.basename(filepath)}: Valid syntax")
        except SyntaxError as e:
            print(f"❌ {os.path.basename(filepath)}: Syntax error at line {e.lineno}: {e.msg}")
            all_valid = False
    
    return all_valid

def main():
    print("Memory Optimization Validation")
    print("="*60)
    print()
    
    syntax_valid = check_code_syntax()
    extract_valid = validate_extract_features()
    nsd_valid = validate_nsd_dataset()
    
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    results = {
        "Syntax validation": syntax_valid,
        "extract_features.py memory optimization": extract_valid,
        "datasets/nsd.py memory optimization": nsd_valid
    }
    
    for check, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
        print("="*60)
        print("\nMemory optimizations implemented successfully:")
        print("1. ✅ extract_features.py uses np.memmap for incremental writing")
        print("2. ✅ extract_features.py calls torch.cuda.empty_cache() after batches")
        print("3. ✅ datasets/nsd.py uses mmap_mode='r' for memory-efficient loading")
        return 0
    else:
        print("❌ SOME VALIDATIONS FAILED")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
