#!/usr/bin/env python
"""Verify GPU setup for PyTorch."""

import torch
import sys

def verify_gpu():
    print("="*60)
    print("GPU Verification for PyTorch")
    print("="*60)
    
    # PyTorch version
    print(f"\n✓ PyTorch Version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n{'✓' if cuda_available else '✗'} CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
        
        # GPU details
        gpu_count = torch.cuda.device_count()
        print(f"\n✓ Number of GPUs: {gpu_count}")
        
        for i in range(gpu_count):
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"    - Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"    - Compute Capability: {props.major}.{props.minor}")
        
        # Quick tensor test
        print("\n" + "="*60)
        print("Running Quick GPU Test...")
        print("="*60)
        
        try:
            # Create tensors
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            
            # Matrix multiplication
            z = torch.matmul(x, y)
            
            print(f"\n✓ GPU computation successful!")
            print(f"  Result shape: {z.shape}")
            print(f"  Result device: {z.device}")
            
            # Cleanup
            del x, y, z
            torch.cuda.empty_cache()
            
            return True
        except Exception as e:
            print(f"\n✗ GPU test failed: {e}")
            return False
    else:
        print("\n⚠ GPU not available. Training will use CPU.")
        print("\nPossible fixes:")
        print("1. Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        print("2. Update NVIDIA drivers")
        print("3. Verify CUDA installation: nvidia-smi")
        return False

if __name__ == "__main__":
    print()
    success = verify_gpu()
    print("\n" + "="*60)
    if success:
        print("✓ GPU Setup: READY")
    else:
        print("✗ GPU Setup: NOT READY")
    print("="*60 + "\n")
    sys.exit(0 if success else 1)
