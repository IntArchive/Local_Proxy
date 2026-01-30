#!/usr/bin/env python3
"""
GPU Diagnostics for Ollama Multi-GPU Setup
Run this AFTER starting server_app.py to verify GPU distribution
"""

import subprocess
import time
import requests
import json

def run_command(cmd):
    """Run shell command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

def check_gpus():
    """Check available GPUs"""
    print("\n" + "="*70)
    print("🔍 GPU DETECTION")
    print("="*70)
    
    gpu_info = run_command("nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv")
    print(gpu_info)
    
    gpu_count = run_command("nvidia-smi --query-gpu=count --format=csv,noheader | head -1")
    print(f"\n✅ Total GPUs detected: {gpu_count}")

def check_ollama_process():
    """Check if Ollama is using both GPUs"""
    print("\n" + "="*70)
    print("🔧 OLLAMA PROCESS CHECK")
    print("="*70)
    
    # Check if Ollama is running
    ollama_pid = run_command("pgrep -f 'ollama serve'")
    if ollama_pid:
        print(f"✅ Ollama PID: {ollama_pid}")
        
        # Check CUDA_VISIBLE_DEVICES
        cuda_devices = run_command(f"cat /proc/{ollama_pid}/environ | tr '\\0' '\\n' | grep CUDA_VISIBLE_DEVICES")
        print(f"   {cuda_devices if cuda_devices else '⚠️  CUDA_VISIBLE_DEVICES not set'}")
        
        # Check GPU utilization
        print("\n📊 GPU Utilization:")
        gpu_util = run_command("nvidia-smi --query-compute-apps=pid,used_memory --format=csv")
        print(gpu_util)
    else:
        print("❌ Ollama is not running!")

def test_inference():
    """Run test inference and monitor GPU usage"""
    print("\n" + "="*70)
    print("🧪 INFERENCE TEST")
    print("="*70)
    
    print("Before inference:")
    before = run_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits")
    print(before)
    
    print("\n🔄 Running test inference (this may take 10-15 seconds)...")
    
    # Start inference in background
    test_prompt = "Explain quantum computing in one sentence."
    payload = {
        "model": "gpt-oss:20b",
        "prompt": test_prompt,
        "stream": False,
        "options": {
            "num_predict": 50
        }
    }
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Inference successful!")
            print(f"Response: {result.get('response', 'No response')[:100]}...")
        else:
            print(f"❌ Inference failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Inference error: {e}")
    
    print("\n⏱️  During/After inference:")
    time.sleep(2)
    after = run_command("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits")
    print(after)

def check_memory_distribution():
    """Analyze GPU memory distribution"""
    print("\n" + "="*70)
    print("📈 MEMORY DISTRIBUTION ANALYSIS")
    print("="*70)
    
    gpu_memory = run_command("nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits")
    lines = gpu_memory.split('\n')
    
    if len(lines) >= 2:
        gpu0_data = lines[0].split(',')
        gpu1_data = lines[1].split(',')
        
        gpu0_mem = int(gpu0_data[1].strip())
        gpu1_mem = int(gpu1_data[1].strip())
        
        print(f"GPU 0: {gpu0_mem} MB used")
        print(f"GPU 1: {gpu1_mem} MB used")
        
        total_mem = gpu0_mem + gpu1_mem
        if total_mem > 0:
            gpu0_pct = (gpu0_mem / total_mem) * 100
            gpu1_pct = (gpu1_mem / total_mem) * 100
            print(f"\nDistribution: GPU0={gpu0_pct:.1f}% | GPU1={gpu1_pct:.1f}%")
            
            # Check if both GPUs are being used
            if gpu0_mem > 100 and gpu1_mem > 100:
                print("✅ Both GPUs are loaded!")
            elif gpu0_mem > 100 and gpu1_mem < 100:
                print("⚠️  Only GPU 0 is loaded (single GPU mode)")
            elif gpu0_mem < 100 and gpu1_mem > 100:
                print("⚠️  Only GPU 1 is loaded (single GPU mode)")
            else:
                print("⚠️  Model not loaded yet")

def main():
    print("\n" + "="*70)
    print("🚀 OLLAMA MULTI-GPU DIAGNOSTIC TOOL")
    print("="*70)
    
    check_gpus()
    check_ollama_process()
    time.sleep(1)
    check_memory_distribution()
    test_inference()
    
    print("\n" + "="*70)
    print("📋 INTERPRETATION GUIDE")
    print("="*70)
    print("""
✅ GOOD SIGNS (Multi-GPU Working):
   - Both GPU 0 and GPU 1 show memory usage > 1000 MB
   - Memory distribution is roughly 50/50 or 60/40
   - Both GPUs show in compute apps list

⚠️  WARNING SIGNS (Single-GPU Mode):
   - Only GPU 0 has memory usage > 1000 MB
   - GPU 1 shows < 100 MB usage
   - CUDA_VISIBLE_DEVICES not showing "0,1"

🔧 FIXES IF SINGLE-GPU:
   1. Check ollama.log for errors
   2. Restart Ollama: pkill ollama && ollama serve
   3. Verify: echo $CUDA_VISIBLE_DEVICES (should be "0,1")
   4. Try: export CUDA_VISIBLE_DEVICES=0,1 before running
    """)
    print("="*70)

if __name__ == "__main__":
    main()
