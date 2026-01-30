import os
import subprocess
import time
import hashlib
from dotenv import load_dotenv

def setup():
    load_dotenv()
    token = os.getenv('NGROK_TOKEN')
    user = os.getenv('OLLAMA_USER', 'admin')
    raw_pass = os.getenv('OLLAMA_PASS', 'your-password-here')

    print("1. Installing Dependencies...")
    os.system("sudo apt-get update > /dev/null")
    os.system("sudo apt-get install zstd jq -y > /dev/null")

    print("2. Installing Ollama & Ngrok...")
    os.system("curl -fsSL https://ollama.com/install.sh | sh")
    os.system("curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null")
    os.system('echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list > /dev/null')
    os.system("sudo apt update > /dev/null && sudo apt install ngrok -y > /dev/null")

    print("\n🔍 Checking Available GPUs...")
    gpu_check = subprocess.getoutput("nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader")
    print(gpu_check)
    
    print("\n3. Starting Ollama Server with Multi-GPU Support...")
    os.environ['OLLAMA_HOST'] = '0.0.0.0'
    os.environ['OLLAMA_ORIGINS'] = '*'
    
    # Multi-GPU Configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Make both GPUs visible
    os.environ['OLLAMA_NUM_GPU'] = '2'          # Use 2 GPUs
    os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'  # Load 1 model across GPUs
    
    # CRITICAL: Force 100% GPU usage, 0% system RAM
    os.environ['OLLAMA_GPU_OVERHEAD'] = '0'     # No GPU memory reserved for overhead
    os.environ['OLLAMA_MAX_VRAM'] = '30000'     # Use full 30GB VRAM (2x15GB T4s)
    
    # Performance tuning for T4s
    os.environ['OLLAMA_NUM_PARALLEL'] = '2'     # Parallel requests
    os.environ['OLLAMA_MAX_QUEUE'] = '512'      # Request queue size
    
    os.system("pkill -9 ollama || true")
    time.sleep(2)
    
    # Start Ollama with explicit logging
    os.system("nohup ollama serve > ollama.log 2>&1 &")
    
    # Wait for Ollama to start
    print("Waiting for Ollama to wake up...")
    for i in range(15):
        if "127.0.0.1:11434" in subprocess.getoutput("netstat -tuln"):
            print("✅ Ollama is listening.")
            break
        time.sleep(2)
        if i % 3 == 0:
            print(f"   Still waiting... ({i*2}s)")
    
    print("\n4. Checking Model: gpt-oss:20b")
    models = subprocess.getoutput("ollama list")
    if "gpt-oss:20b" not in models:
        print("Model not found. Pulling now (this takes time)...")
        os.system("ollama pull gpt-oss:20b")
    else:
        print("✅ Model already exists.")
    
    # Preload model with GPU-only settings to verify it fits
    print("\n📊 Preloading model with GPU-only mode (reduced context)...")
    preload_cmd = '''curl -s http://localhost:11434/api/generate -d '{
        "model":"gpt-oss:20b",
        "prompt":"test",
        "stream":false,
        "options": {
            "num_ctx": 2048,
            "num_gpu": 2,
            "num_thread": 4
        }
    }' > /dev/null 2>&1 &'''
    os.system(preload_cmd)
    time.sleep(8)
    
    print("\n🖥️  GPU Memory Usage After Model Load:")
    os.system("nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv")
    
    print("\n5. Starting Ngrok Tunnel...")
    hashed_pass = hashlib.sha256(raw_pass.encode()).hexdigest()
    os.system("pkill -9 ngrok || true")
    os.system(f"nohup ngrok http 11434 --authtoken {token} --basic-auth {user}:{hashed_pass} > ngrok.log 2>&1 &")
    
    # Wait for tunnel URL
    print("Waiting for URL...")
    url = ""
    for _ in range(30):
        time.sleep(2)
        url = subprocess.getoutput("curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url'")
        if url and url != "null": 
            break
            
    if not url or url == "null":
        print("❌ Ngrok Error. Check ngrok.log")
        return

    print("\n" + "="*60)
    print(f"🚀 SERVER LIVE - DUAL GPU MODE")
    print(f"MODEL: gpt-oss:20b")
    print(f"URL:   {url}")
    print("="*60)
    print(f'\n# Copy this line to run on Windows PowerShell:')
    print(f'$env:OLLAMA_HOST="{url}"; $env:OLLAMA_USERNAME="{user}"; $env:OLLAMA_PASSWORD="{hashed_pass}"; python chat.py')
    print("="*60)
    
    # Monitoring loop
    print("\n📡 Server is running. Monitoring GPU usage every 10 minutes...")
    iteration = 0
    while True:
        time.sleep(600)  # 10 minutes
        iteration += 1
        print(f"\n[Heartbeat #{iteration}] {time.strftime('%H:%M:%S')}")
        print("GPU Status:")
        os.system("nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits")

if __name__ == "__main__":
    setup()