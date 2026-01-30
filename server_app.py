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

    print("\n3. Starting Ollama Server...")
    os.environ['OLLAMA_HOST'] = '0.0.0.0'  # Allow external connections
    os.environ['OLLAMA_ORIGINS'] = '*'     # CORS for all origins
    os.environ['OLLAMA_NUM_GPU'] = '2'
    os.system("pkill -9 ollama || true")
    os.system("nohup ollama serve > ollama.log 2>&1 &")
    
    # Wait for Ollama to start
    print("Waiting for Ollama to wake up...")
    for _ in range(10):
        if "127.0.0.1:11434" in subprocess.getoutput("netstat -tuln"):
            print("✅ Ollama is listening.")
            break
        time.sleep(2)

    print("\n4. Checking Model: gpt-oss:20b")
    models = subprocess.getoutput("ollama list")
    if "gpt-oss:20b" not in models:
        print("Model not found. Pulling now (this takes time)...")
        os.system("ollama pull gpt-oss:20b")  # ~12GB download
    else:
        print("✅ Model already exists.")

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
        if url and url != "null": break
            
    if not url or url == "null":
        print("❌ Ngrok Error. Check ngrok.log")
        return

    print("\n" + "="*50)
    print(f"🚀 SERVER LIVE")
    print(f"MODEL: gpt-oss:20b")
    print(f"URL:   {url}")
    print("="*50)
    print(f'\n# Copy dòng này để chạy trên Windows PowerShell:')
    print(f'$env:OLLAMA_HOST="{url}"; $env:OLLAMA_USERNAME="{user}"; $env:OLLAMA_PASSWORD="{hashed_pass}"; python chat.py')
    print("="*50)

    # Keep alive
    while True:
        time.sleep(600)
        print(f"Heartbeat: {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    setup()