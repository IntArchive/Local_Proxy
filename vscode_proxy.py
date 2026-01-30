from flask import Flask, request, Response, stream_with_context
import requests
import os
import json
import time

app = Flask(__name__)

# --- CONFIGURATION ---
REMOTE_URL = os.getenv("OLLAMA_HOST", "https://your-ngrok-url.ngrok-free.app")
OLLAMA_USER = os.getenv("OLLAMA_USERNAME", "admin")
OLLAMA_PASS = os.getenv("OLLAMA_PASSWORD", "your-hashed-password-here")

# Headers to bypass ngrok browser warning
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
AUTH = (OLLAMA_USER, OLLAMA_PASS)

# Request counter for monitoring
request_count = 0

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    global request_count
    request_count += 1
    
    url = f"{REMOTE_URL}/{path}"
    method = request.method
    params = request.args
    
    print(f"\n[Request #{request_count}] {method} → {path}")

    try:
        # Parse request payload
        payload = request.get_json(force=True, silent=True) or {}
        
        # LOGGING: Track all requests for debugging
        with open("vscode_requests.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Request #{request_count}\n")
            log_file.write(f"Method: {method} | Path: {path}\n")
            log_file.write(f"Payload:\n{json.dumps(payload, indent=2)}\n")
            log_file.write(f"{'='*60}\n")
        
        # FORCE MODEL: Always use gpt-oss:20b
        if payload:
            original_model = payload.get("model", "not specified")
            payload["model"] = "gpt-oss:20b"
            print(f"   Model: {original_model} → gpt-oss:20b")
        
        # OPTIMIZATION: Multi-GPU friendly settings for Kaggle T4s
        if "options" not in payload:
            payload["options"] = {}
        
        # Reduced settings to fit in 2x T4 (15GB each) with low system RAM
        payload["options"].update({
            "num_ctx": 2048,          # Reduced context (was 4096) to avoid system RAM
            "num_predict": -1,        # Unlimited generation
            "num_thread": 4,          # Lower CPU threads (Kaggle has limited RAM)
            "num_gpu": 2,             # Explicit GPU count
            "num_batch": 256,         # Reduced batch size to save memory
        })
        
        # Fallback top-level settings
        payload["num_ctx"] = 2048
        
        print(f"   GPU Config: num_gpu=2, num_ctx=2048, num_batch=256 (RAM-optimized)")
        
        # Forward request with authentication
        start_time = time.time()
        resp = requests.request(
            method=method,
            url=url,
            headers=HEADERS,
            json=payload,
            params=params,
            auth=AUTH,
            stream=True,
            timeout=600  # Increased timeout for large responses
        )
        
        # Remove hop-by-hop headers
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]
        
        # Stream response with timing info
        def generate():
            chunk_count = 0
            for chunk in resp.iter_content(chunk_size=1024):
                chunk_count += 1
                yield chunk
            elapsed = time.time() - start_time
            print(f"   ✅ Completed in {elapsed:.2f}s ({chunk_count} chunks)")
        
        return Response(
            stream_with_context(generate()), 
            status=resp.status_code, 
            headers=headers
        )

    except requests.exceptions.Timeout:
        error_msg = "Request timeout - model may be overloaded or response too long"
        print(f"   ❌ {error_msg}")
        return Response(json.dumps({"error": error_msg}), status=504)
    
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Connection error - check if Ollama server is running: {str(e)}"
        print(f"   ❌ {error_msg}")
        return Response(json.dumps({"error": error_msg}), status=503)
    
    except Exception as e:
        error_msg = f"Proxy error: {str(e)}"
        print(f"   ❌ {error_msg}")
        return Response(json.dumps({"error": error_msg}), status=500)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        resp = requests.get(f"{REMOTE_URL}/api/tags", headers=HEADERS, auth=AUTH, timeout=5)
        return {
            "status": "healthy",
            "remote_url": REMOTE_URL,
            "requests_served": request_count,
            "remote_status": resp.status_code
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}, 503

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 VS Code → Ollama Proxy (Multi-GPU Optimized)")
    print(f"📡 Targeting: {REMOTE_URL}")
    print("="*60)
    print("\n📝 Setup Instructions:")
    print("   1. In VS Code, install 'Continue' extension")
    print("   2. Open Continue settings (JSON)")
    print("   3. Set API Base URL to: http://localhost:11435")
    print("   4. Model will auto-use: gpt-oss:20b on dual T4s")
    print("\n💡 Health check: http://localhost:11435/health")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=11435, threaded=True, debug=False)