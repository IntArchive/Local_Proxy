from flask import Flask, request, Response, stream_with_context
import requests
import os
import json

app = Flask(__name__)

# --- CONFIGURATION ---
# Thay bằng URL Ngrok từ Colab output
REMOTE_URL = os.getenv("OLLAMA_HOST", "https://your-ngrok-url.ngrok-free.app")
OLLAMA_USER = os.getenv("OLLAMA_USERNAME", "admin")
OLLAMA_PASS = os.getenv("OLLAMA_PASSWORD", "your-hashed-password-here")

# Headers để bypass ngrok browser warning
HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
AUTH = (OLLAMA_USER, OLLAMA_PASS)

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f"{REMOTE_URL}/{path}"
    method = request.method
    params = request.args
    
    print(f"DEBUG: Proxying {method} to {url}")

    try:
        # Parse và modify request payload
        payload = request.get_json(force=True, silent=True) or {}
        
        # LOGGING: Ghi log request để debug
        with open("vscode_requests.log", "a", encoding="utf-8") as log_file:
            log_file.write(f"\n--- REQUEST ({method}) ---\n")
            log_file.write(json.dumps(payload, indent=2))
            log_file.write("\n-------------------------\n")
        
        # FORCE MODEL: Luôn dùng gpt-oss:20b
        print(f"🛠️ Forcing Model: gpt-oss:20b")
        payload["model"] = "gpt-oss:20b"

        # CONTEXT INJECTION: Tối ưu memory
        if "options" not in payload:
            payload["options"] = {}
        payload["options"]["num_ctx"] = 2048      # Context window
        payload["options"]["num_predict"] = -1    # Unlimited generation
        payload["num_ctx"] = 8192                 # Fallback top-level
        
        # Forward request với authentication
        resp = requests.request(
            method=method,
            url=url,
            headers=HEADERS,
            json=payload,
            params=params,
            auth=AUTH,
            stream=True,
            timeout=300
        )

        # Loại bỏ hop-by-hop headers
        excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
        headers = [(name, value) for (name, value) in resp.raw.headers.items()
                   if name.lower() not in excluded_headers]

        return Response(
            stream_with_context(resp.iter_content(chunk_size=1024)), 
            status=resp.status_code, 
            headers=headers
        )

    except Exception as e:
        print(f"ERROR: {e}")
        return Response(json.dumps({"error": str(e)}), status=500)

if __name__ == '__main__':
    print("🚀 VS Code → Ollama Proxy Running")
    print(f"Targeting: {REMOTE_URL}")
    print("━" * 50)
    print("📝 Trong VSCode Continue, set API Base URL:")
    print("   http://localhost:11435")
    print("━" * 50)
    app.run(port=11435, threaded=True)
