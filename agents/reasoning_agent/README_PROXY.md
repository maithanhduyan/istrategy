# Together.ai to Ollama Proxy

Simple proxy server để sử dụng Together.ai API thông qua GitHub Copilot BYOK Ollama endpoint.

## Cách sử dụng:

1. **Cài đặt dependencies:**
```bash
pip install flask requests
```

2. **Set Together.ai API key:**
```bash
export TOGETHER_API_KEY=your_together_api_key
```

3. **Chạy proxy server:**
```bash
python together_ollama_proxy.py
```

4. **Cấu hình GitHub Copilot BYOK:**
- Đảm bảo `github.copilot.byokEnabled: true`
- Endpoint sẽ tự động sử dụng `http://127.0.0.1:11434`

## Hoạt động:

- Proxy nhận request từ GitHub Copilot ở format Ollama API
- Chuyển đổi sang Together.ai API format
- Gửi request đến Together.ai
- Chuyển đổi response về format Ollama
- Trả về cho GitHub Copilot

## Endpoints hỗ trợ:

- `/api/generate` - Text generation (chính)
- `/api/tags` - List models
- `/api/version` - Version info
- `/health` - Health check

## Models được map:

- `deepseek-ai/DeepSeek-V3` (recommended)
- `meta-llama/Llama-2-7b-chat-hf`

Với cách này, GitHub Copilot sẽ sử dụng Together.ai thay vì Ollama local.
