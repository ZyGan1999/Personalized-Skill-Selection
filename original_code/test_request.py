import os, requests
api_key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_BASE")

print("Testing API key:", "Present" if api_key else "Missing")
print("Testing API endpoint:", api_base)
print("Using model:", os.getenv("AGENT_MODEL", "gpt-4o-mini"))


r = requests.post(
    f"{api_base}/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json={"model": "claude-sonnet-4-6", "messages": [{"role": "user", "content": "hi"}]},
)
print(r.status_code, r.text)