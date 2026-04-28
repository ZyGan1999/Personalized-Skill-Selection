#!/usr/bin/env python3
"""
test_packy.py
Quick check: can we reach packyapi.com and call various models?

Usage:
    OPENAI_API_KEY=sk-... python test_packy.py
    OPENAI_API_KEY=sk-... python test_packy.py --models gpt-5.2 glm-5 kimi-k2.5
"""

from __future__ import annotations

import argparse
import os
import sys
import time


DEFAULT_BASE = "https://www.packyapi.com"
DEFAULT_MODELS = [
    "glm-5",
    "kimi-k2.5",
    "minimax-m2.5",
    "qwen3-max",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default=DEFAULT_BASE,
                        help="API base URL (default: https://api.packyapi.com/v1)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Model names to test")
    parser.add_argument("--prompt", default='Reply with only this JSON: {"ok": true}',
                        help="Test prompt")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    # Configure for packy via env vars (agents._llm_call reads these)
    os.environ["OPENAI_API_BASE"] = args.api_base
    # Make sure no cached chat URL from earlier runs
    os.environ.pop("OPENAI_CHAT_URL", None)

    print(f"API base: {args.api_base}")
    print(f"Testing {len(args.models)} model(s): {', '.join(args.models)}")
    print("=" * 70)

    import requests

    api_key = os.environ["OPENAI_API_KEY"]
    base = args.api_base.rstrip("/")

    # Try a few common path patterns
    candidate_urls = [
        f"{base}/chat/completions",
        f"{base}/v1/chat/completions",
    ]

    results = []
    for m in args.models:
        print(f"\n[{m}]")
        for url in candidate_urls:
            print(f"  POST {url}", flush=True)
            payload = {
                "model": m,
                "messages": [{"role": "user", "content": args.prompt}],
                "temperature": 0.0,
                "max_tokens": 64,
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            t0 = time.time()
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                elapsed = time.time() - t0
                print(f"    HTTP {r.status_code} ({elapsed:.1f}s)")
                print(f"    Content-Type: {r.headers.get('Content-Type', 'unknown')}")
                body = r.text[:500]
                print(f"    Body[:500]: {body!r}")
                if r.ok:
                    try:
                        data = r.json()
                        msg = data["choices"][0]["message"]["content"]
                        print(f"    Parsed response: {msg[:200]}")
                        results.append((m, url, "OK", elapsed, msg[:80]))
                        break  # success, no need to try other URLs
                    except Exception as e:
                        print(f"    Failed to parse OpenAI-style JSON: {e}")
                        results.append((m, url, "PARSE_FAIL", elapsed, body[:80]))
                else:
                    results.append((m, url, f"HTTP_{r.status_code}", elapsed, body[:80]))
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    EXCEPTION ({elapsed:.1f}s): {type(e).__name__}: {e}")
                results.append((m, url, "EXCEPTION", elapsed, str(e)[:80]))

    print("\n" + "=" * 70)
    print("Summary:")
    for m, url, status, t, msg in results:
        marker = "✓" if status == "OK" else "✗"
        path = url.replace(base, "")
        print(f"  {marker} {m:<20} {path:<25} {status:<12} ({t:.1f}s)  {msg}")


if __name__ == "__main__":
    main()
