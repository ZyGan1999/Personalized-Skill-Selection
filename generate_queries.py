#!/usr/bin/env python3
"""
generate_queries.py
Generate standard and OOD queries for a benchmark config using an LLM.

Usage:
    python generate_queries.py --config benchmark_data/toolbench_100.json \
                               --model gpt-4o-mini \
                               --n-standard 20 --n-ood 5

Reads the config JSON (with domains + tool_metadata), generates queries via LLM,
and writes them back to the same JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, List


def _llm_call(prompt: str, model: str) -> str:
    """Call LLM and return response text. Tries litellm first, then openai."""
    try:
        import litellm
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=2000,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        pass

    import openai
    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_API_BASE", None),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=2000,
    )
    return resp.choices[0].message.content.strip()


def _parse_numbered_list(text: str) -> List[str]:
    """Extract items from a numbered list (1. xxx, 2. xxx, ...)."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # Match patterns like "1. query", "1) query", "- query"
        m = re.match(r'^(?:\d+[\.\)]\s*|[-*]\s*)(.*)', line)
        if m and m.group(1).strip():
            lines.append(m.group(1).strip().strip('"').strip("'"))
    return lines


def generate_standard_queries(
    domains: Dict[str, List[str]],
    tool_metadata: Dict[str, str],
    model: str,
    n_per_domain: int = 20,
) -> Dict[str, List[str]]:
    """Generate domain-specific but tool-agnostic queries."""
    queries: Dict[str, List[str]] = {}

    for domain, tools in domains.items():
        tool_descriptions = "\n".join(
            f"  - {t}: {tool_metadata.get(t, '')}" for t in tools
        )
        prompt = (
            f"Generate {n_per_domain} diverse user queries that someone might ask when "
            f"needing a tool/API from the \"{domain}\" category.\n\n"
            f"Tools in this category:\n{tool_descriptions}\n\n"
            f"IMPORTANT RULES:\n"
            f"- Do NOT mention any specific tool name in the queries\n"
            f"- Each query should be a natural user request (1-2 sentences)\n"
            f"- Cover diverse use cases within this category\n"
            f"- Write in English\n\n"
            f"Return a numbered list (1. query, 2. query, ...)."
        )

        print(f"  Generating standard queries for '{domain}' ...", end=" ", flush=True)
        try:
            response = _llm_call(prompt, model)
            parsed = _parse_numbered_list(response)
            queries[domain] = parsed[:n_per_domain]
            print(f"got {len(queries[domain])}")
        except Exception as e:
            print(f"ERROR: {e}")
            queries[domain] = []
        time.sleep(1)  # Rate limiting

    return queries


def generate_ood_queries(
    domains: Dict[str, List[str]],
    tool_metadata: Dict[str, str],
    model: str,
    n_per_tool: int = 5,
) -> Dict[str, Dict[str, List[str]]]:
    """Generate queries that explicitly name each tool (for OOD testing)."""
    ood: Dict[str, Dict[str, List[str]]] = {}

    for domain, tools in domains.items():
        ood[domain] = {}
        for tool in tools:
            desc = tool_metadata.get(tool, "")
            prompt = (
                f"Generate {n_per_tool} user queries that explicitly request using "
                f"\"{tool}\" (a {domain} tool: {desc}).\n\n"
                f"IMPORTANT RULES:\n"
                f"- Each query MUST mention \"{tool}\" by name\n"
                f"- The query should be a natural user request (1-2 sentences)\n"
                f"- Write in English\n\n"
                f"Return a numbered list."
            )

            try:
                response = _llm_call(prompt, model)
                parsed = _parse_numbered_list(response)
                ood[domain][tool] = parsed[:n_per_tool]
            except Exception as e:
                print(f"  ERROR generating OOD for {tool}: {e}")
                ood[domain][tool] = []
            time.sleep(0.5)

        n_total = sum(len(v) for v in ood[domain].values())
        print(f"  OOD queries for '{domain}': {n_total} across {len(tools)} tools")

    return ood


def main():
    parser = argparse.ArgumentParser(description="Generate queries for benchmark config")
    parser.add_argument("--config", required=True, help="Path to benchmark config JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model for generation")
    parser.add_argument("--n-standard", type=int, default=20, help="Standard queries per domain")
    parser.add_argument("--n-ood", type=int, default=5, help="OOD queries per tool")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing queries")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    domains = config["domains"]
    tool_metadata = config["tool_metadata"]
    n_tools = sum(len(v) for v in domains.values())
    print(f"Config: {len(domains)} domains, {n_tools} tools")

    # Generate standard queries
    existing_std = config.get("standard_queries", {})
    has_std = any(len(v) > 0 for v in existing_std.values())
    if has_std and not args.overwrite:
        print("Standard queries already exist. Use --overwrite to regenerate.")
    else:
        print(f"\nGenerating {args.n_standard} standard queries per domain ...")
        config["standard_queries"] = generate_standard_queries(
            domains, tool_metadata, args.model, args.n_standard
        )

    # Generate OOD queries
    existing_ood = config.get("ood_queries", {})
    has_ood = any(
        any(len(queries) > 0 for queries in tools.values())
        for tools in existing_ood.values()
    )
    if has_ood and not args.overwrite:
        print("OOD queries already exist. Use --overwrite to regenerate.")
    else:
        print(f"\nGenerating {args.n_ood} OOD queries per tool ...")
        config["ood_queries"] = generate_ood_queries(
            domains, tool_metadata, args.model, args.n_ood
        )

    # Save back
    with open(args.config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated config saved to {args.config}")

    # Summary
    n_std = sum(len(v) for v in config["standard_queries"].values())
    n_ood = sum(
        sum(len(q) for q in tools.values())
        for tools in config["ood_queries"].values()
    )
    print(f"Total: {n_std} standard queries, {n_ood} OOD queries")


if __name__ == "__main__":
    main()
