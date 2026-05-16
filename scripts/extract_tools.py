#!/usr/bin/env python3
"""
extract_tools.py
Extract tool descriptions from ToolBench data and build a benchmark config JSON.

Usage:
    python extract_tools.py --toolbench-dir ~/ToolBench/data/toolenv/tools \
                            --n-categories 10 --n-tools-per-category 10 \
                            --output benchmark_data/toolbench_100.json

The output JSON can be used with: python main.py --benchmark benchmark_data/toolbench_100.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_toolbench_tools(toolbench_dir: str) -> Dict[str, List[dict]]:
    """
    Load all tool JSON files from ToolBench's data/toolenv/tools/ directory.

    Directory structure: tools/{Category}/{tool_name}.json
    Each JSON has: tool_name, tool_description, title, api_list, ...

    Returns: {category: [tool_dict, ...]}
    """
    tools_by_category: Dict[str, List[dict]] = {}
    root = Path(toolbench_dir)

    if not root.exists():
        raise FileNotFoundError(f"ToolBench tools directory not found: {root}")

    for category_dir in sorted(root.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith('.'):
            continue
        category = category_dir.name
        tools = []
        for json_file in sorted(category_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                tool_name = data.get("tool_name", json_file.stem)
                description = data.get("tool_description", "")
                if tool_name and description and len(description) > 10:
                    tools.append({
                        "name": tool_name,
                        "description": description.strip(),
                        "title": data.get("title", tool_name),
                        "n_apis": len(data.get("api_list", [])),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
        if tools:
            tools_by_category[category] = tools

    return tools_by_category


def select_tools(
    tools_by_category: Dict[str, List[dict]],
    n_categories: int = 10,
    n_tools_per_category: int = 10,
    min_tools_in_category: int = 10,
) -> Dict[str, List[dict]]:
    """
    Select top categories and tools for the benchmark.

    Selection criteria:
    - Categories with >= min_tools_in_category tools
    - Top n_categories by tool count
    - Within each category, select tools with best description quality
      (measured by description length as a proxy for informativeness)
    """
    # Filter categories with enough tools
    eligible = {
        cat: tools for cat, tools in tools_by_category.items()
        if len(tools) >= min_tools_in_category
    }

    # Sort by number of tools (descending), take top n
    sorted_cats = sorted(eligible.keys(), key=lambda c: len(eligible[c]), reverse=True)
    selected_cats = sorted_cats[:n_categories]

    selected: Dict[str, List[dict]] = {}
    for cat in selected_cats:
        tools = eligible[cat]
        # Sort by description quality: prefer medium-length descriptions (50-300 chars)
        # and tools with more APIs (more functional)
        scored = []
        for t in tools:
            desc_len = len(t["description"])
            # Penalize very short or very long descriptions
            len_score = min(desc_len, 300) / 300.0 - max(0, desc_len - 300) / 1000.0
            api_score = min(t["n_apis"], 10) / 10.0
            scored.append((len_score + api_score, t))
        scored.sort(key=lambda x: -x[0])
        selected[cat] = [t for _, t in scored[:n_tools_per_category]]

    return selected


def build_config(
    selected: Dict[str, List[dict]],
    name: str = "toolbench",
) -> dict:
    """
    Build the benchmark config JSON structure.
    Note: standard_queries and ood_queries are left empty — run generate_queries.py to fill them.
    """
    domains = {}
    tool_metadata = {}

    for category, tools in selected.items():
        # Clean category name: replace underscores with spaces
        domain_name = category.replace("_", " ")
        tool_names = [t["name"] for t in tools]
        domains[domain_name] = tool_names

        for t in tools:
            tool_metadata[t["name"]] = t["description"]

    config = {
        "name": name,
        "description": f"{sum(len(v) for v in domains.values())} tools across {len(domains)} categories from ToolBench/RapidAPI",
        "domains": domains,
        "tool_metadata": tool_metadata,
        "standard_queries": {domain: [] for domain in domains},
        "ood_queries": {domain: {tool: [] for tool in tools} for domain, tools in domains.items()},
    }

    return config


def main():
    parser = argparse.ArgumentParser(description="Extract tools from ToolBench data")
    parser.add_argument(
        "--toolbench-dir", required=True,
        help="Path to ToolBench's data/toolenv/tools/ directory",
    )
    parser.add_argument("--n-categories", type=int, default=10)
    parser.add_argument("--n-tools-per-category", type=int, default=10)
    parser.add_argument(
        "--output", default="benchmark_data/toolbench_100.json",
        help="Output JSON config path",
    )
    parser.add_argument("--name", default="toolbench_100")
    args = parser.parse_args()

    print(f"Loading tools from {args.toolbench_dir} ...")
    all_tools = load_toolbench_tools(args.toolbench_dir)
    print(f"Found {len(all_tools)} categories, {sum(len(v) for v in all_tools.values())} tools total")

    # Show category stats
    for cat in sorted(all_tools, key=lambda c: len(all_tools[c]), reverse=True)[:20]:
        print(f"  {cat}: {len(all_tools[cat])} tools")

    print(f"\nSelecting {args.n_categories} categories x {args.n_tools_per_category} tools ...")
    selected = select_tools(all_tools, args.n_categories, args.n_tools_per_category)

    n_total = sum(len(v) for v in selected.values())
    print(f"Selected {len(selected)} categories, {n_total} tools:")
    for cat, tools in selected.items():
        print(f"  {cat}: {', '.join(t['name'] for t in tools[:3])}{'...' if len(tools) > 3 else ''}")

    config = build_config(selected, name=args.name)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\nConfig saved to {args.output}")
    print(f"NOTE: standard_queries and ood_queries are empty. Run generate_queries.py to fill them.")


if __name__ == "__main__":
    main()
