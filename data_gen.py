"""
data_gen.py
Synthetic user and query generation for the personalized agent tool-selection benchmark.

Supports:
  - generate_users(n_users, seed): one-hot preferences (one preferred tool per domain)
  - generate_users_soft(n_users, seed, concentration): Dirichlet soft preferences

OOD queries explicitly name the target tool, ensuring the LLM can detect the
override signal — fixing the original template-mismatch bug.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Domains and Tools
# ---------------------------------------------------------------------------

DOMAINS: Dict[str, List[str]] = {
    "financial":      ["Tonghua Shun", "East Money", "Bloomberg", "Yahoo Finance"],
    "food_delivery":  ["Meituan", "Ele.me", "Taobao Delivery", "WeChat MiniProgram"],
    "navigation":     ["Amap", "Baidu Maps", "DiDi", "CaoCao"],
    "shopping":       ["Taobao", "JD.com", "Pinduoduo", "Amazon"],
    "entertainment":  ["Netflix", "Bilibili", "Youku", "iQiyi"],
}

# Flat list of all tools across all domains (used when agents see the full tool pool)
ALL_TOOLS: List[str] = [t for tools in DOMAINS.values() for t in tools]

# Short natural-language descriptions for each tool (metadata fed to LLM)
TOOL_METADATA: Dict[str, str] = {
    # Financial
    "Tonghua Shun":        "Chinese A-share stock data platform; real-time quotes, sector analysis, and CSI index charts.",
    "East Money":          "Chinese retail investor portal; mutual fund rankings, stock discussion boards, and fund screener.",
    "Bloomberg":           "Global financial data terminal; macro indicators, bond yields, FX rates, and institutional research.",
    "Yahoo Finance":       "English-language finance portal; international stock quotes, earnings calendars, and ETF comparisons.",
    # Food delivery
    "Meituan":             "Dominant Chinese food-delivery super-app; wide restaurant coverage, Meituan Points loyalty rewards.",
    "Ele.me":              "Alibaba-owned food delivery platform; deep Alipay and 88VIP integration, frequent cashback coupons.",
    "Taobao Delivery":     "Taobao/Alibaba food-delivery service; best for users with Taobao 88VIP membership discounts.",
    "WeChat MiniProgram":  "Food ordering via WeChat mini-programs; supports WeChat Pay red packets and corporate canteen orders.",
    # Navigation
    "Amap":                "Alibaba navigation app; best real-time traffic and road-closure updates for China road trips.",
    "Baidu Maps":          "Baidu navigation app; strong transit scheduling, indoor mall maps, and bus route planning.",
    "DiDi":                "China's leading ride-hailing service; on-demand taxis, Express, and carpooling options.",
    "CaoCao":              "New-energy-vehicle ride-hailing platform; corporate accounts, off-peak discounts, eco-friendly fleet.",
    # Shopping
    "Taobao":              "China's largest C2C/B2C marketplace; widest product variety, manufacturer flagship stores, customisation.",
    "JD.com":              "Chinese B2C retailer; authentic electronics, same-day/next-day delivery, official brand warranties.",
    "Pinduoduo":           "Social-commerce platform; lowest prices via group buying, daily government subsidies on everyday goods.",
    "Amazon":              "Global e-commerce platform; imported goods, Prime two-day delivery, international brand selection.",
    # Entertainment
    "Netflix":             "Global SVOD streaming service; Netflix original series and films, ad-free international content.",
    "Bilibili":            "Chinese video platform popular for anime, fan-subs, live streaming, and creator-uploaded content.",
    "Youku":               "Alibaba-owned streaming platform; exclusive Chinese drama and variety show broadcast rights.",
    "iQiyi":               "Baidu-owned streaming platform; premium VIP content, exclusive domestic drama premieres.",
}

# ---------------------------------------------------------------------------
# Standard query templates (15 per domain, domain-specific but tool-agnostic)
# ---------------------------------------------------------------------------

STANDARD_QUERIES: Dict[str, List[str]] = {
    "financial": [
        "Check today's stock market performance",
        "What is the latest price of AAPL?",
        "Show me financial news for today",
        "Look up the P/E ratio for tech stocks",
        "Get real-time quotes for my portfolio",
        "What are the top gaining stocks today?",
        "Check the forex exchange rates",
        "Show me the candlestick chart for TSLA",
        "Find dividend yield information",
        "What is the current gold price?",
        "Analyze the lithium sector trends",
        "How is my investment portfolio doing?",
        "Compare ETF performance this week",
        "Show me the sector rotation chart",
        "Alert me when BTC falls below 60k",
    ],
    "food_delivery": [
        "Order some milk tea",
        "I want to order lunch",
        "Get me some coffee delivered",
        "Order pizza for dinner",
        "Find a good noodle place nearby",
        "I'm hungry, order something fast",
        "Order bubble tea",
        "Get me some dim sum",
        "Order a sandwich for lunch",
        "I want fried chicken delivered",
        "Find a restaurant that delivers hot pot",
        "Order breakfast for tomorrow morning",
        "Get me some sushi delivered",
        "Order vegetarian food nearby",
        "Find late-night food delivery options",
    ],
    "navigation": [
        "Navigate to the nearest mall",
        "How do I get to the airport?",
        "Find the fastest route to downtown",
        "Take me home",
        "Navigate to the nearest hospital",
        "What's the traffic like on the highway?",
        "Find a route avoiding toll roads",
        "How long to reach the train station?",
        "Get directions to the office",
        "Navigate to the nearest parking lot",
        "Find a gas station on my route",
        "Plan a trip across the city",
        "Check if there are accidents on my commute",
        "Find an alternative route to work",
        "Navigate to the nearest pharmacy",
    ],
    "shopping": [
        "Buy a new laptop for work",
        "Find the cheapest iPhone case",
        "Order running shoes size 42",
        "Search for a birthday gift under 200 yuan",
        "Buy some kitchen supplies",
        "Find a good deal on headphones",
        "Order office stationery",
        "Buy a bookshelf for my room",
        "Find skincare products on sale",
        "Order pet food",
        "Buy a winter jacket",
        "Find camera accessories",
        "Order a gaming controller",
        "Buy baby formula",
        "Search for home furniture deals",
    ],
    "entertainment": [
        "Find a new drama to watch tonight",
        "Recommend a good action movie",
        "Play some background music for studying",
        "Find the latest anime episodes",
        "Show me popular variety shows",
        "Watch the news broadcast",
        "Find a documentary about nature",
        "Show me trending content online",
        "Find a comedy for the weekend",
        "Play the latest episode of my favorite series",
        "Find Korean drama recommendations",
        "Watch a live stream",
        "Find classic films from the 1990s",
        "Show me sports highlights from today",
        "Find educational content for kids",
    ],
}

# ---------------------------------------------------------------------------
# OOD query templates
# OOD_QUERIES[domain][ood_target_tool] = list of queries that EXPLICITLY request
# that specific tool — this ensures the LLM can detect the override signal.
# ---------------------------------------------------------------------------

OOD_QUERIES: Dict[str, Dict[str, List[str]]] = {
    "financial": {
        "Tonghua Shun": [
            "Use Tonghua Shun to check the A-share market data",
            "Open Tonghua Shun and look at the lithium sector news",
            "Check the latest A-share prices on Tonghua Shun today",
            "Get me Tonghua Shun's real-time market analysis",
            "Use Tonghua Shun for the CSI 300 component stocks",
        ],
        "East Money": [
            "Use East Money to look up mutual fund rankings",
            "Check East Money for the latest fund performance data",
            "Open East Money and show me the hot stock discussion board",
            "I need East Money's analysis on the semiconductor sector",
            "Use East Money to find top-performing funds this month",
        ],
        "Bloomberg": [
            "Check Bloomberg for US market macro analysis",
            "I need Bloomberg's data on global bond yields",
            "Use Bloomberg terminal to pull the latest CPI report",
            "Get me Bloomberg's coverage of the Fed rate decision",
            "Use Bloomberg for international market intelligence today",
        ],
        "Yahoo Finance": [
            "Use Yahoo Finance to check international stock prices",
            "Look up this company's fundamentals on Yahoo Finance",
            "Check Yahoo Finance for pre-market trading data",
            "Get me Yahoo Finance's earnings calendar this quarter",
            "Use Yahoo Finance to compare S&P 500 ETF performance",
        ],
    },
    "food_delivery": {
        "Meituan": [
            "Order using Meituan, they have a flash sale right now",
            "Use Meituan for delivery, I have Meituan points to spend",
            "Check Meituan for nearby restaurant promotions",
            "Order on Meituan, I just got a new user coupon",
            "Use Meituan, they are running a free delivery promotion",
        ],
        "Ele.me": [
            "Order on Ele.me, my Alipay has a food discount coupon",
            "Use Ele.me for delivery, they have a coffee promotion",
            "Check Ele.me for the subsidized meal deal today",
            "Order from Ele.me, I have a 20-yuan voucher for them",
            "Use Ele.me, they offer free delivery with Alipay today",
        ],
        "Taobao Delivery": [
            "Order using Taobao delivery, I have an 88VIP coupon",
            "Use Taobao's food delivery service this time",
            "Order through Taobao delivery for the bundled discount",
            "Check Taobao Delivery for this restaurant's listing",
            "Use Taobao delivery, it qualifies for my membership reward",
        ],
        "WeChat MiniProgram": [
            "Order using the WeChat mini program discount",
            "Use WeChat to order food, I have a red packet to use",
            "Order through WeChat MiniProgram for the group-buy deal",
            "Pay with WeChat Pay and get the extra cashback discount",
            "Use WeChat MiniProgram, my company canteen is on there",
        ],
    },
    "navigation": {
        "Amap": [
            "Use Amap to navigate, it has better live traffic data",
            "Navigate with Amap, their construction detour is updated",
            "Open Amap and find the fastest route with real-time traffic",
            "Check Amap for the road closure near my destination",
            "Use Amap today, their map coverage is better here",
        ],
        "Baidu Maps": [
            "Use Baidu Maps for directions, it has transit schedules",
            "Navigate with Baidu Maps today",
            "Open Baidu Maps, it has better indoor mall navigation",
            "Check Baidu Maps for bus route options",
            "Use Baidu Maps, their street-view helped me last time",
        ],
        "DiDi": [
            "Use DiDi to call a cab to the airport",
            "Book a DiDi ride, I don't want to drive in this rain",
            "Call a DiDi Express, I need to get downtown quickly",
            "Order a DiDi, my car is in the shop today",
            "Use DiDi carpooling to save money on this trip",
        ],
        "CaoCao": [
            "Book a CaoCao ride, they have a discount promotion now",
            "Use CaoCao, it's cheaper than DiDi during off-peak hours",
            "Call a CaoCao car for this long-distance trip",
            "Order CaoCao, my company has a corporate account there",
            "Use CaoCao, they have new energy vehicles available",
        ],
    },
    "shopping": {
        "Taobao": [
            "Search on Taobao for the cheapest option available",
            "Buy this on Taobao, I trust this particular seller's store",
            "Check Taobao for this niche item, it's hard to find elsewhere",
            "Order from Taobao, the manufacturer has a flagship store",
            "Use Taobao for the customized version of this product",
        ],
        "JD.com": [
            "Buy on JD.com for same-day delivery",
            "Check JD.com, I need guaranteed authentic electronics",
            "Order from JD.com with my Plus membership discount",
            "Use JD.com for this purchase, they have official warranty",
            "Buy on JD.com, their return policy is better for electronics",
        ],
        "Pinduoduo": [
            "Check Pinduoduo for the absolute lowest price on this",
            "Order from Pinduoduo, it's much cheaper there",
            "Buy on Pinduoduo and use the team purchase deal",
            "Search Pinduoduo for this everyday item",
            "Use Pinduoduo, the daily subsidy price is great today",
        ],
        "Amazon": [
            "Buy this on Amazon, I need international shipping",
            "Order from Amazon for Prime two-day delivery",
            "Check Amazon for this imported product",
            "Use Amazon for this purchase, I have Prime membership",
            "Buy on Amazon, this brand only sells internationally",
        ],
    },
    "entertainment": {
        "Netflix": [
            "Open Netflix, I want to watch the new original series",
            "Find this show on Netflix, it's a Netflix exclusive",
            "Use Netflix for movie night, they just added new titles",
            "Check what's new on Netflix this week",
            "Open Netflix and continue watching my series",
        ],
        "Bilibili": [
            "Watch this on Bilibili, the fan-sub group uploaded it",
            "Open Bilibili for anime, their subtitle quality is best",
            "Check Bilibili for the creator's latest upload",
            "Use Bilibili to watch this live stream event",
            "Go to Bilibili, this documentary is only available there",
        ],
        "Youku": [
            "Watch this drama on Youku, they have exclusive broadcast rights",
            "Check Youku for the latest episodes of this series",
            "Use Youku, they have this show with VIP early access",
            "Open Youku to watch this variety show tonight",
            "Use Youku, this Alibaba production is exclusive to them",
        ],
        "iQiyi": [
            "Use iQiyi, they have exclusive streaming rights for this show",
            "Watch on iQiyi, I have a VIP membership there",
            "Find this series on iQiyi, it premiered there first",
            "Open iQiyi for tonight's live premiere episode",
            "Use iQiyi, their VIP content is what I want to watch",
        ],
    },
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class UserPersona:
    user_id: int
    # Ground-truth preferred tool per domain (hard label for reward computation)
    preferences: Dict[str, str]
    # Optional soft preferences (Dirichlet-sampled probability distribution)
    # None for standard one-hot users
    soft_preferences: Optional[Dict[str, Dict[str, float]]]
    # Pre-generated query pool: (query_text, domain, is_ood, true_tool)
    query_pool: List[Tuple[str, str, bool, str]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Preference assignment
# ---------------------------------------------------------------------------


def _assign_preferences(domains: Dict[str, List[str]]) -> Dict[str, str]:
    """Randomly assign one preferred tool per domain (one-hot)."""
    return {domain: random.choice(tools) for domain, tools in domains.items()}


def _assign_soft_preferences(
    domains: Dict[str, List[str]], concentration: float = 2.0, rng: Optional[np.random.Generator] = None
) -> Dict[str, Dict[str, float]]:
    """
    Assign soft preference distributions via Dirichlet sampling.
    Higher concentration → more peaked (closer to one-hot).
    Returns {domain: {tool: prob}}.
    """
    if rng is None:
        rng = np.random.default_rng()
    soft: Dict[str, Dict[str, float]] = {}
    for domain, tools in domains.items():
        alpha = np.full(len(tools), concentration)
        probs = rng.dirichlet(alpha)
        soft[domain] = {tool: float(p) for tool, p in zip(tools, probs)}
    return soft


# ---------------------------------------------------------------------------
# Query pool generation
# ---------------------------------------------------------------------------


def _generate_query_pool(
    preferences: Dict[str, str],
    domains: Dict[str, List[str]],
    pool_size: int = 200,
    ood_ratio: float = 0.10,
) -> List[Tuple[str, str, bool, str]]:
    """
    Build a pool of (query, domain, is_ood, target_tool) tuples.

    Standard queries (1 - ood_ratio fraction): reward the user's preferred tool.
    OOD queries (ood_ratio fraction): explicitly request a *different* tool via
    OOD_QUERIES[domain][ood_target], guaranteeing query-target consistency.
    """
    pool: List[Tuple[str, str, bool, str]] = []
    domain_list = list(domains.keys())

    n_ood = max(1, int(pool_size * ood_ratio))
    n_standard = pool_size - n_ood

    # Standard queries — true_tool is user's preferred tool
    for _ in range(n_standard):
        domain = random.choice(domain_list)
        query = random.choice(STANDARD_QUERIES[domain])
        preferred = preferences[domain]
        pool.append((query, domain, False, preferred))

    # OOD queries — pick a non-preferred tool and sample a query that *names* it
    for _ in range(n_ood):
        domain = random.choice(domain_list)
        preferred = preferences[domain]
        other_tools = [t for t in domains[domain] if t != preferred]
        ood_target = random.choice(other_tools)
        # Query explicitly mentions ood_target → LLM can detect the override
        query = random.choice(OOD_QUERIES[domain][ood_target])
        pool.append((query, domain, True, ood_target))

    random.shuffle(pool)
    return pool


def _generate_query_pool_soft(
    soft_preferences: Dict[str, Dict[str, float]],
    domains: Dict[str, List[str]],
    pool_size: int = 200,
    ood_ratio: float = 0.10,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[str, str, bool, str]]:
    """
    Build a query pool where true_tool is sampled from soft preference distribution.

    Standard queries: true_tool sampled from soft_preferences[domain] probabilities.
    OOD queries: explicitly request a non-argmax tool (same as _generate_query_pool).
    """
    if rng is None:
        rng = np.random.default_rng()

    pool: List[Tuple[str, str, bool, str]] = []
    domain_list = list(domains.keys())

    n_ood = max(1, int(pool_size * ood_ratio))
    n_standard = pool_size - n_ood

    # Standard queries — true_tool sampled from soft distribution
    for _ in range(n_standard):
        domain = random.choice(domain_list)
        query = random.choice(STANDARD_QUERIES[domain])
        tools = domains[domain]
        probs = np.array([soft_preferences[domain][t] for t in tools])
        probs = probs / probs.sum()  # re-normalise for safety
        true_tool = rng.choice(tools, p=probs)
        pool.append((query, domain, False, true_tool))

    # OOD queries — pick a non-argmax tool and sample a query that *names* it
    for _ in range(n_ood):
        domain = random.choice(domain_list)
        tools = domains[domain]
        probs_dict = soft_preferences[domain]
        argmax_tool = max(probs_dict, key=probs_dict.get)
        other_tools = [t for t in tools if t != argmax_tool]
        ood_target = random.choice(other_tools)
        query = random.choice(OOD_QUERIES[domain][ood_target])
        pool.append((query, domain, True, ood_target))

    random.shuffle(pool)
    return pool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_users(
    n_users: int = 20,
    seed: int = 42,
    pool_size: int = 200,
    ood_ratio: float = 0.10,
    domains: Optional[Dict[str, List[str]]] = None,
) -> List[UserPersona]:
    """
    Generate n_users synthetic user personas with one-hot preferences.
    Fully reproducible given the same seed.
    """
    if domains is None:
        domains = DOMAINS
    random.seed(seed)
    np.random.seed(seed)
    users = []
    for uid in range(n_users):
        prefs = _assign_preferences(domains)
        pool = _generate_query_pool(prefs, domains, pool_size=pool_size, ood_ratio=ood_ratio)
        users.append(
            UserPersona(user_id=uid, preferences=prefs, soft_preferences=None, query_pool=pool)
        )
    return users


def generate_users_soft(
    n_users: int = 20,
    seed: int = 42,
    pool_size: int = 200,
    ood_ratio: float = 0.10,
    concentration: float = 2.0,
    domains: Optional[Dict[str, List[str]]] = None,
) -> List[UserPersona]:
    """
    Generate users with Dirichlet-sampled soft preferences.
    The most-probable tool is used as the hard label for reward computation.
    `concentration` controls peakedness: higher → closer to one-hot.
    """
    if domains is None:
        domains = DOMAINS
    random.seed(seed)
    np_rng = np.random.default_rng(seed)
    users = []
    for uid in range(n_users):
        soft = _assign_soft_preferences(domains, concentration=concentration, rng=np_rng)
        # Hard label = argmax of soft distribution
        prefs = {domain: max(tool_probs, key=tool_probs.get) for domain, tool_probs in soft.items()}
        pool = _generate_query_pool_soft(soft, domains, pool_size=pool_size, ood_ratio=ood_ratio, rng=np_rng)
        users.append(
            UserPersona(user_id=uid, preferences=prefs, soft_preferences=soft, query_pool=pool)
        )
    return users


def sample_query(user: UserPersona, rng: random.Random) -> Tuple[str, str, bool, str]:
    """Sample a random (query, domain, is_ood, true_tool) from the user's pool."""
    return rng.choice(user.query_pool)


if __name__ == "__main__":
    users = generate_users()
    for u in users[:3]:
        print(f"User {u.user_id}: preferences={u.preferences}")
        standard = [(q, d, t) for q, d, ood, t in u.query_pool if not ood]
        ood = [(q, d, t) for q, d, ood, t in u.query_pool if ood]
        print(f"  Pool size: {len(u.query_pool)} ({len(standard)} standard, {len(ood)} OOD)")
        print(f"  Sample standard: {standard[0]}")
        print(f"  Sample OOD:      {ood[0]}")
        print()

    print("--- Soft preferences ---")
    users_soft = generate_users_soft(n_users=2, concentration=3.0)
    for u in users_soft:
        print(f"User {u.user_id}: hard_prefs={u.preferences}")
        for domain, dist in u.soft_preferences.items():
            dist_str = ", ".join(f"{t}: {p:.2f}" for t, p in dist.items())
            print(f"  {domain}: {dist_str}")
        # Show true_tool distribution in pool per domain
        from collections import Counter
        for domain in DOMAINS:
            tool_counts = Counter(t for _, d, ood, t in u.query_pool if d == domain and not ood)
            total = sum(tool_counts.values())
            if total > 0:
                freq_str = ", ".join(f"{t}: {c}/{total}" for t, c in tool_counts.most_common())
                print(f"  Pool [{domain}]: {freq_str}")
        print()
