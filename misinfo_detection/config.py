from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    tavily_api_key: str
    reliable_domains: List[str]
    max_rounds: int = 2
    tavily_max_results: int = 5
    tavily_topic: str = "general"


def load_config() -> AppConfig:
    load_dotenv()
    tavily_api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not tavily_api_key:
        raise ValueError("Missing TAVILY_API_KEY in environment/.env")

    reliable_domains = [
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "politifact.com",
        "snopes.com",
        "cdc.gov",
        "who.int",
    ]

    max_rounds = int(os.getenv("MAX_ROUNDS", "2"))
    tavily_max_results = int(os.getenv("TAVILY_MAX_RESULTS", "5"))
    tavily_topic = os.getenv("TAVILY_TOPIC", "general").strip() or "general"

    return AppConfig(
        tavily_api_key=tavily_api_key,
        reliable_domains=reliable_domains,
        max_rounds=max_rounds,
        tavily_max_results=tavily_max_results,
        tavily_topic=tavily_topic,
    )

