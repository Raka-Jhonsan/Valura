"""
Safety Guard — runs BEFORE the LLM is called.
Pure local computation. No network calls. No LLM. Must complete in < 10 ms.

Design tradeoff (documented per assignment):
  Educational framing is detected first. If a query contains clear educational
  markers ("what is", "explain", "how does X work", "define", "example of") it
  passes through even if harmful keywords are present. This means a narrow band
  of queries like "explain how to do a pump-and-dump step by step" will pass the
  guard and reach the LLM classifier, which applies a second safety_verdict
  (informational only — does not re-block). That is the accepted cost of keeping
  this guard under 10 ms with no network call.
"""
import re
import logging
from typing import Optional, Tuple

# ─── Colors (same pattern as deal_agent_framework.py) ────────────────────────
BG_RED = "\033[41m"
WHITE = "\033[37m"
RESET = "\033[0m"


# ─── Educational framing markers — checked FIRST ─────────────────────────────
_EDUCATIONAL_PATTERNS = [
    r"\bwhat\s+is\b",
    r"\bwhat\s+are\b",
    r"\bwhat'?s\s+(the\s+)?difference\b",
    r"\bdescribe\b",
    r"\bexplain\b",
    r"\bdefine\b",
    r"\bhow\s+does\b",
    r"\btell\s+me\s+about\b",
    r"\blearn\s+about\b",
    r"\beducate\s+me\b",
    r"\bexample\s+of\b",
    r"\bhistory\s+of\b",
    r"\bmeaning\s+of\b",
    r"\bconcept\s+of\b",
    r"\bwhy\s+is\b",
    r"\bwhy\s+do\b",
]

# ─── Block categories: (patterns, response_message) ──────────────────────────
_BLOCK_CATEGORIES: dict[str, Tuple[list[str], str]] = {
    "insider_trading": (
        [
            r"\binsider\s+trad(e|ing|ed)\b",
            r"\bmaterial\s+non.?public\b",
            r"\bnon.?public\s+information\b",
            r"\btrade\s+on\s+(a\s+)?tip\b",
            r"\bbuy\b.{0,40}\bbefore\s+(the\s+)?(announcement|earnings\s+leak|merger)\b",
            r"\bearnings\b.{0,55}\bbefore\b.{0,35}\b(announcement|call)\b",
            r"\bneed\s+to\s+know\s+(the\s+)?earnings\b",
            r"\buse\s+(my\s+)?inside\s+(knowledge|info|information)\b",
            r"\btip\s+about\s+earnings\b",
            r"\bconfidential\s+(merger|deal|news)\b",
            r"\bunannounced\b.{0,50}\bacquisition\b",
            r"\bknow\s+about\b.{0,50}\b(acquisition|merger|buyout)\b",
        ],
        (
            "I'm not able to assist with trading on material non-public information. "
            "Insider trading is a serious criminal offence that carries significant fines "
            "and imprisonment. If you have a legitimate investment question based on "
            "public information, I'm happy to help."
        ),
    ),
    "market_manipulation": (
        [
            r"\bpump\s+(and\s+)?dump\b",
            r"\bwash\s+trad(e|ing)\b",
            r"\bspoofing\b.{0,30}\bmarket\b",
            r"\bmanipulate\s+(the\s+)?market\b",
            r"\bartificially\s+(inflate|deflate)\s+(the\s+)?(price|stock|market)\b",
            r"\bshort\s+(and\s+)?distort\b",
            r"\bcoordinate\s+(a\s+)?(buy|sell|squeeze)\b",
            r"\bpump\b.{0,40}\b(price|up)\b",
            r"\bpump\s+up\b.{0,30}\bprice\b",
            r"\bcoordinated\b.{0,40}\bbuying\b",
            r"\bcoordinated\b.{0,30}\b(scheme|buy)\b",
        ],
        (
            "I'm not able to assist with market manipulation strategies. These activities "
            "are illegal under securities law in most jurisdictions and cause direct harm "
            "to other investors. I can help you with legitimate investment analysis."
        ),
    ),
    "money_laundering": (
        [
            r"\blaunder(ing|ed)?\s+(money|funds|cash|proceeds)\b",
            r"\bclean\s+(dirty|illegal|illicit)\s+money\b",
            r"\bhide\s+(the\s+)?(source|origin|trail)\s+of\s+(funds|money|cash)\b",
            r"\bhide\b.{0,55}\btax\b.{0,30}\bauthorit\b",
            r"\bhid(e|ing)\b.{0,45}\b(trading\s+)?profits\b.{0,35}\btax\b",
            r"\bstructur(e|ing)\b.{0,50}\b(10k|reporting\s+threshold|threshold\s+avoid)\b",
            r"\bdeposits?\b.{0,45}\bavoid\b.{0,35}\b(10\s*k|reporting\s+threshold)\b",
            r"\blayer\b.{0,40}\bobscur(e|ing)\b.{0,35}\bfunds\b",
            r"\bstructur(e|ing)\s+(cash|deposits|transactions)\b.{0,30}\bavoid\b",
            r"\bsmurfi?ng\b",
            r"\bconceal\s+(the\s+)?(origin|source)\s+of\b",
            r"\b500\s*k\b.{0,55}\b(report|reporting)\b",
            r"\b500k\b.{0,55}\bwithout\b.{0,35}\breport(ing)?\b",
            r"\bcash\b.{0,50}\bwithout\b.{0,35}\breport(ing)?\b",
        ],
        (
            "I'm not able to provide guidance on concealing the origin of funds or "
            "money laundering. These are serious financial crimes. If you have questions "
            "about legitimate tax-efficient investment structures, I'm glad to help."
        ),
    ),
    "guaranteed_returns": (
        [
            r"\bguarantee\s+me\b",
            r"\bpromise\s+me\b",
            r"\bguarantee[sd]?\s+(return|profit|gain|yield|income)\b",
            r"\b100\s*%\s*(safe|guaranteed|risk.?free|certain)\b",
            r"\bno.?risk\s+(investment|opportunity|trade)\b",
            r"\bcannot\s+(lose|fail)\b",
            r"\bwill\s+definitely\s+(go\s+up|rise|increase|double|triple)\b",
            r"\bfoolproof\b.{0,40}\b(make|earn|gain|profit|returns?)\b",
            r"\bsure.?fire\s+(profit|return|gain|winner)\b",
            r"\bzero\s+risk\s+(investment|trade|opportunity)\b",
        ],
        (
            "I'm not able to make or endorse claims of guaranteed investment returns. "
            "All investments carry risk, and any promise of guaranteed profits is a "
            "hallmark of financial fraud. I can help you understand realistic expected "
            "returns and evidence-based risk management."
        ),
    ),
    "reckless_advice": (
        [
            r"\bbet\s+(everything|all\s+(my\s+)?(savings|money|capital))\b",
            r"\bgo\s+all.?in\s+on\b",
            r"\ball\s+(my\s+)?retirement\b.{0,40}\bcrypto\b",
            r"\bput\s+(all|everything)\s+(my\s+)?(retirement\s+)?savings\b.{0,40}\bcrypto\b",
            r"\bmortgage\s+(my\s+)?(house|home).{0,40}\b(invest|buy|trade|stock|nvidia)\b",
            r"\bstock\s+to\s+mortgage\s+(my\s+)?(house|home)\s+for\b",
            r"\bborrow\s+(everything|as\s+much\s+as\s+possible).{0,30}\b(invest|stock|crypto)\b",
            r"\bmax\s+out\s+(my\s+)?(credit\s+card|loan).{0,30}\b(invest|buy|trade)\b",
            r"\bmargin\s+loan\b.{0,35}\bbuy\b",
            r"\bleveraged\b.{0,30}\bmargin\b",
            r"\bemergency\s+fund\b.{0,35}\boptions\b",
            r"\btell\s+me\s+to\b.{0,50}\bmargin\b",
        ],
        (
            "I'm not able to recommend putting all assets or borrowed money into a single "
            "investment. Concentrating your entire net worth in one position or investing "
            "borrowed funds you cannot afford to lose carries extreme risk of financial ruin. "
            "I can help you think through a balanced, risk-appropriate strategy instead."
        ),
    ),
    "sanctions_evasion": (
        [
            r"\bbypass\b.{0,55}\bofac\b",
            r"\bofac\b.{0,55}\bbypass\b",
            r"\bshell\s+company\b.{0,45}\bsanction\b",
            r"\bsanctioned\b.{0,55}\b(traced|trace|detect|avoid|tracing)\b",
            r"\binvest\s+in\b.{0,55}\bsanctioned\b",
            r"\bwithout\b.{0,40}\b(being\s+)?(traced|trace|tracing)\b",
        ],
        (
            "I'm not able to help with circumventing sanctions or concealment schemes. "
            "Sanctions compliance applies to brokers and investors; breaching them can "
            "carry severe penalties. For legitimate questions about how sanctions affect "
            "investing, I would be glad to outline high-level obligations."
        ),
    ),
    "fraud": (
        [
            r"\bdraft\b.{0,45}\bfake\b.{0,35}\bcontract\b",
            r"\bfake\s+contract\s+note\b",
            r"\bponzi\s+scheme\b",
            r"\bpyramid\s+scheme\b",
            r"\bfraud(ulent)?\s+(investment|scheme|fund)\b",
            r"\bscam\s+(investors|victims|people)\b",
            r"\bfake\s+(prospectus|filing|audit|returns)\b",
            r"\bmisrepresent\s+(returns|performance|results)\b",
        ],
        (
            "I'm not able to assist with fraudulent investment schemes or fabricated documents. "
            "Creating or promoting fraud causes serious financial harm and carries severe "
            "criminal penalties. If you have a legitimate question about investment "
            "structures or regulations, I'm here to help."
        ),
    ),
}


# Pre-compile all patterns once at import time for speed
_COMPILED_EDUCATIONAL = [re.compile(p, re.IGNORECASE) for p in _EDUCATIONAL_PATTERNS]
_COMPILED_BLOCKS = {
    category: (
        [re.compile(p, re.IGNORECASE) for p in patterns],
        message,
    )
    for category, (patterns, message) in _BLOCK_CATEGORIES.items()
}


class SafetyGuard:
    """
    Synchronous safety filter. Call check() before any LLM call.
    Returns (is_safe, blocked_category, response_message).
    Pure local computation — no network, no LLM, < 10 ms guaranteed.
    """

    def __init__(self):
        self.log("Safety Guard initialised")

    def log(self, message: str):
        text = BG_RED + WHITE + "[Safety Guard] " + message + RESET
        logging.info(text)

    def _is_educational(self, query: str) -> bool:
        """Return True if the query has clear educational framing."""
        return any(p.search(query) for p in _COMPILED_EDUCATIONAL)

    def check(self, query: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Returns:
            (True, None, None)                         — safe, proceed
            (False, category_name, response_message)   — blocked
        """
        if self._is_educational(query):
            return True, None, None

        for category, (compiled_patterns, message) in _COMPILED_BLOCKS.items():
            for pattern in compiled_patterns:
                if pattern.search(query):
                    self.log(f"Blocked query — category: {category}")
                    return False, category, message

        return True, None, None


if __name__ == "__main__":
    guard = SafetyGuard()
    tests = [
        ("How do I do insider trading?", False),
        ("What is insider trading?", True),
        ("Help me pump and dump a stock", False),
        ("How does a pump and dump scheme work?", True),
        ("I want guaranteed 50% returns", False),
        ("What returns can I realistically expect?", True),
    ]
    for query, expected_safe in tests:
        is_safe, category, _ = guard.check(query)
        status = "PASS" if is_safe == expected_safe else "FAIL"
        print(f"[{status}] safe={is_safe} | {query}")