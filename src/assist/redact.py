"""Mask sensitive numbers before they are stored, displayed, or summarized.

Callers read out account numbers and PINs during verification (KB-117).
Those digits should never land in a stored summary or be sent to an LLM.
This is deliberately simple: any run of 4+ digits, and anything spoken
right after "PIN", is masked. Real deployments would add card-number
detection (Luhn check), names, and addresses.
"""

from __future__ import annotations

import re

PIN_RE = re.compile(r"(\bpin(?:\s+(?:is|number))?\s*[:#]?\s*)(\d+)", re.IGNORECASE)
LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")
MASK = "[REDACTED]"


def redact(text: str) -> str:
    text = PIN_RE.sub(lambda m: m.group(1) + MASK, text)
    return LONG_NUMBER_RE.sub(MASK, text)
