"""
Spotting items that are already in the catalog.

Two items are treated as the same when they are the same media type and:

  * they have the same ISBN (books) or barcode/UPC (music, movies, games). An
    ISBN-10 and its ISBN-13 count as the same, as do barcodes with or without
    leading zeros; or
  * there's no code to compare, and the title and creator match, ignoring case,
    accents, punctuation and a leading "The". A blank creator matches any creator.

Different codes, or a different format, edition or (for games) platform, mean a
separate copy you meant to have (Abbey Road on vinyl and on CD), not a duplicate.
"""
from __future__ import annotations

import re
import unicodedata


def text_key(text) -> str:
    """'"The Wall"' and 'the wall' -> 'wall'."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", str(text or "")) if not unicodedata.combining(c))
    words = re.sub(r"[\W_]+", " ", plain.casefold()).split()
    if len(words) > 1 and words[0] == "the":
        words = words[1:]
    return " ".join(words)


def _isbn13(digits: str) -> str:
    """ISBN-10 -> ISBN-13 (same book, same code); anything else unchanged."""
    if not re.fullmatch(r"\d{9}[\dX]", digits):
        return digits
    core = "978" + digits[:9]
    check = (10 - sum(int(c) * (1 if i % 2 == 0 else 3) for i, c in enumerate(core)) % 10) % 10
    return core + str(check)


def item_code(media_type, extra) -> str | None:
    """The item's ISBN (books) or barcode/UPC (others), normalized for comparing."""
    extra = extra or {}
    if media_type == "book":
        digits = re.sub(r"[^0-9Xx]", "", str(extra.get("isbn") or "")).upper()
        return _isbn13(digits) or None
    for field in ("barcode", "upc"):
        digits = re.sub(r"\D", "", str(extra.get(field) or "")).lstrip("0")
        if digits:
            return digits
    return None


def _differs(a, b) -> bool:
    """Both filled in, and not the same: e.g. vinyl vs CD."""
    a, b = text_key(a), text_key(b)
    return bool(a and b and a != b)


class DuplicateIndex:
    """Fast lookups over many items, so a big CSV import isn't slow. Items are plain
    dicts as returned by Catalog.get()/list()."""

    def __init__(self, items=()):
        self._by_code: dict[tuple, list] = {}
        self._by_title: dict[tuple, list] = {}
        for item in items:
            self.add(item)

    def add(self, item: dict):
        code = item_code(item["media_type"], item.get("extra"))
        if code:
            self._by_code.setdefault((item["media_type"], code), []).append(item)
        self._by_title.setdefault((item["media_type"], text_key(item.get("title"))), []).append(item)

    def match(self, media_type, title, creator=None, extra=None, format=None, exclude_id=None) -> list[dict]:
        """Items already present that this new item would duplicate."""
        extra = extra or {}
        code = item_code(media_type, extra)
        seen, found = set(), []

        def keep(item):
            if item.get("id") != exclude_id and id(item) not in seen:
                seen.add(id(item))
                found.append(item)

        if code:
            for item in self._by_code.get((media_type, code), []):
                keep(item)
        for item in self._by_title.get((media_type, text_key(title)), []):
            other_code = item_code(media_type, item.get("extra"))
            if code and other_code:
                continue  # both have codes: only an exact code match (above) counts
            if _differs(creator, item.get("creator")):
                continue
            if (_differs(format, item.get("format"))
                    or _differs(extra.get("edition"), item["extra"].get("edition"))
                    or (media_type == "game" and _differs(extra.get("platform"), item["extra"].get("platform")))):
                continue  # a different copy: other format, edition or platform
            keep(item)
        return found
