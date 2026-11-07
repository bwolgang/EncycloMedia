"""
Fill in books' details from their ISBNs (Open Library), e.g. right after a CSV import.

Only blank fields are filled: anything already in the catalog (or in the CSV) is
kept. Filled in when available: author, first-published year, format, cover,
edition, edition year, publisher and page count. Genres are left alone, because
Open Library's subject lists are long and mixed ("Accessible book", "Fiction",
"Large type books") and better chosen by you.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import metadata

# Fields copied from a lookup when the item's own value is blank.
CORE_FILL = ("creator", "year", "format")
EXTRA_FILL = ("cover_url", "edition", "edition_year", "publisher", "pages")


@dataclass
class AutofillReport:
    filled: list[str] = field(default_factory=list)      # titles that got new details
    complete: list[str] = field(default_factory=list)    # found, but nothing was missing
    not_found: list[str] = field(default_factory=list)   # "Title (ISBN)" Open Library doesn't know
    invalid: list[str] = field(default_factory=list)     # "Title (value)" that isn't a valid ISBN
    error: str | None = None                             # set if Open Library couldn't be reached

    @property
    def looked_up(self) -> int:
        return len(self.filled) + len(self.complete) + len(self.not_found)

    def summary(self, retry_hint="Use “Fill in from ISBNs” on the Books page to try again.") -> str:
        """One or two plain sentences for a message or the command line."""
        if self.error:
            return f"Couldn't fill in book details ({self.error}). {retry_hint}"
        if not (self.looked_up or self.invalid):
            return ""
        parts = []
        if self.filled:
            n = len(self.filled)
            parts.append(f"Filled in details for {n} book{'s' if n != 1 else ''} from their ISBNs.")
        if self.not_found:
            parts.append(f"Not found on Open Library: {_short_list(self.not_found)}.")
        if self.invalid:
            parts.append(f"Not valid ISBNs (check for typos, or a spreadsheet turning them into "
                         f"numbers like 9.78E+12): {_short_list(self.invalid)}.")
        if not parts and self.complete:
            parts.append("Those books' details were already complete.")
        return " ".join(parts)


def _short_list(names, limit=4) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def _missing(item: dict) -> bool:
    return any(not item.get(f) for f in CORE_FILL) or any(not item["extra"].get(f) for f in EXTRA_FILL)


def books_needing_details(catalog) -> list[int]:
    """Ids of books that have an ISBN but are missing details a lookup could fill."""
    return [it["id"] for it in catalog.list("book") if it["extra"].get("isbn") and _missing(it)]


def autofill_books(catalog, item_ids=None, lookup=None) -> AutofillReport:
    """Look up the given items (default: every book needing details) by ISBN and fill
    in their blank fields. Items that aren't books, or have no ISBN, are skipped."""
    lookup = lookup or metadata.books_by_isbns
    report = AutofillReport()
    if item_ids is None:
        item_ids = books_needing_details(catalog)

    books = []
    for item_id in item_ids:
        item = catalog.get(item_id)
        if not item or item["media_type"] != "book" or not item["extra"].get("isbn"):
            continue
        isbn = metadata.normalize_isbn(item["extra"]["isbn"])
        if isbn is None:
            report.invalid.append(f"{item['title']} ({item['extra']['isbn']})")
        else:
            books.append((item, isbn))
    if not books:
        return report

    try:
        found = lookup([isbn for _, isbn in books])
    except metadata.MetadataError as e:
        report.error = str(e)
        return report

    for item, isbn in books:
        cand = found.get(isbn)
        if cand is None:
            report.not_found.append(f"{item['title']} ({isbn})")
            continue
        core = {f: cand[f] for f in CORE_FILL if not item.get(f) and cand.get(f)}
        extra_new = {f: cand["extra"][f] for f in EXTRA_FILL
                     if not item["extra"].get(f) and cand["extra"].get(f)}
        if core or extra_new:
            catalog.update(item["id"], extra={**item["extra"], **extra_new}, **core)
            report.filled.append(item["title"])
        else:
            report.complete.append(item["title"])
    return report
