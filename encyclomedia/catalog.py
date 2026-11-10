"""
Core catalog: storage, type-specific fields, keyword (BM25) search and hybrid
keyword + semantic search.

Storage is a single SQLite file:
  items        one row per item; shared fields as columns, type-specific fields as JSON
  genres       every genre in use, with a normalized key so "Hip-Hop" == "hip hop" == "hiphop"
  item_genres  which genres each item has (many-to-many, ordered)
  items_fts    FTS5 full-text index over the searchable text, kept in sync by triggers
  embeddings   optional sentence-embedding vectors for semantic search

items.genre keeps a ", "-joined copy of an item's genres so keyword search,
CSV export and embeddings can use it as plain text; the genre tables are the
source of truth and the catalog keeps the two in sync.
"""
from __future__ import annotations

import csv
import difflib
import hashlib
import io
import json
import re
import sqlite3
import unicodedata
import uuid
from pathlib import Path

from . import covers

DB_DEFAULT = Path.home() / ".encyclomedia.db"
SCHEMA_VERSION = 3

MEDIA_TYPES = ("book", "music", "movie", "game")

# Fields every item has.
CORE_FIELDS = ("title", "creator", "year", "genre", "format", "tags", "notes")

# What "creator" (the main credit) means for each type. For games it is the studio
# that made the game, which is what earlier versions stored there as "Developer".
CREATOR_LABEL = {"book": "Author", "music": "Artist", "movie": "Director", "game": "Game studio"}

# Further credits for each type, shown next to the main credit. Stored in the JSON
# `extra` column. Names can be comma separated ("Bilal, Anna Wise").
CREDIT_FIELDS = {
    "book": ("illustrator",),
    "music": ("performers", "label"),
    "movie": ("composer",),
    "game": ("designer",),
}
CREDIT_LABELS = {"illustrator": "Illustrator", "performers": "Performers", "label": "Record label",
                 "composer": "Composer", "designer": "Game designer"}
# Credits that name people (as opposed to a record label), used for "More like this".
PEOPLE_FIELDS = ("illustrator", "performers", "composer", "designer")

# What "year" means for each type. It is always the original release, so a 1937
# novel says 1937 even if your copy is a 2012 printing; that goes in edition_year.
YEAR_LABEL = {"book": "First published", "music": "Original release",
              "movie": "Release year", "game": "Release year"}

# Details about your physical copy, the same for every media type. Stored in the
# JSON `extra` column alongside the type-specific fields, and included in search.
COPY_FIELDS = ("edition", "location", "origin", "condition")
COPY_LABELS = {"edition": "Edition", "location": "Location", "origin": "Origin",
               "condition": "Condition notes"}

# Type-specific fields, stored in the JSON `extra` column.
# Credit fields come first so CSV exports and forms list them together.
EXTRA_FIELDS = {
    "book": (*CREDIT_FIELDS["book"], "isbn", "edition_year", "publisher", "pages", "series", "cover_url"),
    "music": (*CREDIT_FIELDS["music"], "barcode", "edition_year", "tracklist", "discs", "cover_url"),
    "movie": (*CREDIT_FIELDS["movie"], "upc", "runtime", "rating", "cast", "cover_url"),
    "game": (*CREDIT_FIELDS["game"], "platform", "publisher", "players", "barcode", "cover_url"),
}


def split_names(value) -> list[str]:
    """'Bilal, Anna Wise;Thundercat' -> ['Bilal', 'Anna Wise', 'Thundercat']."""
    return [n.strip() for n in re.split(r"[,;\n]", str(value or "")) if n.strip()]

# Ranking weights per FTS column (same order as the items_fts definition).
WEIGHTS = {"title": 10.0, "creator": 6.0, "genre": 3.0, "tags": 3.0, "notes": 1.0, "extra_text": 2.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY,
    media_type TEXT NOT NULL CHECK (media_type IN ('book','music','movie','game')),
    title      TEXT NOT NULL,
    creator    TEXT,
    year       INTEGER,
    genre      TEXT,
    format     TEXT,
    tags       TEXT,
    notes      TEXT,
    extra      TEXT NOT NULL DEFAULT '{}',   -- JSON of type-specific fields
    extra_text TEXT,                        -- searchable flattening of `extra`
    added_at   TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS embeddings (
    item_id   INTEGER PRIMARY KEY REFERENCES items(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    vector    BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS genres (
    id   INTEGER PRIMARY KEY,
    name TEXT NOT NULL,          -- display name, as first entered
    key  TEXT NOT NULL UNIQUE    -- normalized: lowercase letters/digits only
);
CREATE TABLE IF NOT EXISTS item_genres (
    item_id  INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
    genre_id INTEGER NOT NULL REFERENCES genres(id) ON DELETE CASCADE,
    position INTEGER NOT NULL DEFAULT 0,   -- keeps the order genres were listed in
    PRIMARY KEY (item_id, genre_id)
);
CREATE INDEX IF NOT EXISTS item_genres_by_genre ON item_genres(genre_id);
-- "File under" for a creator, e.g. Kate Bush -> "Bush, Kate". One row per name,
-- shared by every item with that creator; names without a row are filed automatically.
CREATE TABLE IF NOT EXISTS sort_names (
    key     TEXT PRIMARY KEY,   -- name_key(creator)
    sort_as TEXT NOT NULL
);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE items_fts USING fts5(
    title, creator, genre, tags, notes, extra_text,
    content='items', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);
CREATE TRIGGER items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, title, creator, genre, tags, notes, extra_text)
    VALUES (new.id, new.title, new.creator, new.genre, new.tags, new.notes, new.extra_text);
END;
CREATE TRIGGER items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, creator, genre, tags, notes, extra_text)
    VALUES ('delete', old.id, old.title, old.creator, old.genre, old.tags, old.notes, old.extra_text);
END;
CREATE TRIGGER items_au AFTER UPDATE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, creator, genre, tags, notes, extra_text)
    VALUES ('delete', old.id, old.title, old.creator, old.genre, old.tags, old.notes, old.extra_text);
    INSERT INTO items_fts(rowid, title, creator, genre, tags, notes, extra_text)
    VALUES (new.id, new.title, new.creator, new.genre, new.tags, new.notes, new.extra_text);
END;
"""


_LEADING_THE = re.compile(r"^the\s+", re.I)
_LEADING_PUNCT = re.compile(r"^[\W_]+")   # quotes, dots, brackets, dashes... before the first letter or digit


def sort_name(name) -> str:
    """How a title or name is alphabetized. Case and accents are ignored (Élan files with
    the E's), and so are leading punctuation and a leading "The":
        '"Experience" Jimi Hendrix'      -> experience" jimi hendrix
        '...Somewhere More Familiar'      -> somewhere more familiar
        '(What's the Story) Morning Glory?' -> what's the story) morning glory?
        'The Beatles' / '"The Wall"'      -> beatles / wall"
    Names that are nothing but punctuation ('!!!') or just "The The" are kept as they are."""
    text = " ".join(str(name or "").split())
    plain = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    trimmed = _LEADING_PUNCT.sub("", plain)
    key = _LEADING_PUNCT.sub("", _LEADING_THE.sub("", trimmed))
    return (key or trimmed or plain).casefold()


def name_key(name) -> str:
    """The same creator however it's spaced or capitalized."""
    return " ".join(str(name or "").split()).casefold()


# Media types whose main credit is (almost always) a person, so it files by surname
# automatically: authors and directors. Artists can be bands, and game studios are
# companies, so those file as written unless given a "File under" name.
PERSON_TYPES = ("book", "movie")

_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "phd", "md"}
# Lowercase particles go after the given name, as libraries do: Beethoven, Ludwig van.
# Capitalized ones belong to the surname: Le Guin, Ursula K.; De Niro, Robert.
_PARTICLES = {"van", "von", "de", "der", "den", "du", "da", "di", "del", "della", "la", "le", "ten", "ter"}


def surname_first(name) -> str:
    """'Kate Bush' -> 'Bush, Kate'; 'Martin Luther King Jr.' -> 'King, Martin Luther, Jr.';
    'Ludwig van Beethoven' -> 'Beethoven, Ludwig van'; 'Ursula K. Le Guin' -> 'Le Guin, Ursula K.'.
    With several people ('Neil Gaiman, Terry Pratchett') the first one decides.
    A one-word name ('Madonna', 'Moebius') is left as it is."""
    text = " ".join(str(name or "").split())
    first = re.split(r"\s*(?:,|;|&|\band\b)\s*", text, maxsplit=1)[0] or text
    words = first.split(" ")
    suffix = []
    while len(words) > 2 and words[-1].casefold().rstrip(",") in _SUFFIXES:
        suffix.insert(0, words.pop())
    if len(words) < 2:
        return text
    surname = [words.pop()]
    while len(words) > 1 and words[-1].casefold() in _PARTICLES:
        surname.insert(0, words.pop())
    lower = [w for w in surname[:-1] if w.islower()]          # van, von... move after given name
    surname = [w for w in surname if w not in lower]
    out = f"{' '.join(surname)}, {' '.join(words + lower)}"
    return out + (f", {' '.join(suffix)}" if suffix else "")


def _clean(value):
    """Blank strings become None; everything else is stripped text or left alone."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


_GENRE_SEPARATORS = re.compile(r"[,;/|]")


def genre_key(name: str) -> str:
    """Normalized form used to treat spelling variants as one genre."""
    return re.sub(r"[\W_]+", "", name.casefold())


def split_genres(value) -> list[str]:
    """'Hip-Hop, jazz rap; Jazz Rap' -> ['Hip-Hop', 'jazz rap'] (deduplicated, order kept)."""
    if value is None:
        return []
    parts = value if isinstance(value, (list, tuple)) else _GENRE_SEPARATORS.split(str(value))
    names, seen = [], set()
    for part in parts:
        name = " ".join(str(part).split())
        k = genre_key(name)
        if k and k not in seen:
            seen.add(k)
            names.append(name)
    return names


def _flatten_extra(extra: dict) -> str:
    """Searchable text from type-specific fields (URLs and IDs are not useful to search)."""
    parts = []
    for key, value in extra.items():
        if value in (None, "") or key.endswith("_url") or key.endswith("_id"):
            continue
        parts.append(", ".join(map(str, value)) if isinstance(value, list) else str(value))
    return " ".join(parts)


class Catalog:
    """The media catalog. Pass an `embedder` (see semantic.py) to enable semantic search."""

    def __init__(self, db_path=DB_DEFAULT, embedder=None, check_same_thread=True):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.create_function("sort_name", 1, sort_name, deterministic=True)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.embedder = embedder
        self._migrate()

    # ------------------------------------------------------------------ schema
    def _migrate(self):
        """Create a fresh database, or upgrade a version-1 database in place."""
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        self.conn.executescript(SCHEMA)
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(items)")}
        for col, ddl in (("extra", "TEXT NOT NULL DEFAULT '{}'"), ("extra_text", "TEXT"), ("updated_at", "TEXT")):
            if col not in cols:  # database created by the original single-file script
                self.conn.execute(f"ALTER TABLE items ADD COLUMN {col} {ddl}")
        if version < SCHEMA_VERSION:
            self.conn.executescript(
                "DROP TRIGGER IF EXISTS items_ai; DROP TRIGGER IF EXISTS items_ad;"
                "DROP TRIGGER IF EXISTS items_au; DROP TABLE IF EXISTS items_fts;"
            )
            self.conn.executescript(FTS_SCHEMA)
            self.conn.execute("INSERT INTO items_fts(items_fts) VALUES ('rebuild')")
        if version < 3:
            # Turn existing free-text genres ("rock, pop") into genre records.
            for row in self.conn.execute("SELECT id, genre FROM items WHERE genre IS NOT NULL").fetchall():
                self._set_genres(row["id"], row["genre"])
        if version < SCHEMA_VERSION:
            self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.conn.commit()

    # ------------------------------------------------------------------ genres (internal)
    def _resolve_genres(self, value) -> list[tuple[int, str]]:
        """(id, display name) for each genre in `value`, creating new genres as needed.
        A spelling variant of an existing genre resolves to the existing one."""
        out = []
        for name in split_genres(value):
            k = genre_key(name)
            row = self.conn.execute("SELECT id, name FROM genres WHERE key = ?", (k,)).fetchone()
            if row is None:
                cur = self.conn.execute("INSERT INTO genres (name, key) VALUES (?, ?)", (name, k))
                out.append((cur.lastrowid, name))
            else:
                out.append((row["id"], row["name"]))
        return out

    def _set_genres(self, item_id, value):
        """Replace an item's genres and refresh its plain-text genre column."""
        resolved = self._resolve_genres(value)
        self.conn.execute("DELETE FROM item_genres WHERE item_id = ?", (item_id,))
        self.conn.executemany(
            "INSERT INTO item_genres (item_id, genre_id, position) VALUES (?, ?, ?)",
            [(item_id, gid, pos) for pos, (gid, _) in enumerate(resolved)],
        )
        self.conn.execute("UPDATE items SET genre = ? WHERE id = ?",
                          (", ".join(n for _, n in resolved) or None, item_id))
        self._prune_genres()

    def _refresh_genre_text(self, item_ids):
        for item_id in item_ids:
            names = [r[0] for r in self.conn.execute(
                "SELECT g.name FROM item_genres ig JOIN genres g ON g.id = ig.genre_id "
                "WHERE ig.item_id = ? ORDER BY ig.position, g.name", (item_id,))]
            self.conn.execute("UPDATE items SET genre = ? WHERE id = ?", (", ".join(names) or None, item_id))

    def _prune_genres(self):
        """Drop genres no item uses any more, so autocomplete stays clean."""
        self.conn.execute("DELETE FROM genres WHERE id NOT IN (SELECT genre_id FROM item_genres)")

    @staticmethod
    def _filters(media_type=None, genre=None, alias="items", location=None, origin=None, genre_match="any"):
        """WHERE clauses + args for the media-type, genre, location and origin filters.
        `genre` may be one genre, a list, or "Rock, Psychedelic rock". With several genres,
        genre_match="any" keeps items with at least one of them; "all" only items with every one.
        Location and origin match the whole value, ignoring case and surrounding spaces."""
        clauses, args = [], []
        if media_type:
            clauses.append(f"{alias}.media_type = ?")
            args.append(media_type)
        keys = list(dict.fromkeys(genre_key(g) for g in split_genres(genre)))
        if keys:
            marks = ", ".join("?" * len(keys))
            if genre_match == "all" and len(keys) > 1:
                clauses.append(f"(SELECT COUNT(DISTINCT g.key) FROM item_genres ig JOIN genres g ON g.id = ig.genre_id "
                               f"WHERE ig.item_id = {alias}.id AND g.key IN ({marks})) = {len(keys)}")
            else:
                clauses.append(f"EXISTS (SELECT 1 FROM item_genres ig JOIN genres g ON g.id = ig.genre_id "
                               f"WHERE ig.item_id = {alias}.id AND g.key IN ({marks}))")
            args.extend(keys)
        for field, value in (("location", location), ("origin", origin)):
            if value and value.strip():
                clauses.append(f"lower(trim(json_extract({alias}.extra, '$.{field}'))) = lower(?)")
                args.append(value.strip())
        return clauses, args

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize(media_type, fields: dict, extra: dict | None):
        media_type = (media_type or "").lower().strip()
        if media_type not in MEDIA_TYPES:
            raise ValueError(f"media_type must be one of {MEDIA_TYPES}, got {media_type!r}")
        core = {k: _clean(fields.get(k)) for k in CORE_FIELDS if k in fields}
        if core.get("year") is not None:
            try:
                core["year"] = int(str(core["year"])[:4])
            except ValueError:
                core["year"] = None
        clean_extra = {}
        for k, v in (extra or {}).items():
            v = _clean(v)
            if v is not None:
                clean_extra[k] = v
        return media_type, core, clean_extra

    @staticmethod
    def to_dict(row) -> dict:
        d = dict(row)
        d["extra"] = json.loads(d.get("extra") or "{}")
        d["genres"] = [g for g in (d.get("genre") or "").split(", ") if g]
        d.pop("extra_text", None)
        return d

    @staticmethod
    def _embedding_text(item: dict) -> str:
        bits = [f"{item['media_type']}: {item['title']}"]
        if item.get("creator"):
            bits.append(f"by {item['creator']}")
        for label in ("genre", "tags", "notes"):
            if item.get(label):
                bits.append(f"{label}: {item[label]}")
        extra = _flatten_extra(item.get("extra") or {})
        if extra:
            bits.append(extra)
        return ". ".join(bits)

    # ------------------------------------------------------------------ write
    def add(self, media_type, title, extra=None, **fields) -> int:
        """Add an item. `genre` may be a list or a string like "Hip-Hop, Jazz rap"."""
        genres = fields.pop("genre", None)
        fields["title"] = title
        media_type, core, extra = self._normalize(media_type, fields, extra)
        if not core.get("title"):
            raise ValueError("title is required")
        cols = ["media_type", *core.keys(), "extra", "extra_text"]
        vals = [media_type, *core.values(), json.dumps(extra), _flatten_extra(extra)]
        cur = self.conn.execute(
            f"INSERT INTO items ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", vals
        )
        self._set_genres(cur.lastrowid, genres)
        self.conn.commit()
        self._embed_items([cur.lastrowid])
        return cur.lastrowid

    def update(self, item_id, extra=None, media_type=None, **fields) -> bool:
        """Change only the fields given. Passing `extra` replaces the whole extra dict;
        passing `genre` replaces the item's whole genre list."""
        current = self.get(item_id)
        if current is None:
            return False
        has_genre = "genre" in fields
        genres = fields.pop("genre", None)
        media_type, core, extra_clean = self._normalize(
            media_type or current["media_type"], fields, extra if extra is not None else current["extra"]
        )
        if "title" in core and not core["title"]:
            raise ValueError("title is required")
        sets = {"media_type": media_type, **core, "extra": json.dumps(extra_clean),
                "extra_text": _flatten_extra(extra_clean)}
        assignments = ", ".join(f"{k} = ?" for k in sets) + ", updated_at = CURRENT_TIMESTAMP"
        self.conn.execute(f"UPDATE items SET {assignments} WHERE id = ?", [*sets.values(), item_id])
        if has_genre:
            self._set_genres(item_id, genres)
        self.conn.commit()
        if current["extra"].get("cover_url") != extra_clean.get("cover_url"):
            self._discard_cover(current["extra"].get("cover_url"))
        self._embed_items([item_id])
        return True

    def remove(self, item_id) -> bool:
        current = self.get(item_id)
        cur = self.conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        self._prune_genres()
        self.conn.commit()
        if current:
            self._discard_cover(current["extra"].get("cover_url"))
        return cur.rowcount > 0

    # ------------------------------------------------------------------ cover images
    @property
    def covers_dir(self) -> Path | None:
        return covers.covers_dir_for(self.db_path)

    def cover_file(self, filename) -> Path | None:
        """Path of an uploaded cover, or None if the name isn't one of ours or doesn't exist."""
        name = covers.filename_from_url(covers.URL_PREFIX + str(filename))
        if not name or self.covers_dir is None:
            return None
        path = self.covers_dir / name
        return path if path.is_file() else None

    def set_cover_image(self, item_id, data: bytes) -> str:
        """Store an uploaded image as this item's cover. Raises covers.CoverError for bad files."""
        item = self.get(item_id)
        if item is None:
            raise ValueError(f"No item with id {item_id}")
        if self.covers_dir is None:
            raise covers.CoverError("Image uploads need a database file, not an in-memory catalog.")
        image, ext = covers.prepare(data)
        self.covers_dir.mkdir(parents=True, exist_ok=True)
        # A fresh name each time, so browsers never show a stale cached cover.
        name = f"{item_id}-{uuid.uuid4().hex[:10]}.{ext}"
        (self.covers_dir / name).write_bytes(image)
        url = covers.URL_PREFIX + name
        self.update(item_id, extra={**item["extra"], "cover_url": url})
        return url

    def clear_cover(self, item_id) -> bool:
        item = self.get(item_id)
        if item is None or not item["extra"].get("cover_url"):
            return False
        extra = {k: v for k, v in item["extra"].items() if k != "cover_url"}
        return self.update(item_id, extra=extra)

    def _discard_cover(self, url):
        """Called when a cover stops being used; tidies up unused cover files."""
        if covers.filename_from_url(url):
            self.prune_covers()

    def prune_covers(self) -> int:
        """Delete uploaded cover files that no item uses (copies can share one file).

        A file that can't be deleted right now (on Windows, one that is still open,
        e.g. being sent to a browser) is skipped and picked up on a later run.
        Returns the number of files deleted.
        """
        if self.covers_dir is None or not self.covers_dir.is_dir():
            return 0
        in_use = {covers.filename_from_url(r[0]) for r in self.conn.execute(
            "SELECT json_extract(extra, '$.cover_url') FROM items")}
        deleted = 0
        for path in self.covers_dir.iterdir():
            if path.is_file() and covers.filename_from_url(covers.URL_PREFIX + path.name) and path.name not in in_use:
                try:
                    path.unlink()
                    deleted += 1
                except OSError:
                    pass  # in use or locked; try again next time
        return deleted

    # ------------------------------------------------------------------ genres (public)
    def genres(self, media_type=None) -> list[dict]:
        """Every genre in use with its item count, most used first."""
        clauses, args = self._filters(media_type, None)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            "SELECT g.name, COUNT(*) AS count FROM genres g "
            "JOIN item_genres ig ON ig.genre_id = g.id JOIN items ON items.id = ig.item_id "
            f"{where} GROUP BY g.id ORDER BY count DESC, g.name COLLATE NOCASE", args)
        return [dict(r) for r in rows]

    def rename_genre(self, old, new) -> int:
        """Rename a genre on every item. If `new` already exists, the two are merged
        (use this to fold spelling variants together). Returns items affected."""
        old_row = self.conn.execute("SELECT id FROM genres WHERE key = ?", (genre_key(old),)).fetchone()
        if old_row is None:
            raise ValueError(f"No genre called {old!r}")
        names = split_genres(new)
        if len(names) != 1:
            raise ValueError("Give exactly one new genre name")
        new_name = names[0]
        affected = [r[0] for r in self.conn.execute(
            "SELECT item_id FROM item_genres WHERE genre_id = ?", (old_row["id"],))]
        target = self.conn.execute("SELECT id FROM genres WHERE key = ?", (genre_key(new_name),)).fetchone()
        if target and target["id"] != old_row["id"]:
            self.conn.execute(
                "INSERT OR IGNORE INTO item_genres (item_id, genre_id, position) "
                "SELECT item_id, ?, position FROM item_genres WHERE genre_id = ?", (target["id"], old_row["id"]))
            self.conn.execute("DELETE FROM genres WHERE id = ?", (old_row["id"],))
        else:
            self.conn.execute("UPDATE genres SET name = ?, key = ? WHERE id = ?",
                              (new_name, genre_key(new_name), old_row["id"]))
        self._refresh_genre_text(affected)
        self.conn.commit()
        self._embed_items(affected)
        return len(affected)

    def remove_genre(self, name) -> int:
        """Take a genre off every item that has it. Returns items affected."""
        row = self.conn.execute("SELECT id FROM genres WHERE key = ?", (genre_key(name),)).fetchone()
        if row is None:
            return 0
        affected = [r[0] for r in self.conn.execute(
            "SELECT item_id FROM item_genres WHERE genre_id = ?", (row["id"],))]
        self.conn.execute("DELETE FROM genres WHERE id = ?", (row["id"],))
        self._refresh_genre_text(affected)
        self.conn.commit()
        self._embed_items(affected)
        return len(affected)

    def import_csv(self, source, ids_out: list | None = None) -> int:
        """Import from a CSV path or file object. Returns how many items were added;
        pass a list as `ids_out` to also get their ids (e.g. to auto-fill them).

        Header must include media_type and title. Other columns may be any core
        field or any type-specific field for that row's type (e.g. isbn, platform).
        """
        f = open(source, newline="", encoding="utf-8") if isinstance(source, (str, Path)) else source
        count = 0
        try:
            for raw in csv.DictReader(f):
                row = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
                media_type, title = row.pop("media_type", "").lower(), row.pop("title", "")
                if not title or media_type not in MEDIA_TYPES:
                    continue
                core = {k: v for k, v in row.items() if k in CORE_FIELDS}
                extra = {k: v for k, v in row.items() if k in EXTRA_FIELDS[media_type] or k in COPY_FIELDS}
                item_id = self.add(media_type, title, extra=extra, **core)
                if ids_out is not None:
                    ids_out.append(item_id)
                count += 1
        finally:
            if f is not source:
                f.close()
        return count

    def export_csv(self, media_type=None) -> str:
        """The catalog (or one media type) as CSV text; copy and type-specific fields
        become their own columns."""
        types = [media_type] if media_type else MEDIA_TYPES
        extra_cols = [f for f in dict.fromkeys(f for t in types for f in EXTRA_FIELDS[t]) if f not in COPY_FIELDS]
        if not media_type:
            extra_cols = sorted(extra_cols)
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=["media_type", *CORE_FIELDS, *COPY_FIELDS, *extra_cols],
                           extrasaction="ignore")
        w.writeheader()
        for item in self.list(media_type):
            w.writerow({**{k: item.get(k) for k in ("media_type", *CORE_FIELDS)}, **item["extra"]})
        return out.getvalue()

    # ------------------------------------------------------------------ read
    def get(self, item_id) -> dict | None:
        row = self.conn.execute("SELECT * FROM items WHERE id = ?", (item_id,)).fetchone()
        return self.to_dict(row) if row else None

    # Orderings for list(). Blank years and creators sort last.
    # Titles ignore a leading "The" (The Shining files under S).
    SORTS = {
        "type": "media_type, sort_name(title)",
        "title": "sort_name(title)",
        "creator": "sort_name(title)",   # then re-sorted in Python by file-under name; see list()
        "newest": "year IS NULL, year DESC, sort_name(title)",
        "oldest": "year IS NULL, year, sort_name(title)",
        "added": "added_at DESC, id DESC",
    }

    def list(self, media_type=None, genre=None, location=None, origin=None, sort=None,
             genre_match="any") -> list[dict]:
        """Items matching the filters. `sort` is one of SORTS; by default everything is
        grouped by media type, and a single type is sorted by title."""
        clauses, args = self._filters(media_type, genre, location=location, origin=origin,
                                      genre_match=genre_match)
        sql = "SELECT * FROM items" + (f" WHERE {' AND '.join(clauses)}" if clauses else "")
        order = self.SORTS.get(sort) or self.SORTS["title" if media_type else "type"]
        items = [self.to_dict(r) for r in self.conn.execute(f"{sql} ORDER BY {order}", args)]
        if sort == "creator":
            # Python sort, because filing depends on each creator's settings: people by
            # surname, bands as written but ignoring "The". Within a creator, oldest first.
            overrides = self._sort_overrides()
            items.sort(key=lambda it: (not it.get("creator"), self.file_under_key(it, overrides),
                                       it.get("year") is None, it.get("year") or 0, sort_name(it["title"])))
        return items

    # ------------------------------------------------------------------ filing creators
    def _sort_overrides(self) -> dict[str, str]:
        return {r["key"]: r["sort_as"] for r in self.conn.execute("SELECT key, sort_as FROM sort_names")}

    @staticmethod
    def automatic_file_under(name, media_type) -> str:
        """Where a creator is filed without a File-under setting: authors and directors by
        surname ('Herbert, Frank'); artists and studios as written ('The Beatles')."""
        return surname_first(name) if media_type in PERSON_TYPES else " ".join(str(name or "").split())

    def file_under(self, name, media_type) -> str:
        """The File-under name in effect for a creator: its setting, or the automatic one."""
        row = self.conn.execute("SELECT sort_as FROM sort_names WHERE key = ?", (name_key(name),)).fetchone()
        return row["sort_as"] if row else self.automatic_file_under(name, media_type)

    def file_under_setting(self, name) -> str | None:
        row = self.conn.execute("SELECT sort_as FROM sort_names WHERE key = ?", (name_key(name),)).fetchone()
        return row["sort_as"] if row else None

    def set_file_under(self, name, sort_as, media_type=None) -> None:
        """Set where a creator is filed, for every item with that creator. A blank value,
        or one matching the automatic filing, removes the setting."""
        key = name_key(name)
        if not key:
            return
        sort_as = " ".join(str(sort_as or "").split())
        # "Beatles, The" (MusicBrainz style) files the same as automatic "The Beatles".
        same = lambda a, b: sort_name(re.sub(r",\s*the$", "", a, flags=re.I)) == sort_name(b)
        if not sort_as or (media_type and same(sort_as, self.automatic_file_under(name, media_type))):
            self.conn.execute("DELETE FROM sort_names WHERE key = ?", (key,))
        else:
            self.conn.execute("INSERT OR REPLACE INTO sort_names (key, sort_as) VALUES (?, ?)", (key, sort_as))
        self.conn.commit()

    def creators(self, media_type) -> list[dict]:
        """Every creator of a media type, in filing order, for the Artists page:
        {name, count, setting, file_under, surname, kind ('band', 'solo' or 'custom'), letter}."""
        groups: dict[str, dict] = {}
        for name, n in self.conn.execute(
                "SELECT creator, COUNT(*) FROM items WHERE media_type = ? AND trim(coalesce(creator, '')) != '' "
                "GROUP BY creator", (media_type,)):
            g = groups.setdefault(name_key(name), {"count": 0, "spellings": {}})
            g["count"] += n
            g["spellings"][" ".join(name.split())] = g["spellings"].get(" ".join(name.split()), 0) + n
        overrides = self._sort_overrides()
        out = []
        for key, g in groups.items():
            name = max(g["spellings"], key=g["spellings"].get)
            setting = overrides.get(key)
            surname = surname_first(name)
            filed = setting or self.automatic_file_under(name, media_type)
            if not setting:
                kind = "solo" if media_type in PERSON_TYPES else "band"
            elif sort_name(setting) == sort_name(surname):
                kind = "solo"
            else:
                kind = "custom"
            first = sort_name(filed)[:1].upper()
            out.append({"name": name, "count": g["count"], "setting": setting, "file_under": filed,
                        "surname": surname, "kind": kind, "one_word": surname == name,
                        "letter": first if first.isalpha() else "#", "_key": sort_name(filed)})
        out.sort(key=lambda c: (c["letter"] == "#", c["_key"]))
        return out

    def file_under_key(self, item, overrides=None) -> str:
        name = item.get("creator")
        overrides = self._sort_overrides() if overrides is None else overrides
        filed = overrides.get(name_key(name)) or self.automatic_file_under(name, item["media_type"])
        return sort_name(filed)

    def copy_values(self, field, media_type=None) -> list[dict]:
        """Every distinct value used for a copy field (e.g. all your locations), with
        how many items have it, most used first. Spelling differences in capital
        letters or spaces count as the same value; the most common spelling is shown."""
        if field not in COPY_FIELDS:
            raise ValueError(f"Unknown copy field {field!r}")
        clauses, args = self._filters(media_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        groups: dict[str, dict] = {}
        for (value,) in self.conn.execute(f"SELECT json_extract(extra, '$.{field}') FROM items {where}", args):
            value = " ".join(str(value or "").split())
            if not value:
                continue
            g = groups.setdefault(value.casefold(), {"count": 0, "spellings": {}})
            g["count"] += 1
            g["spellings"][value] = g["spellings"].get(value, 0) + 1
        out = [{"name": max(g["spellings"], key=g["spellings"].get), "count": g["count"]} for g in groups.values()]
        return sorted(out, key=lambda v: (-v["count"], v["name"].casefold()))

    def counts(self) -> dict:
        return {r[0]: r[1] for r in self.conn.execute("SELECT media_type, COUNT(*) FROM items GROUP BY media_type")}

    # ------------------------------------------------------------------ search
    def _keyword_scores(self, query, filters: dict, limit) -> dict[int, float]:
        terms = re.findall(r"\w+", query.lower())
        if not terms:
            return {}
        fts_query = " OR ".join(f'"{t}"*' for t in terms)
        weights = ", ".join(str(w) for w in WEIGHTS.values())
        sql = f"""SELECT items.id, bm25(items_fts, {weights}) AS score
                  FROM items_fts JOIN items ON items.id = items_fts.rowid
                  WHERE items_fts MATCH ?"""
        args = [fts_query]
        clauses, fargs = self._filters(**filters)
        for c in clauses:
            sql += f" AND {c}"
        args += fargs
        sql += " ORDER BY score LIMIT ?"
        args.append(limit)
        # bm25() is negative; flip so bigger = better.
        return {r["id"]: -r["score"] for r in self.conn.execute(sql, args)}

    def _semantic_scores(self, query, filters: dict, min_similarity) -> dict[int, float]:
        import numpy as np

        sql = "SELECT e.item_id, e.vector FROM embeddings e JOIN items i ON i.id = e.item_id WHERE e.model = ?"
        args = [self.embedder.model_name]
        clauses, fargs = self._filters(**filters, alias="i")
        for c in clauses:
            sql += f" AND {c}"
        args += fargs
        rows = self.conn.execute(sql, args).fetchall()
        if not rows:
            return {}
        ids = [r["item_id"] for r in rows]
        matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        q = self.embedder.encode([query])[0]
        sims = matrix @ q  # vectors are unit length, so this is cosine similarity
        return {i: float(s) for i, s in zip(ids, sims) if s >= min_similarity}

    def search(self, query, media_type=None, limit=10, semantic=True,
               alpha=0.5, min_similarity=0.3, genre=None, location=None, origin=None,
               genre_match="any") -> list[dict]:
        """Best matches first. Each result gets `score` (0-1) and `match`.

        With an embedder and semantic=True this is hybrid search:
            score = alpha * keyword_score + (1 - alpha) * semantic_similarity
        where keyword_score is BM25 scaled so the best keyword hit is 1.0.
        Items can surface through either signal; matching both ranks highest.
        """
        filters = {"media_type": media_type, "genre": genre, "location": location, "origin": origin,
                   "genre_match": genre_match}
        kw = self._keyword_scores(query, filters, max(limit * 5, 50))
        top_kw = max(kw.values(), default=0) or 1.0
        kw = {i: s / top_kw for i, s in kw.items()}

        sem = {}
        if semantic and self.embedder is not None and query.strip():
            sem = self._semantic_scores(query, filters, min_similarity)

        if sem:
            ids = set(kw) | set(sem)
            combined = {i: alpha * kw.get(i, 0.0) + (1 - alpha) * sem.get(i, 0.0) for i in ids}
        else:
            combined = kw

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        results = []
        for item_id, score in ranked:
            item = self.get(item_id)
            item["score"] = round(score, 4)
            item["match"] = ("both" if item_id in kw and item_id in sem
                             else "keyword" if item_id in kw else "semantic")
            results.append(item)
        return results

    _TITLE_STOPWORDS = {"the", "a", "an", "of", "and", "part", "vol", "volume", "edition", "to", "in", "on"}

    @classmethod
    def _title_words(cls, title) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", (title or "").casefold())
                if len(w) > 2 and w not in cls._TITLE_STOPWORDS and not w.isdigit()}

    def similar(self, item_id, limit=6) -> list[dict]:
        """Items related to this one, best first, each with a short `reason`.

        Signals: same creator or other shared credits (composer, performer...), related title (e.g. the Dune novel and the Dune films),
        shared genres, shared tags, and, when semantic search is on, similar subject matter.
        """
        item = self.get(item_id)
        if item is None:
            return []
        my_genres = {genre_key(g): g for g in item["genres"]}
        my_creator = (item.get("creator") or "").strip().casefold()
        people = lambda it: {n.casefold(): n for f in PEOPLE_FIELDS for n in split_names(it["extra"].get(f))}
        my_people = {**people(item), **{n.casefold(): n for n in split_names(item.get("creator"))}}
        my_words = self._title_words(item["title"])
        tag_set = lambda text: {genre_key(t): t.strip() for t in (text or "").split(",") if genre_key(t)}
        my_tags = tag_set(item.get("tags"))

        sims = {}
        if self.embedder is not None:
            row = self.conn.execute("SELECT vector FROM embeddings WHERE item_id = ? AND model = ?",
                                    (item_id, self.embedder.model_name)).fetchone()
            if row:
                import numpy as np
                mine = np.frombuffer(row["vector"], dtype=np.float32)
                for r in self.conn.execute("SELECT item_id, vector FROM embeddings WHERE model = ? AND item_id != ?",
                                           (self.embedder.model_name, item_id)):
                    sims[r["item_id"]] = float(np.frombuffer(r["vector"], dtype=np.float32) @ mine)

        scored = []
        for other in self.list():
            if other["id"] == item_id:
                continue
            score, reasons = 0.0, []
            if my_creator and (other.get("creator") or "").strip().casefold() == my_creator:
                score += 3
                reasons.append(f"Also by {item['creator']}")
            else:
                # Shared composer, illustrator, performer, designer... (e.g. Hans Zimmer scores)
                theirs = {**people(other), **{n.casefold(): n for n in split_names(other.get("creator"))}}
                shared_people = [my_people[k] for k in my_people if k in theirs]
                if shared_people:
                    score += 2.5
                    reasons.append("Also with " + ", ".join(shared_people[:2]))
            if my_words & self._title_words(other["title"]):
                score += 2.5
                reasons.append("Related title")
            shared = [g for g in other["genres"] if genre_key(g) in my_genres]
            if shared:
                score += len(shared)
                reasons.append(", ".join(shared))
            tags = [t for k, t in tag_set(other.get("tags")).items() if k in my_tags]
            if tags:
                score += 0.75 * len(tags)
                reasons.append("Tagged " + ", ".join(tags))
            sim = sims.get(other["id"], 0.0)
            if sim >= 0.35:
                score += 3 * sim
                if not reasons:
                    reasons.append("Similar subject")
            if score > 0:
                other["reason"] = " · ".join(reasons)
                scored.append((score, other))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["title"].casefold()))
        return [other for _, other in scored[:limit]]

    def suggest(self, query, n=3) -> list[str]:
        """Typo fallback: closest titles/creators by string similarity."""
        names = set()
        for row in self.conn.execute("SELECT title, creator FROM items"):
            names.update(x for x in row if x)
        lookup = {name.lower(): name for name in names}
        return [lookup[h] for h in difflib.get_close_matches(query.lower(), lookup, n=n, cutoff=0.6)]

    # ------------------------------------------------------------------ embeddings
    def _embed_items(self, ids, force=False) -> int:
        """Compute embeddings for the given items (skipping unchanged ones). Returns count."""
        if self.embedder is None or not ids:
            return 0
        todo = []
        for item_id in ids:
            item = self.get(item_id)
            if item is None:
                continue
            text = self._embedding_text(item)
            h = hashlib.sha1(text.encode()).hexdigest()
            if not force:
                row = self.conn.execute(
                    "SELECT model, text_hash FROM embeddings WHERE item_id = ?", (item_id,)
                ).fetchone()
                if row and row["model"] == self.embedder.model_name and row["text_hash"] == h:
                    continue
            todo.append((item_id, text, h))
        if not todo:
            return 0
        vectors = self.embedder.encode([t for _, t, _ in todo])
        self.conn.executemany(
            "INSERT OR REPLACE INTO embeddings (item_id, model, text_hash, vector) VALUES (?, ?, ?, ?)",
            [(i, self.embedder.model_name, h, v.astype("float32").tobytes())
             for (i, _, h), v in zip(todo, vectors)],
        )
        self.conn.commit()
        return len(todo)

    def reindex(self, force=False, batch_size=64) -> int:
        """(Re)build semantic embeddings for every item. Needed once after enabling semantic search."""
        ids = [r[0] for r in self.conn.execute("SELECT id FROM items")]
        done = 0
        for start in range(0, len(ids), batch_size):
            done += self._embed_items(ids[start:start + batch_size], force=force)
        return done
