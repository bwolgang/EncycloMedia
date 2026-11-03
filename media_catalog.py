#!/usr/bin/env python3
"""
media_catalog.py - Catalog books, music, movies, and games with ranked keyword search.

Uses only the Python standard library:
  * SQLite stores the catalog (one file, no server).
  * SQLite's FTS5 full-text index handles keyword search and ranks results with BM25.
  * difflib gives a "did you mean" fallback for typos.

Usage examples:
  python media_catalog.py add book "Dune" --creator "Frank Herbert" --year 1965 --genre "sci-fi" --tags "desert, politics"
  python media_catalog.py add game "Hades" --creator "Supergiant" --format "Switch" --genre roguelike
  python media_catalog.py search "herbert desert"
  python media_catalog.py search "sci fi" --type book --limit 5
  python media_catalog.py list --type music
  python media_catalog.py import my_collection.csv
  python media_catalog.py remove 12
"""
import argparse
import csv
import difflib
import re
import sqlite3
from pathlib import Path

DB_DEFAULT = Path.home() / ".media_catalog.db"
MEDIA_TYPES = ("book", "music", "movie", "game")
FIELDS = ("title", "creator", "year", "genre", "format", "tags", "notes")

# Relative importance of each searchable column when ranking (order matches items_fts).
WEIGHTS = {"title": 10.0, "creator": 6.0, "genre": 3.0, "tags": 3.0, "notes": 1.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id         INTEGER PRIMARY KEY,
    media_type TEXT NOT NULL CHECK (media_type IN ('book','music','movie','game')),
    title      TEXT NOT NULL,
    creator    TEXT,          -- author / artist / director / developer
    year       INTEGER,
    genre      TEXT,
    format     TEXT,          -- hardcover, vinyl, Blu-ray, PS5, ...
    tags       TEXT,          -- free-form, comma separated
    notes      TEXT,
    added_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Full-text index that mirrors the searchable columns of `items`.
-- porter = stemming (run/running/runs match), unicode61 = case/accent folding.
CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
    title, creator, genre, tags, notes,
    content='items', content_rowid='id',
    tokenize='porter unicode61 remove_diacritics 2'
);

-- Triggers keep the index in sync automatically.
CREATE TRIGGER IF NOT EXISTS items_ai AFTER INSERT ON items BEGIN
    INSERT INTO items_fts(rowid, title, creator, genre, tags, notes)
    VALUES (new.id, new.title, new.creator, new.genre, new.tags, new.notes);
END;
CREATE TRIGGER IF NOT EXISTS items_ad AFTER DELETE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, creator, genre, tags, notes)
    VALUES ('delete', old.id, old.title, old.creator, old.genre, old.tags, old.notes);
END;
CREATE TRIGGER IF NOT EXISTS items_au AFTER UPDATE ON items BEGIN
    INSERT INTO items_fts(items_fts, rowid, title, creator, genre, tags, notes)
    VALUES ('delete', old.id, old.title, old.creator, old.genre, old.tags, old.notes);
    INSERT INTO items_fts(rowid, title, creator, genre, tags, notes)
    VALUES (new.id, new.title, new.creator, new.genre, new.tags, new.notes);
END;
"""


class Catalog:
    def __init__(self, db_path=DB_DEFAULT):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    # ---------- write ----------
    def add(self, media_type, title, **fields):
        media_type = media_type.lower()
        if media_type not in MEDIA_TYPES:
            raise ValueError(f"media_type must be one of {MEDIA_TYPES}")
        data = {k: fields.get(k) or None for k in FIELDS if k != "title"}
        if data["year"] is not None:
            data["year"] = int(data["year"])
        cur = self.conn.execute(
            "INSERT INTO items (media_type, title, creator, year, genre, format, tags, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (media_type, title, data["creator"], data["year"], data["genre"],
             data["format"], data["tags"], data["notes"]),
        )
        self.conn.commit()
        return cur.lastrowid

    def remove(self, item_id):
        cur = self.conn.execute("DELETE FROM items WHERE id = ?", (item_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def import_csv(self, path):
        """CSV needs a header row with media_type,title and any of the other FIELDS."""
        count = 0
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                if not row.get("title") or not row.get("media_type"):
                    continue
                self.add(row.pop("media_type"), row.pop("title"), **row)
                count += 1
        return count

    # ---------- read ----------
    def list(self, media_type=None):
        sql, args = "SELECT * FROM items", []
        if media_type:
            sql += " WHERE media_type = ?"
            args.append(media_type)
        return self.conn.execute(sql + " ORDER BY media_type, title", args).fetchall()

    def search(self, query, media_type=None, limit=10):
        """Rank items by how well they match the keywords (BM25, weighted by column).

        Every keyword is OR'd together and prefix-matched, so items matching more
        keywords, rarer keywords, or matching in the title rank highest, while
        partial matches still show up lower in the list.
        """
        terms = re.findall(r"\w+", query.lower())
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"*' for t in terms)
        weights = ", ".join(str(w) for w in WEIGHTS.values())
        sql = f"""
            SELECT items.*, bm25(items_fts, {weights}) AS score
            FROM items_fts JOIN items ON items.id = items_fts.rowid
            WHERE items_fts MATCH ?
        """
        args = [fts_query]
        if media_type:
            sql += " AND items.media_type = ?"
            args.append(media_type)
        sql += " ORDER BY score LIMIT ?"   # bm25: lower (more negative) = better
        args.append(limit)
        return self.conn.execute(sql, args).fetchall()

    def suggest(self, query, n=3):
        """Typo fallback: closest titles/creators by string similarity."""
        names = set()
        for row in self.conn.execute("SELECT title, creator FROM items"):
            names.update(x for x in row if x)
        lookup = {name.lower(): name for name in names}
        hits = difflib.get_close_matches(query.lower(), lookup, n=n, cutoff=0.6)
        return [lookup[h] for h in hits]


# ---------- CLI ----------
def show(rows, with_score=False):
    if not rows:
        print("No items.")
        return
    for r in rows:
        line = f"[{r['id']:>4}] {r['media_type']:<5} {r['title']}"
        if r["creator"]:
            line += f" - {r['creator']}"
        if r["year"]:
            line += f" ({r['year']})"
        extras = ", ".join(x for x in (r["genre"], r["format"]) if x)
        if extras:
            line += f"  [{extras}]"
        if with_score:
            line += f"  relevance={-r['score']:.2f}"
        print(line)


def main():
    p = argparse.ArgumentParser(description="Media catalog with ranked keyword search")
    p.add_argument("--db", default=str(DB_DEFAULT), help="database file (default: ~/.media_catalog.db)")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add an item")
    a.add_argument("media_type", choices=MEDIA_TYPES)
    a.add_argument("title")
    for f in FIELDS[1:]:
        a.add_argument(f"--{f}")

    s = sub.add_parser("search", help="keyword search, best matches first")
    s.add_argument("query")
    s.add_argument("--type", choices=MEDIA_TYPES)
    s.add_argument("--limit", type=int, default=10)

    l = sub.add_parser("list", help="list everything")
    l.add_argument("--type", choices=MEDIA_TYPES)

    r = sub.add_parser("remove", help="remove an item by id")
    r.add_argument("id", type=int)

    i = sub.add_parser("import", help="bulk import from CSV")
    i.add_argument("csv_path")

    args = p.parse_args()
    cat = Catalog(args.db)

    if args.cmd == "add":
        fields = {f: getattr(args, f) for f in FIELDS[1:]}
        print(f"Added item #{cat.add(args.media_type, args.title, **fields)}")
    elif args.cmd == "search":
        rows = cat.search(args.query, args.type, args.limit)
        show(rows, with_score=True)
        if not rows:
            hint = cat.suggest(args.query)
            if hint:
                print("Did you mean: " + ", ".join(hint) + "?")
    elif args.cmd == "list":
        show(cat.list(args.type))
    elif args.cmd == "remove":
        print("Removed." if cat.remove(args.id) else "No item with that id.")
    elif args.cmd == "import":
        print(f"Imported {cat.import_csv(args.csv_path)} items.")


if __name__ == "__main__":
    main()
