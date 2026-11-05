"""
Command line interface.   python -m encyclomedia <command> ...

  add      add an item (optionally auto-filled with --lookup)
  lookup   search online databases and optionally add a match
  search   ranked search of your catalog
  list     list items
  show     show every field of one item
  edit     change fields of an item
  remove   delete an item
  import   bulk import from CSV
  export   write the catalog to CSV
  genres         list genres with item counts
  locations      list storage locations (or --origins) with item counts
  genre-rename   rename a genre everywhere (or merge it into another)
  genre-remove   take a genre off every item
  reindex  build embeddings for semantic search
  serve    start the web interface
"""
from __future__ import annotations

import argparse
import sys

from . import metadata, semantic
from .catalog import COPY_FIELDS, CORE_FIELDS, CREDIT_FIELDS, CREDIT_LABELS, DB_DEFAULT, EXTRA_FIELDS, MEDIA_TYPES, Catalog


def parse_kv(pairs) -> dict:
    out = {}
    for pair in pairs or []:
        if "=" not in pair:
            sys.exit(f"--field expects key=value, got {pair!r}")
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def one_line(it: dict) -> str:
    line = f"[{it['id']:>4}] {it['media_type']:<5} {it['title']}"
    if it.get("creator"):
        line += f" - {it['creator']}"
    if it.get("year"):
        line += f" ({it['year']})"
    extras = ", ".join(x for x in (it.get("genre"), it.get("format"), it["extra"].get("platform")) if x)
    if extras:
        line += f"  [{extras}]"
    if "score" in it:
        line += f"  {it['score']:.2f} {it['match']}"
    return line


def show_candidates(cands):
    for n, c in enumerate(cands, 1):
        bits = [c.get("creator"), str(c["year"]) if c.get("year") else None,
                c["extra"].get("platform") or c.get("format"), c["source"]]
        print(f"  {n}. {c['title']}  -  " + " · ".join(b for b in bits if b))


def add_core_args(p):
    for f in CORE_FIELDS:
        if f != "title":
            p.add_argument(f"--{f}")
    copy_help = {"edition": "e.g. \"First edition\", \"Director's cut\"",
                 "location": "where it's stored, e.g. \"Shelf A2\"",
                 "origin": "where you got it, e.g. \"Gift from Chris\"",
                 "condition": "condition notes, e.g. \"Like new\""}
    for f in COPY_FIELDS:
        p.add_argument(f"--{f}", help=copy_help[f])
    used_by = {f: t for t, fs in CREDIT_FIELDS.items() for f in fs}
    for f in used_by:
        p.add_argument(f"--{f}", help=f"{CREDIT_LABELS[f].lower()} ({used_by[f]}s); separate several with commas")
    p.add_argument("--field", action="append", metavar="KEY=VALUE",
                   help="type-specific field, repeatable (e.g. --field isbn=978... --field platform=PS5)")


def copy_args(args) -> dict:
    """Copy and credit options given on the command line, as extra fields."""
    names = (*COPY_FIELDS, *{f for fs in CREDIT_FIELDS.values() for f in fs})
    return {f: getattr(args, f) for f in names if getattr(args, f, None) is not None}


def check_credits(media_type, extra):
    for t, fields in CREDIT_FIELDS.items():
        for f in fields:
            if f in extra and f not in CREDIT_FIELDS[media_type]:
                sys.exit(f"--{f} is for {t}s; a {media_type} doesn't have {CREDIT_LABELS[f].lower()}.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="encyclomedia", description="Media catalog with smart search")
    p.add_argument("--db", default=str(DB_DEFAULT), help=f"database file (default {DB_DEFAULT})")
    p.add_argument("--no-semantic", action="store_true", help="don't load the embedding model")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add an item")
    a.add_argument("media_type", choices=MEDIA_TYPES)
    a.add_argument("title", help="title, or ISBN/barcode when used with --lookup")
    a.add_argument("--lookup", action="store_true", help="auto-fill from the best online match first")
    add_core_args(a)

    lk = sub.add_parser("lookup", help="search online databases")
    lk.add_argument("media_type", choices=MEDIA_TYPES)
    lk.add_argument("query", help="title, or ISBN/barcode")
    lk.add_argument("--add", type=int, metavar="N", help="add candidate number N to the catalog")

    s = sub.add_parser("search", help="ranked search")
    s.add_argument("query")
    s.add_argument("--type", choices=MEDIA_TYPES)
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--keyword-only", action="store_true", help="skip semantic matching")
    s.add_argument("--genre", help="only items with this genre")
    s.add_argument("--location", help="only items stored here")
    s.add_argument("--origin", help="only items acquired from here")

    ls = sub.add_parser("list", help="list items")
    ls.add_argument("--type", choices=MEDIA_TYPES)
    ls.add_argument("--genre", help="only items with this genre")
    ls.add_argument("--location", help="only items stored here")
    ls.add_argument("--origin", help="only items acquired from here")

    lv = sub.add_parser("locations", help="list storage locations (or --origins) with counts")
    lv.add_argument("--origins", action="store_true", help="list where items came from instead")

    gl = sub.add_parser("genres", help="list genres with counts")
    gl.add_argument("--type", choices=MEDIA_TYPES)

    gr = sub.add_parser("genre-rename", help="rename a genre everywhere; merges if NEW exists")
    gr.add_argument("old")
    gr.add_argument("new")

    gx = sub.add_parser("genre-remove", help="take a genre off every item")
    gx.add_argument("name")

    sh = sub.add_parser("show", help="show one item")
    sh.add_argument("id", type=int)

    e = sub.add_parser("edit", help="edit an item")
    e.add_argument("id", type=int)
    e.add_argument("--title")
    add_core_args(e)

    r = sub.add_parser("remove", help="remove an item")
    r.add_argument("id", type=int)

    im = sub.add_parser("import", help="import CSV")
    im.add_argument("csv_path")

    ex = sub.add_parser("export", help="export CSV")
    ex.add_argument("csv_path", nargs="?", help="output file (default: print)")

    ri = sub.add_parser("reindex", help="build semantic-search embeddings")
    ri.add_argument("--force", action="store_true", help="recompute every item")

    sv = sub.add_parser("serve", help="start the web interface")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=5000)

    args = p.parse_args(argv)

    needs_model = args.cmd in ("add", "lookup", "edit", "import", "search", "reindex", "serve",
                               "genre-rename", "genre-remove")
    embedder = None
    if needs_model and not args.no_semantic:
        embedder = semantic.load_embedder(quiet=args.cmd not in ("reindex", "search", "serve"))
    if args.cmd == "reindex" and embedder is None:
        sys.exit("Semantic search needs: pip install sentence-transformers")

    if args.cmd == "serve":
        from .web import create_app
        print(f"EncycloMedia running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
        create_app(args.db, embedder).run(host=args.host, port=args.port)
        return

    cat = Catalog(args.db, embedder=embedder)

    try:
        if args.cmd == "add":
            fields = {f: getattr(args, f) for f in CORE_FIELDS if f != "title" and getattr(args, f)}
            extra = {**parse_kv(args.field), **copy_args(args)}
            check_credits(args.media_type, copy_args(args))
            title = args.title
            if args.lookup:
                cands = metadata.lookup(args.media_type, args.title, limit=1)
                if not cands:
                    sys.exit("No online match found; add it without --lookup.")
                best = cands[0]
                print(f"Matched: {best['title']} ({best['source']})")
                title = best["title"]
                fields = {**{k: best[k] for k in CORE_FIELDS if k != "title" and best.get(k)}, **fields}
                extra = {**{k: v for k, v in best["extra"].items() if k in EXTRA_FIELDS[args.media_type]}, **extra}
            print(f"Added item #{cat.add(args.media_type, title, extra=extra, **fields)}")

        elif args.cmd == "lookup":
            cands = metadata.lookup(args.media_type, args.query)
            if not cands:
                print("No matches.")
                return
            show_candidates(cands)
            if args.add:
                if not 1 <= args.add <= len(cands):
                    sys.exit(f"Pick a number from 1 to {len(cands)}.")
                c = cands[args.add - 1]
                core = {k: c[k] for k in CORE_FIELDS if k != "title" and c.get(k)}
                extra = {k: v for k, v in c["extra"].items() if k in EXTRA_FIELDS[args.media_type]}
                print(f"Added item #{cat.add(args.media_type, c['title'], extra=extra, **core)}")

        elif args.cmd == "search":
            results = cat.search(args.query, args.type, args.limit, semantic=not args.keyword_only,
                                 genre=args.genre, location=args.location, origin=args.origin)
            for it in results:
                print(one_line(it))
            if not results:
                print("No matches.")
                hint = cat.suggest(args.query)
                if hint:
                    print("Did you mean: " + ", ".join(hint) + "?")

        elif args.cmd == "list":
            items = cat.list(args.type, args.genre, location=args.location, origin=args.origin)
            for it in items:
                print(one_line(it))
            if not items:
                print("No items.")

        elif args.cmd == "show":
            it = cat.get(args.id) or sys.exit("No item with that id.")
            for k in ("id", "media_type", *CORE_FIELDS, "added_at", "updated_at"):
                if it.get(k) not in (None, ""):
                    print(f"{k:>12}: {it[k]}")
            for k, v in it["extra"].items():
                if k not in COPY_FIELDS:
                    print(f"{k:>12}: {v}")
            mine = [(k, it["extra"][k]) for k in COPY_FIELDS if it["extra"].get(k)]
            if mine:
                print("   Your copy:")
                for k, v in mine:
                    print(f"{k:>12}: {v}")

        elif args.cmd == "edit":
            fields = {f: getattr(args, f) for f in CORE_FIELDS if getattr(args, f) is not None}
            current = cat.get(args.id) or sys.exit("No item with that id.")
            check_credits(current["media_type"], copy_args(args))
            changes = {**parse_kv(args.field), **copy_args(args)}
            extra = {**current["extra"], **changes} if changes else None
            cat.update(args.id, extra=extra, **fields)
            print("Saved.")

        elif args.cmd == "genres":
            rows = cat.genres(args.type)
            width = max((len(g["name"]) for g in rows), default=0)
            for g in rows:
                print(f"  {g['name']:<{width}}  {g['count']}")
            if not rows:
                print("No genres yet.")

        elif args.cmd == "locations":
            rows = cat.copy_values("origin" if args.origins else "location")
            width = max((len(v["name"]) for v in rows), default=0)
            for v in rows:
                print(f"  {v['name']:<{width}}  {v['count']}")
            if not rows:
                print("None recorded yet.")

        elif args.cmd == "genre-rename":
            n = cat.rename_genre(args.old, args.new)
            print(f"Updated {n} item{'s' if n != 1 else ''}.")

        elif args.cmd == "genre-remove":
            n = cat.remove_genre(args.name)
            print(f"Removed from {n} item{'s' if n != 1 else ''}.")

        elif args.cmd == "remove":
            print("Removed." if cat.remove(args.id) else "No item with that id.")

        elif args.cmd == "import":
            print(f"Imported {cat.import_csv(args.csv_path)} items.")

        elif args.cmd == "export":
            text = cat.export_csv()
            if args.csv_path:
                with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
                    f.write(text)
                print(f"Wrote {args.csv_path}")
            else:
                print(text, end="")

        elif args.cmd == "reindex":
            print(f"Embedded {cat.reindex(force=args.force)} items with {embedder.model_name}.")

    except (metadata.MetadataError, ValueError) as err:
        sys.exit(f"Error: {err}")


if __name__ == "__main__":
    main()
