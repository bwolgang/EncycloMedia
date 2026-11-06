"""
Web front end (Flask). Start it with:   python -m encyclomedia serve
then open http://127.0.0.1:5000 in a browser.
"""
from __future__ import annotations

import io
import re
from urllib.parse import quote_plus

from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template, request,
                   send_file, url_for)

from . import covers, metadata
from .catalog import COPY_FIELDS, COPY_LABELS, CORE_FIELDS, CREATOR_LABEL, CREDIT_FIELDS, CREDIT_LABELS, split_names, YEAR_LABEL, DB_DEFAULT, EXTRA_FIELDS, MEDIA_TYPES, Catalog, genre_key, split_genres

FIELD_LABELS = {"isbn": "ISBN", "upc": "UPC", "edition_year": "Edition year (your copy)", "cover_url": "Cover image URL", "tracklist": "Track list",
                "discs": "Discs", "players": "Players"}


def label(field: str) -> str:
    return (FIELD_LABELS.get(field) or CREDIT_LABELS.get(field) or COPY_LABELS.get(field)
            or field.replace("_", " ").capitalize())


# How each type credits its creator on the details page ("Directed by Denis Villeneuve").
CREDIT_PHRASE = {"book": "by", "music": "by", "movie": "Directed by", "game": "Developed by"}

# Fields that get their own section on the details page rather than a table row.
LIST_FIELDS = {"tracklist": ";", "cast": ","}
HIDDEN_FIELDS = {"cover_url"}


def detail_rows(item: dict) -> list[tuple[str, str]]:
    """(label, value) rows for the details table, in a sensible order, skipping blanks."""
    rows = []
    if item.get("year"):
        rows.append((YEAR_LABEL[item["media_type"]], str(item["year"])))
    credits = CREDIT_FIELDS.get(item["media_type"], ())
    for field in EXTRA_FIELDS.get(item["media_type"], ()):
        value = item["extra"].get(field)
        if value not in (None, "") and field not in LIST_FIELDS and field not in HIDDEN_FIELDS \
                and field not in credits:
            rows.append((label(field), str(value)))
    if item.get("format"):
        rows.append(("Format", item["format"]))
    return rows


def credit_rows(item: dict) -> list[tuple[str, list[str], bool]]:
    """(label, names, searchable) for the details page's Credits section, main credit first.
    People are links that search the catalog; a record label is shown as plain text."""
    t = item["media_type"]
    rows = []
    if item.get("creator"):
        rows.append((CREATOR_LABEL[t], split_names(item["creator"]), True))
    for field in CREDIT_FIELDS.get(t, ()):
        names = split_names(item["extra"].get(field))
        if names:
            rows.append((CREDIT_LABELS[field], names, field != "label"))
    return rows


def copy_rows(item: dict) -> list[tuple[str, str, str]]:
    """(field, label, value) for the filled-in "Your copy" fields."""
    return [(f, COPY_LABELS[f], str(item["extra"][f])) for f in COPY_FIELDS
            if item["extra"].get(f) not in (None, "")]


# Each media type has its own page: /books, /music, /movies, /games.
TYPE_SLUG = {"book": "books", "music": "music", "movie": "movies", "game": "games"}
SLUG_TYPE = {slug: t for t, slug in TYPE_SLUG.items()}
TYPE_TITLE = {"book": "Books", "music": "Music", "movie": "Movies", "game": "Games"}
ITEM_NOUN = {"book": "book", "music": "album", "movie": "movie", "game": "game"}


def browse_url(params: dict, drop=None) -> str:
    """URL of a browse page with the given filters, leaving out blank ones and `drop`.
    A `type` filter goes to that type's own page (/books?genre=...), otherwise to All."""
    params = {k: v for k, v in params.items() if v and k != drop}
    media_type = params.pop("type", None)
    if media_type in TYPE_SLUG:
        return url_for("media_page", kind=TYPE_SLUG[media_type], **params)
    return url_for("index", **params)


def sort_options(media_type) -> list[tuple[str, str]]:
    """(value, label) choices for the sort menu on a browse page."""
    who = CREATOR_LABEL[media_type] if media_type else "Creator"
    opts = [("title", "Title"), ("creator", who), ("newest", "Year (newest)"),
            ("oldest", "Year (oldest)"), ("added", "Recently added")]
    return opts if media_type else [("type", "Media type"), *opts]


def split_list(value, sep) -> list[str]:
    return [part.strip() for part in re.split(rf"\s*{re.escape(sep)}\s*|\n", str(value or "")) if part.strip()]


def external_links(item: dict) -> list[tuple[str, str]]:
    """Where to read more about this item online."""
    title = item["title"]
    who = item.get("creator") or ""
    t = item["media_type"]
    if t == "book":
        isbn = re.sub(r"[^0-9Xx]", "", str(item["extra"].get("isbn") or ""))
        return [("Open Library", f"https://openlibrary.org/isbn/{isbn}" if isbn
                 else f"https://openlibrary.org/search?q={quote_plus(f'{title} {who}'.strip())}")]
    if t == "music":
        return [("MusicBrainz", f"https://musicbrainz.org/search?type=release&query={quote_plus(f'{title} {who}'.strip())}")]
    if t == "movie":
        query = f"{title} y:{item['year']}" if item.get("year") else title
        return [("TMDB", f"https://www.themoviedb.org/search/movie?query={quote_plus(query)}")]
    if t == "game":
        return [("IGDB", f"https://www.igdb.com/search?type=1&q={quote_plus(title)}")]
    return []


def create_app(db_path=DB_DEFAULT, embedder=None) -> Flask:
    app = Flask(__name__)
    app.config.update(DB_PATH=str(db_path), SECRET_KEY="encyclomedia-local",
                      MAX_CONTENT_LENGTH=covers.MAX_UPLOAD_BYTES + 1024 * 1024)
    app.jinja_env.globals.update(CREDIT_PHRASE=CREDIT_PHRASE, LIST_FIELDS=LIST_FIELDS, split_list=split_list, browse_url=browse_url,
                                 TYPE_SLUG=TYPE_SLUG, TYPE_TITLE=TYPE_TITLE, ITEM_NOUN=ITEM_NOUN,
                                 MEDIA_TYPES=MEDIA_TYPES, EXTRA_FIELDS=EXTRA_FIELDS, COPY_FIELDS=COPY_FIELDS,
                                 CREDIT_FIELDS=CREDIT_FIELDS,
                                 CREATOR_LABEL=CREATOR_LABEL, YEAR_LABEL=YEAR_LABEL, label=label,
                                 semantic_on=embedder is not None)

    # Clear out any cover files left behind last time (e.g. ones Windows had locked).
    startup = Catalog(app.config["DB_PATH"])
    startup.prune_covers()
    startup.conn.close()

    def catalog() -> Catalog:
        # One SQLite connection per request; the (expensive) embedder is shared.
        if "catalog" not in g:
            g.catalog = Catalog(app.config["DB_PATH"], embedder=embedder)
        return g.catalog

    @app.context_processor
    def nav_counts():
        """Item counts for the media tabs at the top of every page."""
        try:
            return {"nav_counts": catalog().counts()}
        except Exception:  # never let the tab bar break a page
            return {"nav_counts": {}}

    @app.teardown_appcontext
    def _close(_exc):
        cat = g.pop("catalog", None)
        if cat is not None:
            cat.conn.close()

    def genre_context():
        """Genre lists for the item form: all of them (to reuse existing spellings)
        and per media type (for the autocomplete suggestions)."""
        cat = catalog()
        return {"all_genres": cat.genres(),
                "genres_by_type": {t: [x["name"] for x in cat.genres(t)] for t in MEDIA_TYPES},
                # Autocomplete for places you've already used, so "Shelf A2" stays one spelling.
                "locations": [v["name"] for v in cat.copy_values("location")],
                "origins": [v["name"] for v in cat.copy_values("origin")]}

    def apply_cover_upload(item_id) -> bool:
        """Save an uploaded cover from the request, if any. Returns False (and flashes why) on a bad file."""
        upload = request.files.get("cover_file")
        if not upload or not upload.filename:
            return True
        try:
            catalog().set_cover_image(item_id, upload.read())
            return True
        except covers.CoverError as e:
            flash(f"The cover wasn't saved: {e}", "error")
            return False

    @app.errorhandler(413)
    def too_large(_e):
        flash(f"That file is too big. Images must be under {covers.MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", "error")
        return redirect(request.referrer or url_for("index"))

    @app.get("/covers/<filename>")
    def cover_image(filename):
        path = catalog().cover_file(filename)
        if path is None:
            abort(404)
        return send_file(path, max_age=60 * 60 * 24 * 365)  # names are unique, so cache forever

    @app.post("/item/<int:item_id>/cover")
    def upload_cover(item_id):
        catalog().get(item_id) or abort(404)
        if not (request.files.get("cover_file") and request.files["cover_file"].filename):
            flash("Choose an image first.", "error")
        elif apply_cover_upload(item_id):
            flash("Cover updated.", "ok")
        return redirect(url_for("item", item_id=item_id))

    @app.post("/item/<int:item_id>/cover/remove")
    def remove_cover(item_id):
        if catalog().clear_cover(item_id):
            flash("Cover removed.", "ok")
        return redirect(url_for("item", item_id=item_id))

    def form_to_item():
        f = request.form
        media_type = f.get("media_type", "")
        core = {k: f.get(k, "") for k in CORE_FIELDS}
        extra = {k: f.get(f"extra__{k}", "") for k in EXTRA_FIELDS.get(media_type, ())}
        extra["cover_url"] = f.get("cover_url", "")
        for k in COPY_FIELDS:
            extra[k] = f.get(f"extra__{k}", "")
        return media_type, core, extra

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        # Older links like /?type=book&genre=Fantasy now live on the type's own page.
        if request.args.get("type") in TYPE_SLUG:
            return redirect(browse_url(request.args.to_dict()))
        return browse(None)

    @app.get("/<any(books, music, movies, games):kind>")
    def media_page(kind):
        return browse(SLUG_TYPE[kind])

    def browse(media_type):
        """The All page (media_type None) or one type's page: list, search and filters."""
        q = request.args.get("q", "").strip()
        genre = request.args.get("genre") or None
        location = request.args.get("location", "").strip() or None
        origin = request.args.get("origin", "").strip() or None
        semantic = request.args.get("mode", "hybrid") != "keyword"
        sorts = sort_options(media_type)
        sort = request.args.get("sort") if request.args.get("sort") in dict(sorts) else sorts[0][0]
        cat = catalog()
        suggestions = []
        if q:
            items = cat.search(q, media_type, limit=50, semantic=semantic, genre=genre,
                               location=location, origin=origin)
            if not items and not (genre or location or origin):
                suggestions = cat.suggest(q)
        else:
            items = cat.list(media_type, genre, location=location, origin=origin, sort=sort)
        genre_options = cat.genres(media_type)
        if genre and genre_key(genre) not in {genre_key(g["name"]) for g in genre_options}:
            genre_options.append({"name": genre, "count": 0})  # keep the active filter selectable
        return render_template("index.html", items=items, q=q, media_type=media_type, genre=genre,
                               location=location, origin=origin, sort=sort, sorts=sorts,
                               genre_options=genre_options, mode="hybrid" if semantic else "keyword",
                               counts=cat.counts(), suggestions=suggestions)

    @app.get("/genres")
    def genres_page():
        cat = catalog()
        per_type = {t: {g["name"]: g["count"] for g in cat.genres(t)} for t in MEDIA_TYPES}
        return render_template("genres.html", genres=cat.genres(), per_type=per_type)

    @app.post("/genres/rename")
    def genres_rename():
        old, new = request.form.get("old", ""), request.form.get("new", "")
        try:
            n = catalog().rename_genre(old, new)
            flash(f"“{old}” is now “{new.strip()}” on {n} item{'s' if n != 1 else ''}.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        return redirect(url_for("genres_page"))

    @app.post("/genres/remove")
    def genres_remove():
        name = request.form.get("name", "")
        n = catalog().remove_genre(name)
        flash(f"Removed “{name}” from {n} item{'s' if n != 1 else ''}.", "ok")
        return redirect(url_for("genres_page"))

    @app.route("/add", methods=["GET", "POST"])
    def add():
        if request.method == "POST":
            media_type, core, extra = form_to_item()
            fields = {k: v for k, v in core.items() if k != "title"}
            try:
                item_id = catalog().add(media_type, core["title"], extra=extra, **fields)
            except ValueError as e:
                flash(str(e), "error")
                item = {"media_type": media_type, **core, "extra": extra, "genres": split_genres(core["genre"])}
                return render_template("form.html", item=item, **genre_context()), 400
            if not apply_cover_upload(item_id):
                return redirect(url_for("edit", item_id=item_id))  # item saved; let them retry the image
            flash(f"Added “{core['title'].strip()}”.", "ok")
            return redirect(url_for("index"))
        source = catalog().get(request.args.get("copy", type=int) or 0)
        if source:
            # "Add another copy": same details, ready to adjust for a different edition/format.
            item = {k: v for k, v in source.items() if k not in ("id", "added_at", "updated_at")}
            # Edition, location, origin and condition describe one physical copy, so start them blank.
            item["extra"] = {k: v for k, v in source["extra"].items() if k not in COPY_FIELDS}
            item["copy_of"] = source["title"]
        else:
            item = {"media_type": request.args.get("type", "book"), "extra": {}, "genres": []}
        return render_template("form.html", item=item, **genre_context())

    @app.get("/item/<int:item_id>")
    def item(item_id):
        """Read-only details page."""
        cat = catalog()
        current = cat.get(item_id) or abort(404)
        return render_template("detail.html", item=current, details=detail_rows(current),
                               copy_details=copy_rows(current), credits=credit_rows(current),
                               links=external_links(current), related=cat.similar(item_id))

    @app.route("/item/<int:item_id>/edit", methods=["GET", "POST"])
    def edit(item_id):
        cat = catalog()
        current = cat.get(item_id) or abort(404)
        if request.method == "POST":
            media_type, core, extra = form_to_item()
            try:
                cat.update(item_id, media_type=media_type, extra=extra, **core)
            except ValueError as e:
                flash(str(e), "error")
                return redirect(url_for("edit", item_id=item_id))
            if not apply_cover_upload(item_id):
                return redirect(url_for("edit", item_id=item_id))
            flash(f"Saved “{core['title'].strip()}”.", "ok")
            return redirect(url_for("index"))
        return render_template("form.html", item=current, **genre_context())

    @app.post("/item/<int:item_id>/delete")
    def delete(item_id):
        cat = catalog()
        current = cat.get(item_id)
        if current and cat.remove(item_id):
            flash(f"Deleted “{current['title']}”.", "ok")
        return redirect(url_for("index"))

    @app.post("/import")
    def import_csv():
        upload = request.files.get("file")
        if not upload or not upload.filename:
            flash("Choose a CSV file first.", "error")
        else:
            text = io.TextIOWrapper(upload.stream, encoding="utf-8-sig", newline="")
            n = catalog().import_csv(text)
            flash(f"Imported {n} items.", "ok")
        return redirect(url_for("index"))

    @app.get("/export.csv")
    def export_csv():
        media_type = request.args.get("type") if request.args.get("type") in TYPE_SLUG else None
        name = f"encyclomedia-{TYPE_SLUG[media_type]}.csv" if media_type else "encyclomedia.csv"
        return Response(catalog().export_csv(media_type), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}"})

    # ---------------------------------------------------------------- JSON API
    @app.get("/api/lookup")
    def api_lookup():
        try:
            results = metadata.lookup(request.args.get("type", ""), request.args.get("q", ""))
        except metadata.MetadataError as e:
            return jsonify(error=str(e)), 502
        return jsonify(results=results)

    @app.get("/api/search")
    def api_search():
        q = request.args.get("q", "")
        return jsonify(results=catalog().search(q, request.args.get("type") or None,
                                                limit=int(request.args.get("limit", 20)),
                                                semantic=request.args.get("mode") != "keyword",
                                                genre=request.args.get("genre") or None))

    @app.get("/api/genres")
    def api_genres():
        return jsonify(genres=catalog().genres(request.args.get("type") or None))

    return app
