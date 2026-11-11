"""
Web front end (Flask). Start it with:   python -m encyclomedia serve
then open http://127.0.0.1:5000 in a browser.
"""
from __future__ import annotations

import io
import re
from datetime import date
from urllib.parse import quote_plus

from flask import (Flask, Response, abort, flash, g, jsonify, redirect, render_template, request,
                   send_file, url_for)

from . import autofill, covers, metadata
from .catalog import COPY_FIELDS, COPY_LABELS, CORE_FIELDS, CREATOR_LABEL, CREDIT_FIELDS, CREDIT_LABELS, split_names, name_key, surname_first, YEAR_LABEL, DB_DEFAULT, EXTRA_FIELDS, MEDIA_TYPES, Catalog, genre_key, split_genres

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


def browse_url(params: dict, drop=None, **overrides) -> str:
    """URL of a browse page with the given filters, leaving out blank ones and `drop`.
    A `type` filter goes to that type's own page (/books?genre=...), otherwise to All.
    A list value (several genres) becomes a repeated parameter: ?genre=Rock&genre=Jazz."""
    clean = {}
    for k, v in {**params, **overrides}.items():
        if k == drop:
            continue
        if isinstance(v, (list, tuple)):
            v = [x for x in v if x]
        if v:
            clean[k] = v
    params = clean
    media_type = params.pop("type", None)
    if isinstance(media_type, list):
        media_type = media_type[0]
    if params.get("match") and len(params.get("genre") or []) < 2:
        params.pop("match")  # "match all" only means something with two or more genres
    if media_type in TYPE_SLUG:
        return url_for("media_page", kind=TYPE_SLUG[media_type], **params)
    return url_for("index", **params)


def genre_toggle_url(params: dict, genre: str) -> str:
    """The same browse page with `genre` added to the selected genres, or removed if
    it's already selected. Used by the genre chips."""
    selected = list(params.get("genre") or [])
    if genre_key(genre) in {genre_key(g) for g in selected}:
        selected = [g for g in selected if genre_key(g) != genre_key(genre)]
    else:
        selected.append(genre)
    return browse_url({**params, "genre": selected})


def sort_options(media_type) -> list[tuple[str, str]]:
    """(value, label) choices for the sort menu on a browse page."""
    who = CREATOR_LABEL[media_type] if media_type else "Creator"
    opts = [("title", "Title"), ("creator", who), ("newest", "Year (newest)"),
            ("oldest", "Year (oldest)"), ("added", "Recently added")]
    return opts if media_type else [("type", "Media type"), *opts]


def item_summary(item: dict) -> str:
    """'The Beatles · 1969 · vinyl · Record shelf': enough to tell copies apart."""
    extra = item.get("extra") or {}
    bits = [item.get("creator"), item.get("year"), item.get("format"), extra.get("platform"),
            extra.get("edition"), extra.get("location")]
    return " · ".join(str(b) for b in bits if b)


def short_list(names, limit=4) -> str:
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def nice_date(iso) -> str:
    """'2026-11-10' -> 'Nov 10, 2026'."""
    if not iso:
        return ""
    d = date.fromisoformat(str(iso)[:10])
    return f"{d:%b} {d.day}, {d.year}"


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
    app.jinja_env.globals.update(CREDIT_PHRASE=CREDIT_PHRASE, LIST_FIELDS=LIST_FIELDS, split_list=split_list, browse_url=browse_url, nice_date=nice_date, item_summary=item_summary, genre_toggle_url=genre_toggle_url,
                                 genre_key=genre_key,
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

    def save_file_under(cat, media_type, creator, previous_creator=None):
        """Store the form's File-under name for this creator (shared by all their items).
        A blank box clears the setting only when editing the same creator, so adding
        another album by an artist never undoes how they're already filed."""
        if "file_under" not in request.form or not (creator or "").strip():
            return
        value = request.form.get("file_under", "").strip()
        same_creator = previous_creator is not None and name_key(previous_creator) == name_key(creator)
        if value or same_creator:
            cat.set_file_under(creator, value, media_type)

    def file_under_context(cat, item):
        creator, media_type = item.get("creator") or "", item.get("media_type") or "book"
        return {"file_under": cat.file_under_setting(creator) or "" if creator else "",
                "file_under_auto": cat.automatic_file_under(creator, media_type) if creator else ""}

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
            return redirect(browse_url(request.args.to_dict(flat=False)))
        return browse(None)

    @app.get("/<any(books, music, movies, games):kind>")
    def media_page(kind):
        return browse(SLUG_TYPE[kind])

    def browse(media_type):
        """The All page (media_type None) or one type's page: list, search and filters."""
        q = request.args.get("q", "").strip()
        # Several genres can be selected (?genre=Rock&genre=Psychedelic+rock). By default an
        # item needs any one of them; ?match=all asks for items that have every one.
        genres = list({genre_key(g): g.strip() for g in request.args.getlist("genre") if genre_key(g)}.values())
        genre_match = "all" if request.args.get("match") == "all" and len(genres) > 1 else "any"
        location = request.args.get("location", "").strip() or None
        origin = request.args.get("origin", "").strip() or None
        semantic = request.args.get("mode", "hybrid") != "keyword"
        sorts = sort_options(media_type)
        sort = request.args.get("sort") if request.args.get("sort") in dict(sorts) else sorts[0][0]
        cat = catalog()
        suggestions = []
        if q:
            items = cat.search(q, media_type, limit=50, semantic=semantic, genre=genres,
                               location=location, origin=origin, genre_match=genre_match)
            if not items and not (genres or location or origin):
                suggestions = cat.suggest(q)
        else:
            items = cat.list(media_type, genres, location=location, origin=origin, sort=sort,
                             genre_match=genre_match)
        loans = cat.current_loans_by_item()
        genre_options = cat.genres(media_type)
        needing = len(autofill.books_needing_details(cat)) if media_type == "book" else 0
        return render_template("index.html", items=items, q=q, media_type=media_type, loans=loans,
                               genres=genres, genre_match=genre_match,
                               selected_keys={genre_key(g) for g in genres},
                               books_needing_details=needing,
                               location=location, origin=origin, sort=sort, sorts=sorts,
                               genre_options=genre_options, mode="hybrid" if semantic else "keyword",
                               counts=cat.counts(), suggestions=suggestions)

    @app.route("/artists", methods=["GET", "POST"])
    def artists_page():
        """Every music artist in one list, to choose how each is filed: as written
        (bands), by surname (solo artists), or a custom File-under name."""
        cat = catalog()
        if request.method == "POST":
            changed = 0
            for i in range(request.form.get("count", 0, type=int)):
                name = request.form.get(f"name_{i}", "")
                kind = request.form.get(f"kind_{i}", "band")
                value = {"band": "", "solo": surname_first(name),
                         "custom": request.form.get(f"custom_{i}", "").strip()}.get(kind, "")
                before = cat.file_under(name, "music")
                cat.set_file_under(name, value, "music")
                changed += cat.file_under(name, "music") != before
            flash(f"Saved. {changed} artist{'s' if changed != 1 else ''} re-filed." if changed
                  else "Saved. Nothing changed.", "ok")
            return redirect(url_for("artists_page"))
        artists = cat.creators("music")
        for n, a in enumerate(artists):
            a["i"] = n  # form field number for this artist
        return render_template("artists.html", artists=artists,
                               solo_count=sum(a["kind"] == "solo" and not a["one_word"] for a in artists))

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
            # Already in the catalog? Show what's there and let them choose; "Add anyway"
            # (or coming from "Add another copy") sends allow_duplicate and skips this.
            if not request.form.get("allow_duplicate") and media_type in MEDIA_TYPES and core["title"].strip():
                dups = catalog().find_duplicates(media_type, core["title"], core["creator"], extra,
                                                 format=core["format"])
                if dups:
                    item = {"media_type": media_type, **core, "extra": extra, "genres": split_genres(core["genre"])}
                    upload = request.files.get("cover_file")
                    ctx = {**genre_context(), **file_under_context(catalog(), item),
                           "file_under": request.form.get("file_under", "")}
                    return render_template("form.html", item=item, duplicates=dups,
                                           cover_lost=bool(upload and upload.filename), **ctx)
            try:
                item_id = catalog().add(media_type, core["title"], extra=extra, **fields)
            except ValueError as e:
                flash(str(e), "error")
                item = {"media_type": media_type, **core, "extra": extra, "genres": split_genres(core["genre"])}
                return render_template("form.html", item=item, **genre_context()), 400
            save_file_under(catalog(), media_type, core.get("creator"))
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
        return render_template("form.html", item=item, **genre_context(), **file_under_context(catalog(), item))

    @app.get("/item/<int:item_id>")
    def item(item_id):
        """Read-only details page."""
        cat = catalog()
        current = cat.get(item_id) or abort(404)
        return render_template("detail.html", item=current, details=detail_rows(current),
                               copy_details=copy_rows(current), credits=credit_rows(current),
                               links=external_links(current), related=cat.similar(item_id),
                               loan=cat.current_loan(item_id), loan_history=cat.loan_history(item_id),
                               borrowers=cat.borrowers(), today=date.today().isoformat())

    # ---------------------------------------------------------------- circulation
    @app.post("/item/<int:item_id>/lend")
    def lend(item_id):
        cat = catalog()
        item = cat.get(item_id) or abort(404)
        f = request.form
        try:
            cat.lend(item_id, f.get("borrower"), f.get("loaned_on"), f.get("due_on"), f.get("notes"))
            flash(f"Lent “{item['title']}” to {' '.join(f.get('borrower', '').split())}.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        return redirect(url_for("item", item_id=item_id) + "#circulation")

    @app.post("/item/<int:item_id>/return")
    def return_item(item_id):
        cat = catalog()
        item = cat.get(item_id) or abort(404)
        try:
            loan = cat.return_item(item_id, request.form.get("returned_on"))
            flash(f"“{item['title']}” is back from {loan['borrower']} after {loan['days']} "
                  f"day{'s' if loan['days'] != 1 else ''}.", "ok")
        except ValueError as e:
            flash(str(e), "error")
        back = request.form.get("next")
        return redirect(back if back == url_for("loans_page") else url_for("item", item_id=item_id) + "#circulation")

    @app.post("/loan/<int:loan_id>/delete")
    def delete_loan(loan_id):
        item_id = catalog().delete_loan(loan_id)
        if item_id is None:
            abort(404)
        flash("Loan record deleted.", "ok")
        return redirect(url_for("item", item_id=item_id) + "#circulation")

    @app.get("/loans")
    def loans_page():
        cat = catalog()
        return render_template("loans.html", out=cat.loans_out(), returned=cat.recent_returns(15),
                               today=date.today().isoformat())

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
            save_file_under(cat, media_type, core.get("creator"), previous_creator=current.get("creator") or "")
            if not apply_cover_upload(item_id):
                return redirect(url_for("edit", item_id=item_id))
            flash(f"Saved “{core['title'].strip()}”.", "ok")
            return redirect(url_for("index"))
        return render_template("form.html", item=current, **genre_context(), **file_under_context(cat, current))

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
            cat = catalog()
            text = io.TextIOWrapper(upload.stream, encoding="utf-8-sig", newline="")
            ids, skipped = [], []
            n = cat.import_csv(text, ids_out=ids, skip_duplicates=bool(request.form.get("skip_duplicates")),
                               skipped_out=skipped)
            flash(f"Imported {n} item{'s' if n != 1 else ''}.", "ok")
            if skipped:
                flash(f"Skipped {len(skipped)} already in your catalog: {short_list(skipped)}. "
                      "Untick “Skip items I already have” to import them anyway.", "ok")
            if request.form.get("autofill"):
                report = autofill.autofill_books(cat, ids)
                if report.summary():
                    flash(report.summary(), "error" if report.error else "ok")
        return redirect(url_for("index"))

    @app.post("/books/autofill")
    def books_autofill():
        report = autofill.autofill_books(catalog())
        flash(report.summary() or "Every book with an ISBN already has its details.",
              "error" if report.error else "ok")
        return redirect(url_for("media_page", kind="books"))

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
                                                genre=request.args.getlist("genre") or None,
                                                genre_match=request.args.get("match", "any")))

    @app.get("/api/duplicates")
    def api_duplicates():
        """Items already in the catalog that the add form's current details would duplicate."""
        a = request.args
        media_type = a.get("type") if a.get("type") in MEDIA_TYPES else None
        if not media_type or not a.get("title", "").strip():
            return jsonify(matches=[])
        extra = {k: a.get(k, "") for k in ("isbn", "barcode", "upc", "edition", "platform")}
        dups = catalog().find_duplicates(media_type, a["title"], a.get("creator"), extra, format=a.get("format"))
        return jsonify(matches=[{"id": d["id"], "title": d["title"], "summary": item_summary(d),
                                 "url": url_for("item", item_id=d["id"])} for d in dups])

    @app.get("/api/file-under")
    def api_file_under():
        """How a creator is filed: their setting (if any), the automatic filing, and the
        surname-first form used by the form's "Solo artist" button."""
        cat, name = catalog(), request.args.get("name", "")
        media_type = request.args.get("type") if request.args.get("type") in MEDIA_TYPES else "book"
        return jsonify(setting=cat.file_under_setting(name) if name.strip() else None,
                       automatic=cat.automatic_file_under(name, media_type) if name.strip() else "",
                       surname=surname_first(name))

    @app.get("/api/genres")
    def api_genres():
        return jsonify(genres=catalog().genres(request.args.get("type") or None))

    return app
