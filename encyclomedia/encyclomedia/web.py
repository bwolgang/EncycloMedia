"""
Web front end (Flask). Start it with:   python -m encyclomedia serve
then open http://127.0.0.1:5000 in a browser.
"""
from __future__ import annotations

import io

from flask import Flask, Response, abort, flash, g, jsonify, redirect, render_template, request, url_for

from . import metadata
from .catalog import CORE_FIELDS, CREATOR_LABEL, YEAR_LABEL, DB_DEFAULT, EXTRA_FIELDS, MEDIA_TYPES, Catalog, genre_key, split_genres

FIELD_LABELS = {"isbn": "ISBN", "upc": "UPC", "edition_year": "Edition year (your copy)", "cover_url": "Cover image URL", "tracklist": "Track list",
                "discs": "Discs", "players": "Players"}


def label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").capitalize())


def create_app(db_path=DB_DEFAULT, embedder=None) -> Flask:
    app = Flask(__name__)
    app.config.update(DB_PATH=str(db_path), SECRET_KEY="encyclomedia-local")
    app.jinja_env.globals.update(MEDIA_TYPES=MEDIA_TYPES, EXTRA_FIELDS=EXTRA_FIELDS,
                                 CREATOR_LABEL=CREATOR_LABEL, YEAR_LABEL=YEAR_LABEL, label=label,
                                 semantic_on=embedder is not None)

    def catalog() -> Catalog:
        # One SQLite connection per request; the (expensive) embedder is shared.
        if "catalog" not in g:
            g.catalog = Catalog(app.config["DB_PATH"], embedder=embedder)
        return g.catalog

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
                "genres_by_type": {t: [x["name"] for x in cat.genres(t)] for t in MEDIA_TYPES}}

    def form_to_item():
        f = request.form
        media_type = f.get("media_type", "")
        core = {k: f.get(k, "") for k in CORE_FIELDS}
        extra = {k: f.get(f"extra__{k}", "") for k in EXTRA_FIELDS.get(media_type, ())}
        return media_type, core, extra

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        q = request.args.get("q", "").strip()
        media_type = request.args.get("type") or None
        genre = request.args.get("genre") or None
        semantic = request.args.get("mode", "hybrid") != "keyword"
        cat = catalog()
        suggestions = []
        if q:
            items = cat.search(q, media_type, limit=50, semantic=semantic, genre=genre)
            if not items and not genre:
                suggestions = cat.suggest(q)
        else:
            items = cat.list(media_type, genre)
        genre_options = cat.genres(media_type)
        if genre and genre_key(genre) not in {genre_key(g["name"]) for g in genre_options}:
            genre_options.append({"name": genre, "count": 0})  # keep the active filter selectable
        return render_template("index.html", items=items, q=q, media_type=media_type, genre=genre,
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
            flash(f"Added “{core['title'].strip()}”.", "ok")
            return redirect(url_for("index"))
        media_type = request.args.get("type", "book")
        return render_template("form.html", item={"media_type": media_type, "extra": {}, "genres": []},
                               **genre_context())

    @app.route("/item/<int:item_id>", methods=["GET", "POST"])
    def item(item_id):
        cat = catalog()
        current = cat.get(item_id) or abort(404)
        if request.method == "POST":
            media_type, core, extra = form_to_item()
            try:
                cat.update(item_id, media_type=media_type, extra=extra, **core)
            except ValueError as e:
                flash(str(e), "error")
                return redirect(url_for("item", item_id=item_id))
            flash(f"Saved “{core['title'].strip()}”.", "ok")
            return redirect(url_for("index"))
        return render_template("form.html", item=current, **genre_context())

    @app.post("/item/<int:item_id>/delete")
    def delete(item_id):
        if catalog().remove(item_id):
            flash("Deleted.", "ok")
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
        return Response(catalog().export_csv(), mimetype="text/csv",
                        headers={"Content-Disposition": "attachment; filename=encyclomedia.csv"})

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
