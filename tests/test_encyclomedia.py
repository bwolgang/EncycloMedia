"""Run with:  python -m pytest -q"""
import html
import io
import json
import re
import sqlite3

import numpy as np
import pytest

from encyclomedia import metadata
from encyclomedia.catalog import Catalog
from encyclomedia.web import create_app


# --------------------------------------------------------------------------- fixtures
class FakeEmbedder:
    """Stand-in for a sentence-transformer: words that share a 'concept' get similar
    vectors, so we can test meaning-based matching without downloading a model."""
    model_name = "fake-concepts"
    CONCEPTS = {
        "space": 0, "interstellar": 0, "galaxy": 0, "planet": 0, "planets": 0, "spaceship": 0,
        "myth": 1, "mythology": 1, "gods": 1, "greek": 1, "olympus": 1,
        "scary": 2, "horror": 2, "frightening": 2, "haunted": 2,
    }

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(64, dtype=np.float32)
            for w in re.findall(r"\w+", t.lower()):
                if w in self.CONCEPTS:
                    v[self.CONCEPTS[w]] += 3.0
            v[63] = 0.5  # small shared component so no vector is all zeros
            out.append(v / np.linalg.norm(v))
        return np.vstack(out)


@pytest.fixture
def cat(tmp_path):
    c = Catalog(tmp_path / "test.db")
    c.add("book", "Dune", creator="Frank Herbert", year=1965, genre="sci-fi", tags="desert, politics",
          extra={"isbn": "9780441013593", "publisher": "Ace", "pages": 604})
    c.add("movie", "Dune: Part Two", creator="Denis Villeneuve", year=2024, genre="sci-fi", format="Blu-ray",
          extra={"runtime": "166 min", "cast": "Timothée Chalamet, Zendaya"})
    c.add("music", "Abbey Road", creator="The Beatles", year=1969, format="vinyl", genre="rock",
          extra={"label": "Apple", "tracklist": "Come Together; Something; Here Comes the Sun"})
    c.add("game", "Hades", creator="Supergiant Games", genre="roguelike", tags="greek mythology",
          extra={"platform": "Switch"})
    c.add("book", "Mythos", creator="Stephen Fry", genre="mythology", tags="greek")
    return c


@pytest.fixture
def semantic_cat(tmp_path):
    c = Catalog(tmp_path / "sem.db", embedder=FakeEmbedder())
    c.add("book", "Foundation", creator="Isaac Asimov", notes="A galaxy-spanning empire collapses")
    c.add("movie", "The Shining", creator="Stanley Kubrick", notes="a haunted hotel in winter")
    c.add("book", "Cooking Basics", creator="Some Chef", notes="recipes")
    return c


# --------------------------------------------------------------------------- catalog
def test_ranking_prefers_items_matching_more_terms(cat):
    results = cat.search("dune herbert")
    assert [r["title"] for r in results[:2]] == ["Dune", "Dune: Part Two"]
    assert results[0]["score"] == 1.0


def test_type_specific_fields_are_stored_and_searchable(cat):
    dune = cat.search("9780441013593")[0]
    assert dune["title"] == "Dune" and dune["extra"]["publisher"] == "Ace"
    assert cat.search("zendaya")[0]["title"] == "Dune: Part Two"
    assert cat.search("here comes the sun", media_type="music")[0]["title"] == "Abbey Road"


def test_filter_prefix_and_stemming(cat):
    assert {r["title"] for r in cat.search("myth")} == {"Mythos", "Hades"}
    assert [r["title"] for r in cat.search("sci", media_type="movie")] == ["Dune: Part Two"]


def test_update_and_remove_keep_index_in_sync(cat):
    hades = cat.search("hades")[0]
    cat.update(hades["id"], tags="underworld", extra={"platform": "PS5"})
    assert cat.search("underworld")[0]["id"] == hades["id"]
    assert not cat.search("switch")
    assert cat.get(hades["id"])["extra"] == {"platform": "PS5"}
    assert cat.remove(hades["id"]) and not cat.search("underworld")


def test_suggest_handles_typos(cat):
    assert not cat.search("beetles")
    assert cat.suggest("beetles") == ["The Beatles"]


def test_rejects_bad_type_and_missing_title(cat):
    with pytest.raises(ValueError):
        cat.add("comic", "Watchmen")
    with pytest.raises(ValueError):
        cat.add("book", "  ")


def test_csv_round_trip_keeps_type_specific_fields(cat, tmp_path):
    text = cat.export_csv()
    copy = Catalog(tmp_path / "copy.db")
    assert copy.import_csv(io.StringIO(text)) == 5
    dune = copy.search("dune")[0]
    assert dune["extra"]["isbn"] == "9780441013593" and dune["extra"]["pages"] == "604"


def test_migrates_database_from_original_script(tmp_path):
    path = tmp_path / "v1.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE items (id INTEGER PRIMARY KEY, media_type TEXT NOT NULL, title TEXT NOT NULL,
            creator TEXT, year INTEGER, genre TEXT, format TEXT, tags TEXT, notes TEXT,
            added_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE VIRTUAL TABLE items_fts USING fts5(title, creator, genre, tags, notes,
            content='items', content_rowid='id');
        CREATE TRIGGER items_ai AFTER INSERT ON items BEGIN
            INSERT INTO items_fts(rowid, title, creator, genre, tags, notes)
            VALUES (new.id, new.title, new.creator, new.genre, new.tags, new.notes); END;
        INSERT INTO items (media_type, title, creator) VALUES ('book', 'Neuromancer', 'William Gibson');
    """)
    old.commit()
    old.close()

    c = Catalog(path)
    assert c.search("gibson")[0]["title"] == "Neuromancer"
    new_id = c.add("game", "Celeste", extra={"platform": "PC"})
    assert c.search("pc")[0]["id"] == new_id
    assert Catalog(path).search("neuromancer")  # reopening doesn't re-migrate or break


# --------------------------------------------------------------------------- semantic
def test_semantic_finds_items_without_shared_keywords(semantic_cat):
    results = semantic_cat.search("outer space planets")
    assert results[0]["title"] == "Foundation" and results[0]["match"] == "semantic"
    assert semantic_cat.search("frightening")[0]["title"] == "The Shining"


def test_keyword_only_mode_and_hybrid_boost(semantic_cat):
    assert not semantic_cat.search("outer space planets", semantic=False)
    both = semantic_cat.search("asimov galaxy")[0]
    assert both["title"] == "Foundation" and both["match"] == "both"


def test_unrelated_items_are_not_returned(semantic_cat):
    assert "Cooking Basics" not in [r["title"] for r in semantic_cat.search("haunted galaxy")]


def test_reindex_skips_unchanged_items(semantic_cat):
    assert semantic_cat.reindex() == 0
    assert semantic_cat.reindex(force=True) == 3
    item = semantic_cat.search("foundation")[0]
    semantic_cat.update(item["id"], notes="greek gods")
    assert semantic_cat.search("mythology")[0]["title"] == "Foundation"


# --------------------------------------------------------------------------- metadata parsers
def test_parse_openlibrary_isbn():
    d = {"title": "Dune", "authors": [{"name": "Frank Herbert"}], "publish_date": "August 1990",
         "publishers": [{"name": "Ace Books"}], "number_of_pages": 535,
         "subjects": [{"name": "Science fiction"}], "cover": {"medium": "https://covers/x.jpg"}}
    c = metadata.parse_openlibrary_book(d, "0441172717", first_published=1965)
    # year = when the book first came out; edition_year = the printing this ISBN is
    assert (c["creator"], c["year"], c["genre"]) == ("Frank Herbert", 1965, "Science fiction")
    assert c["extra"] == {"isbn": "0441172717", "edition_year": 1990, "publisher": "Ace Books",
                          "pages": 535, "cover_url": "https://covers/x.jpg"}
    # If the original year can't be found, fall back to the edition year.
    assert metadata.parse_openlibrary_book(d, "0441172717")["year"] == 1990


def test_isbn_lookup_fetches_original_year(monkeypatch):
    edition = {"title": "The Hobbit", "authors": [{"name": "J.R.R. Tolkien"}], "publish_date": "2012"}
    calls = []

    def fake_http(url, params=None, **kw):
        calls.append((url, params))
        if url.endswith("/api/books"):
            return {"ISBN:9780547928227": edition}
        return {"docs": [{"first_publish_year": 1937}]}

    monkeypatch.setattr(metadata, "_http", fake_http)
    c = metadata.lookup("book", "978-0-547-92822-7")[0]
    assert (c["year"], c["extra"]["edition_year"]) == (1937, 2012)
    search = [p for u, p in calls if u.endswith("/search.json")]
    assert search[0]["q"] == "isbn:9780547928227"

    def search_down(url, params=None, **kw):
        if url.endswith("/api/books"):
            return {"ISBN:9780547928227": edition}
        raise metadata.MetadataError("search unavailable")

    monkeypatch.setattr(metadata, "_http", search_down)
    assert metadata.lookup("book", "9780547928227")[0]["year"] == 2012  # still works


def test_parse_musicbrainz_release():
    r = {"id": "abc", "title": "Abbey Road", "date": "2019-09-27", "barcode": "077774644624",
         "release-group": {"first-release-date": "1969-09-26"},
         "artist-credit": [{"name": "The Beatles", "joinphrase": ""}],
         "label-info": [{"label": {"name": "Apple Records"}}],
         "media": [{"format": "12\" Vinyl", "tracks": [{"title": "Come Together"}, {"title": "Something"}]}]}
    c = metadata.parse_musicbrainz_release(r)
    assert (c["creator"], c["year"], c["format"]) == ("The Beatles", 1969, '12" Vinyl')
    assert c["extra"]["edition_year"] == 2019  # this pressing is the 50th-anniversary reissue
    assert c["extra"]["tracklist"] == "Come Together; Something" and c["extra"]["label"] == "Apple Records"
    # Search results without the release group leave the original year blank rather than guess.
    assert metadata.parse_musicbrainz_release({**r, "release-group": {}})["year"] is None


def test_parse_tmdb_movie():
    d = {"id": 1, "title": "Arrival", "release_date": "2016-11-11", "runtime": 116,
         "genres": [{"name": "Drama"}, {"name": "Science Fiction"}], "overview": "Linguist...",
         "poster_path": "/p.jpg",
         "credits": {"crew": [{"job": "Director", "name": "Denis Villeneuve"}], "cast": [{"name": "Amy Adams"}]},
         "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": [{"certification": "PG-13"}]}]}}
    c = metadata.parse_tmdb_movie(d)
    assert (c["creator"], c["year"], c["genre"]) == ("Denis Villeneuve", 2016, "Drama, Science Fiction")
    assert c["extra"]["rating"] == "PG-13" and c["extra"]["runtime"] == "116 min"


def test_parse_igdb_game():
    g = {"id": 9, "name": "Hades", "first_release_date": 1600387200,
         "genres": [{"name": "Roguelike"}], "platforms": [{"name": "Nintendo Switch"}, {"name": "PC"}],
         "involved_companies": [{"developer": True, "publisher": True, "company": {"name": "Supergiant Games"}}],
         "cover": {"image_id": "co1"}}
    c = metadata.parse_igdb_game(g)
    assert (c["creator"], c["year"]) == ("Supergiant Games", 2020)
    assert c["extra"]["platform"] == "Nintendo Switch, PC"


def test_lookup_routing(monkeypatch):
    calls = []
    monkeypatch.setattr(metadata, "books_by_isbn", lambda code: calls.append(("isbn", code)) or [])
    monkeypatch.setattr(metadata, "books_by_title", lambda q, limit: calls.append(("title", q)) or [])
    metadata.lookup("book", "978-0-441-01359-3")
    metadata.lookup("book", "Dune")
    assert calls == [("isbn", "978-0-441-01359-3"), ("title", "Dune")]
    with pytest.raises(metadata.MetadataError, match="barcode"):
        metadata.lookup("movie", "012345678905")


def test_missing_api_keys_give_clear_errors(monkeypatch):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.delenv("IGDB_CLIENT_ID", raising=False)
    with pytest.raises(metadata.MetadataError, match="TMDB_API_KEY"):
        metadata.lookup("movie", "Arrival")
    with pytest.raises(metadata.MetadataError, match="IGDB_CLIENT_ID"):
        metadata.lookup("game", "Hades")


# --------------------------------------------------------------------------- web
@pytest.fixture
def client(cat):
    app = create_app(cat.db_path)
    app.testing = True
    return app.test_client()


def test_web_browse_and_search(client):
    page = client.get("/").get_data(as_text=True)
    assert "Abbey Road" in page and "Hades" in page
    page = client.get("/?q=herbert").get_data(as_text=True)
    assert "Dune" in page and "Abbey Road" not in page
    assert "Did you mean" in client.get("/?q=beetles").get_data(as_text=True)


def test_web_add_edit_delete(client):
    resp = client.post("/add", data={"media_type": "game", "title": "Celeste", "creator": "Maddy Makes Games",
                                     "extra__platform": "PC", "year": "2018"})
    assert resp.status_code == 302 and resp.headers["Location"] == "/"  # back to the home screen
    home = client.get("/").get_data(as_text=True)
    assert "Added “Celeste”." in home
    item_id = client.get("/api/search?q=celeste").get_json()["results"][0]["id"]
    item_url = f"/item/{item_id}"
    assert "Celeste" in client.get(item_url).get_data(as_text=True)
    resp = client.post(item_url + "/edit", data={"media_type": "game", "title": "Celeste", "extra__platform": "Steam Deck"})
    assert resp.headers["Location"] == "/" and "Saved “Celeste”." in client.get("/").get_data(as_text=True)
    assert client.get("/api/search?q=steam deck").get_json()["results"][0]["title"] == "Celeste"
    client.post(item_url + "/delete")
    assert client.get(item_url).status_code == 404
    assert client.post("/add", data={"media_type": "book", "title": ""}).status_code == 400


def test_web_lookup_import_export(client, monkeypatch):
    monkeypatch.setattr(metadata, "lookup", lambda t, q: [{"title": "X", "media_type": t, "extra": {}, "source": "T"}])
    assert client.get("/api/lookup?type=book&q=x").get_json()["results"][0]["title"] == "X"

    def boom(t, q):
        raise metadata.MetadataError("offline")
    monkeypatch.setattr(metadata, "lookup", boom)
    resp = client.get("/api/lookup?type=book&q=x")
    assert resp.status_code == 502 and resp.get_json()["error"] == "offline"

    csv_bytes = b"media_type,title,creator,platform\ngame,Tetris,Alexey Pajitnov,Game Boy\n"
    client.post("/import", data={"file": (io.BytesIO(csv_bytes), "c.csv")}, content_type="multipart/form-data")
    assert client.get("/api/search?q=game boy").get_json()["results"][0]["title"] == "Tetris"
    assert "Tetris" in client.get("/export.csv").get_data(as_text=True)


# --------------------------------------------------------------------------- genres
@pytest.fixture
def music(tmp_path):
    c = Catalog(tmp_path / "music.db")
    c.add("music", "Swimming", creator="Mac Miller", genre="Hip-Hop, Jazz rap")
    c.add("music", "Kind of Blue", creator="Miles Davis", genre="Jazz")
    c.add("music", "To Pimp a Butterfly", creator="Kendrick Lamar", genre=["hip hop", "jazz-rap", "Funk"])
    c.add("game", "Rap Battle Royale", tags="rap battle", genre="Rhythm")
    return c


def test_item_can_have_several_genres_in_order(music):
    swimming = music.search("swimming")[0]
    assert swimming["genres"] == ["Hip-Hop", "Jazz rap"]
    assert swimming["genre"] == "Hip-Hop, Jazz rap"


def test_spelling_variants_share_one_genre(music):
    tpab = music.search("butterfly")[0]
    assert tpab["genres"] == ["Hip-Hop", "Jazz rap", "Funk"]  # reused existing spellings
    assert {g["name"]: g["count"] for g in music.genres()} == {
        "Hip-Hop": 2, "Jazz rap": 2, "Funk": 1, "Jazz": 1, "Rhythm": 1}


def test_genre_filter_is_exact(music):
    # Plain keyword search for "jazz rap" also hits the jazz album and the rap-tagged game...
    assert len(music.search("jazz rap")) == 4
    # ...the genre filter returns only items actually in that genre.
    assert {i["title"] for i in music.list(genre="jazz rap")} == {"Swimming", "To Pimp a Butterfly"}
    assert [i["title"] for i in music.search("mac", genre="JAZZ-RAP")] == ["Swimming"]
    assert music.list(media_type="game", genre="hip-hop") == []
    assert [g["name"] for g in music.genres("game")] == ["Rhythm"]


def test_update_replaces_genres_and_prunes_unused(music):
    blue = music.search("kind of blue")[0]
    music.update(blue["id"], genre="Modal jazz, Cool jazz")
    assert music.get(blue["id"])["genres"] == ["Modal jazz", "Cool jazz"]
    assert "Jazz" not in [g["name"] for g in music.genres()]
    music.update(blue["id"], notes="no genre change")  # genre not passed -> untouched
    assert music.get(blue["id"])["genres"] == ["Modal jazz", "Cool jazz"]
    music.remove(blue["id"])
    assert "Modal jazz" not in [g["name"] for g in music.genres()]


def test_rename_and_merge_genres(music):
    assert music.rename_genre("hip hop", "Hip hop") == 2  # pure rename (same key)
    assert music.search("swimming")[0]["genres"] == ["Hip hop", "Jazz rap"]
    music.add("music", "Madvillainy", genre="Underground hip-hop, Rap")
    assert music.rename_genre("Rap", "Hip hop") == 1  # merge into existing
    assert "Madvillainy" in [i["title"] for i in music.list(genre="hip hop")]
    assert "Rap" not in [g["name"] for g in music.genres()]
    with pytest.raises(ValueError):
        music.rename_genre("Polka", "Jazz")
    assert music.remove_genre("funk") == 1 and music.search("butterfly")[0]["genres"] == ["Hip hop", "Jazz rap"]


def test_genres_survive_csv_round_trip(music, tmp_path):
    copy = Catalog(tmp_path / "copy.db")
    copy.import_csv(io.StringIO(music.export_csv()))
    assert copy.search("swimming")[0]["genres"] == ["Hip-Hop", "Jazz rap"]


def test_migrates_v2_genre_text_into_genre_records(tmp_path):
    path = tmp_path / "v2.db"
    c = Catalog(path)
    c.add("music", "Swimming")
    c.conn.executescript("""
        UPDATE items SET genre = 'hip-hop, Jazz Rap; hip hop';
        DROP TABLE item_genres; DROP TABLE genres; PRAGMA user_version = 2;""")
    c.conn.close()
    migrated = Catalog(path)
    assert migrated.get(1)["genres"] == ["hip-hop", "Jazz Rap"]
    assert migrated.list(genre="jazz rap")[0]["title"] == "Swimming"


def test_web_genre_filter_chips_and_rename(music):
    app = create_app(music.db_path)
    app.testing = True
    client = app.test_client()
    page = client.get("/?genre=Jazz+rap").get_data(as_text=True)
    assert "Swimming" in page and "Kind of Blue" not in page and "Jazz rap ✕" in page
    resp = client.post("/add", data={"media_type": "music", "title": "Faces", "genre": "Hip-Hop, Jazz rap"})
    assert client.get(resp.headers["Location"]).status_code == 200
    assert {g["name"]: g["count"] for g in client.get("/api/genres").get_json()["genres"]}["Jazz rap"] == 3
    assert "Rename" in client.get("/genres").get_data(as_text=True)
    client.post("/genres/rename", data={"old": "Jazz rap", "new": "Jazz-rap"})
    assert "Jazz-rap" in [g["name"] for g in music.genres()]


# --------------------------------------------------------------------------- form fixes
def _datalist_data(page):
    m = re.search(r"const GENRES_BY_TYPE = (\{.*?\});", page)
    return json.loads(m.group(1))


def test_form_suggests_only_genres_for_its_media_type(music):
    music.add("book", "Dune", genre="Science fiction")
    music.add("movie", "Arrival", genre="Science fiction, Drama")
    app = create_app(music.db_path)
    app.testing = True
    page = app.test_client().get("/add?type=music").get_data(as_text=True)
    by_type = _datalist_data(page)
    assert set(by_type["music"]) == {"Hip-Hop", "Jazz rap", "Funk", "Jazz"}
    assert by_type["book"] == ["Science fiction"]
    assert set(by_type["movie"]) == {"Science fiction", "Drama"}
    assert by_type["game"] == ["Rhythm"]
    assert '<datalist id="genreList"></datalist>' in page  # filled by script for the selected type


def test_year_labels_and_edition_year_field(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    page = client.get("/add?type=book").get_data(as_text=True)
    assert "First published" in page and "Edition year (your copy)" in page
    client.post("/add", data={"media_type": "book", "title": "The Hobbit", "year": "1937",
                              "extra__edition_year": "2012", "extra__isbn": "9780547928227"})
    hobbit = cat.search("hobbit")[0]
    assert hobbit["year"] == 1937 and hobbit["extra"]["edition_year"] == "2012"


# --------------------------------------------------------------------------- details page
def test_details_page_shows_everything_saved(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    dune2 = cat.search("part two")[0]
    cat.update(dune2["id"], genre="Science fiction, Adventure", tags="desert, sequel", notes="Steelbook edition",
               extra={**dune2["extra"], "cover_url": "https://img/dune2.jpg", "rating": "PG-13"})
    page = client.get(f"/item/{dune2['id']}").get_data(as_text=True)
    for expected in ("Dune: Part Two", "Directed by", "Denis Villeneuve", "Release year", "2024",
                     "https://img/dune2.jpg", "166 min", "PG-13", "4K" if False else "Blu-ray",
                     ">Timothée Chalamet<", ">Zendaya<", "Science fiction", "Steelbook edition",
                     "Edit details", "Add another copy", "View on TMDB", "Delete"):
        assert expected in page, expected
    assert 'name="title"' not in page  # read-only, not the edit form
    # Edit form lives at /edit and its Cancel goes back to the details page.
    edit_page = client.get(f"/item/{dune2['id']}/edit").get_data(as_text=True)
    assert 'name="title"' in edit_page and f'href="/item/{dune2["id"]}">Cancel' in edit_page


def test_details_page_track_list_and_book_link(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    abbey = cat.search("abbey road")[0]
    page = client.get(f"/item/{abbey['id']}").get_data(as_text=True)
    assert "<li>Come Together</li>" in page and "<li>Here Comes the Sun</li>" in page
    dune = cat.search("9780441013593")[0]
    page = client.get(f"/item/{dune['id']}").get_data(as_text=True)
    assert "https://openlibrary.org/isbn/9780441013593" in page and "First published" in page
    assert client.get("/item/9999").status_code == 404


def test_add_another_copy_prefills_without_saving(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    dune = cat.search("9780441013593")[0]
    before = len(cat.list())
    page = client.get(f"/add?copy={dune['id']}").get_data(as_text=True)
    assert "Add another copy" in page and 'value="Frank Herbert"' in page and "9780441013593" in page
    assert len(cat.list()) == before  # nothing saved until the form is submitted


def test_more_like_this(cat):
    dune = cat.search("9780441013593")[0]
    related = cat.similar(dune["id"])
    assert related[0]["title"] == "Dune: Part Two" and "Related title" in related[0]["reason"]
    mythos = cat.search("mythos")[0]
    assert cat.similar(mythos["id"]) == []  # nothing in common with anything -> nothing listed
    cat.update(mythos["id"], tags="greek mythology")
    assert [(r["title"], r["reason"]) for r in cat.similar(mythos["id"])] == [("Hades", "Tagged greek mythology")]
    cat.add("book", "Children of Dune", creator="Frank Herbert")
    reasons = {r["title"]: r["reason"] for r in cat.similar(dune["id"])}
    assert reasons["Children of Dune"].startswith("Also by Frank Herbert")


def test_more_like_this_uses_meaning_when_available(semantic_cat):
    foundation = semantic_cat.search("foundation")[0]
    semantic_cat.add("movie", "Interstellar", notes="space travel through a wormhole near a distant planet")
    related = semantic_cat.similar(foundation["id"])
    assert [r["title"] for r in related] == ["Interstellar"] and related[0]["reason"] == "Similar subject"


# --------------------------------------------------------------------------- cover uploads
from encyclomedia import covers  # noqa: E402


def make_image(fmt="JPEG", size=(400, 600), color=(200, 120, 40), mode="RGB", exif_orientation=None):
    from PIL import Image
    img = Image.new(mode, size, color)
    out = io.BytesIO()
    kwargs = {}
    if exif_orientation:
        exif = Image.Exif()
        exif[0x0112] = exif_orientation  # Orientation tag
        kwargs["exif"] = exif
    img.save(out, fmt, **kwargs)
    return out.getvalue()


def test_prepare_rejects_non_images():
    for bad in (b"", b"just some text pretending to be a jpg", b"%PDF-1.7 ..."):
        with pytest.raises(covers.CoverError):
            covers.prepare(bad)
    with pytest.raises(covers.CoverError, match="damaged"):
        covers.prepare(b"\xff\xd8\xff" + b"\x00" * 100)  # JPEG header, garbage body


def test_prepare_resizes_and_rotates_phone_photos():
    from PIL import Image
    # A 4000x3000 photo taken with the phone turned sideways (EXIF orientation 6 = rotate 90°).
    data, ext = covers.prepare(make_image(size=(4000, 3000), exif_orientation=6))
    with Image.open(io.BytesIO(data)) as img:
        assert ext == "jpg" and img.size[0] < img.size[1]          # now portrait, the right way up
        assert img.size[0] <= 1200 and img.size[1] <= 1800         # shrunk
    data, ext = covers.prepare(make_image("PNG", mode="RGBA", color=(0, 0, 0, 0)))
    assert ext == "png"                                            # transparency kept
    gif = make_image("GIF", mode="P", color=1)
    assert covers.prepare(gif) == (gif, "gif")                    # left alone (may be animated)


def test_cover_files_are_created_replaced_and_cleaned_up(cat):
    hades = cat.search("hades")[0]
    url1 = cat.set_cover_image(hades["id"], make_image())
    file1 = cat.covers_dir / url1.rsplit("/", 1)[1]
    assert url1.startswith("/covers/") and file1.is_file()
    assert cat.covers_dir.name == "test_covers"                    # next to test.db
    assert cat.get(hades["id"])["extra"]["cover_url"] == url1

    url2 = cat.set_cover_image(hades["id"], make_image(color=(10, 10, 10)))
    assert url2 != url1 and not file1.exists()                     # old file deleted on replace

    copy_id = cat.add("game", "Hades (2nd copy)", extra={"cover_url": url2})  # copies can share a file
    cat.remove(hades["id"])
    assert cat.cover_file(url2.rsplit("/", 1)[1]) is not None      # still used by the copy
    assert cat.clear_cover(copy_id)
    assert cat.cover_file(url2.rsplit("/", 1)[1]) is None          # last user gone -> file gone
    assert cat.cover_file("../test.db") is None and cat.cover_file("x.exe") is None


def test_web_cover_upload_flow(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()

    # Upload while adding a new item.
    resp = client.post("/add", content_type="multipart/form-data", data={
        "media_type": "book", "title": "Leviathan Wakes",
        "cover_file": (io.BytesIO(make_image()), "photo.jpg")})
    assert resp.headers["Location"] == "/"
    book = cat.search("leviathan")[0]
    url = book["extra"]["cover_url"]
    with client.get(url) as img:  # close the response so the file isn't left open (matters on Windows)
        assert img.status_code == 200 and img.mimetype == "image/jpeg"
    assert f'src="{url}"' in client.get(f"/item/{book['id']}").get_data(as_text=True)

    # Saving the edit form unchanged keeps the cover; clearing cover_url removes it and its file.
    form = {"media_type": "book", "title": "Leviathan Wakes", "cover_url": url}
    client.post(f"/item/{book['id']}/edit", data=form)
    assert cat.get(book["id"])["extra"]["cover_url"] == url
    client.post(f"/item/{book['id']}/edit", data={**form, "cover_url": ""})
    assert "cover_url" not in cat.get(book["id"])["extra"]
    with client.get(url) as gone:
        assert gone.status_code == 404

    # Quick upload from the details page; a bad file is refused with a message.
    resp = client.post(f"/item/{book['id']}/cover", content_type="multipart/form-data",
                       data={"cover_file": (io.BytesIO(make_image("PNG")), "scan.png")})
    assert resp.headers["Location"] == f"/item/{book['id']}"
    quick = cat.get(book["id"])["extra"]["cover_url"]
    with client.get(quick) as resp:
        assert quick.startswith("/covers/") and resp.status_code == 200
    client.post(f"/item/{book['id']}/cover", content_type="multipart/form-data",
                data={"cover_file": (io.BytesIO(b"not an image"), "notes.jpg")})
    page = client.get(f"/item/{book['id']}").get_data(as_text=True)
    assert "a supported image" in page
    assert cat.get(book["id"])["extra"]["cover_url"] == quick  # old cover kept

    client.post(f"/item/{book['id']}/cover/remove")
    assert "cover_url" not in cat.get(book["id"])["extra"]


def test_web_bad_upload_on_add_keeps_item_and_reopens_form(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    resp = client.post("/add", content_type="multipart/form-data", data={
        "media_type": "game", "title": "Celeste", "cover_file": (io.BytesIO(b"oops"), "cover.png")})
    item = cat.search("celeste")[0]
    assert resp.headers["Location"] == f"/item/{item['id']}/edit"
    assert "The cover wasn&#39;t saved" in client.get(resp.headers["Location"]).get_data(as_text=True)


def test_web_rejects_oversized_upload(cat):
    app = create_app(cat.db_path)
    app.testing = True
    app.config["MAX_CONTENT_LENGTH"] = 2000
    client = app.test_client()
    dune = cat.search("9780441013593")[0]
    resp = client.post(f"/item/{dune['id']}/cover", content_type="multipart/form-data",
                       data={"cover_file": (io.BytesIO(b"x" * 5000), "big.jpg")},
                       headers={"Referer": f"/item/{dune['id']}"})
    assert resp.status_code == 302
    assert "too big" in client.get(f"/item/{dune['id']}").get_data(as_text=True)


def test_locked_cover_file_does_not_break_saving(cat, monkeypatch):
    """On Windows an open file can't be deleted. Removing a cover must still work,
    and the leftover file is cleaned up later."""
    from pathlib import Path
    hades = cat.search("hades")[0]
    url = cat.set_cover_image(hades["id"], make_image())
    path = cat.covers_dir / url.rsplit("/", 1)[1]

    real_unlink = Path.unlink

    def locked(self, *a, **kw):
        raise PermissionError(32, "The process cannot access the file because it is being used")

    monkeypatch.setattr(Path, "unlink", locked)
    assert cat.clear_cover(hades["id"])                 # no crash
    assert "cover_url" not in cat.get(hades["id"])["extra"] and path.exists()

    monkeypatch.setattr(Path, "unlink", real_unlink)  # file no longer locked
    assert cat.prune_covers() == 1 and not path.exists()
    create_app(cat.db_path)                            # starting the app also tidies up


# --------------------------------------------------------------------------- "your copy" fields
def test_copy_fields_work_for_every_media_type(cat):
    for media_type in ("book", "music", "movie", "game"):
        item_id = cat.add(media_type, f"Test {media_type}", extra={
            "edition": "Collector's edition", "location": "Shelf B1",
            "origin": "Gift from Chris", "condition": "Like new"})
        extra = cat.get(item_id)["extra"]
        assert (extra["edition"], extra["location"], extra["origin"], extra["condition"]) == (
            "Collector's edition", "Shelf B1", "Gift from Chris", "Like new")


def test_copy_fields_are_searchable_and_filterable(cat):
    dune = cat.search("9780441013593")[0]
    hades = cat.search("hades")[0]
    cat.update(dune["id"], extra={**dune["extra"], "location": "Living room shelf A2", "origin": "Barnes & Noble",
                                  "condition": "Spine creased"})
    cat.update(hades["id"], extra={**hades["extra"], "location": "living room SHELF a2 ", "origin": "GameStop"})
    assert cat.search("creased")[0]["title"] == "Dune"             # condition notes are searchable
    # Exact location filter ignores case and stray spaces, and doesn't match partial words.
    assert {i["title"] for i in cat.list(location="Living Room Shelf A2")} == {"Dune", "Hades"}
    assert cat.list(location="Shelf A") == []
    assert [i["title"] for i in cat.list(origin="gamestop")] == ["Hades"]
    assert [i["title"] for i in cat.search("dune", location="living room shelf a2")] == ["Dune"]
    assert [i["title"] for i in cat.list(media_type="game", location="living room shelf a2")] == ["Hades"]
    # The suggestion list merges spelling variants under the most common spelling.
    cat.add("music", "Kind of Blue", extra={"location": "Living room shelf A2"})
    assert cat.copy_values("location") == [{"name": "Living room shelf A2", "count": 3}]
    with pytest.raises(ValueError):
        cat.copy_values("isbn")


def test_copy_fields_csv_round_trip(cat, tmp_path):
    dune = cat.search("9780441013593")[0]
    cat.update(dune["id"], extra={**dune["extra"], "edition": "First edition", "location": "Shelf A2",
                                  "origin": "Estate sale", "condition": "Foxing on pages"})
    text = cat.export_csv()
    header = text.splitlines()[0].split(",")
    assert header[8:12] == ["edition", "location", "origin", "condition"]  # right after the core fields
    copy = Catalog(tmp_path / "copy.db")
    copy.import_csv(io.StringIO(text))
    extra = copy.search("9780441013593")[0]["extra"]
    assert (extra["edition"], extra["location"], extra["origin"], extra["condition"]) == (
        "First edition", "Shelf A2", "Estate sale", "Foxing on pages")


def test_existing_book_edition_still_shows(cat):
    """Books saved before this change already had an 'edition'; it now appears under Your copy."""
    app = create_app(cat.db_path)
    app.testing = True
    dune = cat.search("9780441013593")[0]
    cat.update(dune["id"], extra={**dune["extra"], "edition": "Ace mass-market"})
    page = app.test_client().get(f"/item/{dune['id']}").get_data(as_text=True)
    assert "Your copy" in page and "Ace mass-market" in page
    assert page.count("Ace mass-market") == 1                      # not listed twice


def test_web_form_saves_copy_fields_and_details_page_shows_them(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    client.post("/add", data={"media_type": "movie", "title": "Arrival", "extra__edition": "4K Steelbook",
                              "extra__location": "Media cabinet", "extra__origin": "Best Buy",
                              "extra__condition": "Sealed\nSlipcover has a dent"})
    arrival = cat.search("arrival")[0]
    assert arrival["extra"]["condition"] == "Sealed\nSlipcover has a dent"
    page = client.get(f"/item/{arrival['id']}").get_data(as_text=True)
    for text in ("Your copy", "Edition", "4K Steelbook", "Condition notes", "Slipcover has a dent",
                 'href="/?location=Media+cabinet"', 'href="/?origin=Best+Buy"'):
        assert text in page, text
    # Browse filtered by location, with a removable filter chip and the location on each card.
    browse = client.get("/?location=media+cabinet").get_data(as_text=True)
    assert "Arrival" in browse and "Dune" not in browse and "Location: Media cabinet" in browse
    assert 'title="Remove filter">media cabinet ✕' in browse

    # Editing keeps them unless changed, and the form suggests locations already used.
    edit = client.get(f"/item/{arrival['id']}/edit").get_data(as_text=True)
    assert 'value="4K Steelbook"' in edit and '<option value="Media cabinet">' in edit
    client.post(f"/item/{arrival['id']}/edit", data={"media_type": "movie", "title": "Arrival",
                                                     "extra__location": "Storage box 3"})
    extra = cat.get(arrival["id"])["extra"]
    assert extra["location"] == "Storage box 3" and "edition" not in extra  # blank fields are cleared


def test_add_another_copy_starts_copy_fields_blank(cat):
    app = create_app(cat.db_path)
    app.testing = True
    dune = cat.search("9780441013593")[0]
    cat.update(dune["id"], extra={**dune["extra"], "edition": "First edition", "location": "Shelf A2",
                                  "origin": "Estate sale", "condition": "Worn"})
    page = app.test_client().get(f"/add?copy={dune['id']}").get_data(as_text=True)
    assert 'value="Frank Herbert"' in page and "9780441013593" in page  # shared details copied
    for field in ("edition", "location", "origin"):                        # copy-specific ones are not
        assert re.search(rf'name="extra__{field}" value=""', page), field
    assert re.search(r'name="extra__condition"[^>]*>\s*</textarea>', page)


def test_cli_copy_fields(tmp_path, capsys):
    from encyclomedia.__main__ import main
    db = str(tmp_path / "cli.db")
    main(["--db", db, "--no-semantic", "add", "game", "Hades", "--location", "Media cabinet",
          "--origin", "GameStop", "--edition", "Physical", "--condition", "Like new"])
    main(["--db", db, "--no-semantic", "add", "book", "Dune", "--location", "Shelf A2"])
    main(["--db", db, "--no-semantic", "edit", "2", "--condition", "Spine creased"])
    capsys.readouterr()
    main(["--db", db, "--no-semantic", "list", "--location", "media cabinet"])
    assert "Hades" in capsys.readouterr().out
    main(["--db", db, "--no-semantic", "locations"])
    out = capsys.readouterr().out
    assert "Media cabinet" in out and "Shelf A2" in out
    main(["--db", db, "--no-semantic", "show", "2"])
    out = capsys.readouterr().out
    assert "Your copy:" in out and "Spine creased" in out and "Shelf A2" in out


# --------------------------------------------------------------------------- per-type credits
def test_credit_fields_per_type():
    from encyclomedia.catalog import CREATOR_LABEL, CREDIT_FIELDS, EXTRA_FIELDS
    assert CREATOR_LABEL == {"book": "Author", "music": "Artist", "movie": "Director", "game": "Game studio"}
    assert CREDIT_FIELDS == {"book": ("illustrator",), "music": ("performers", "label"),
                             "movie": ("composer",), "game": ("designer",)}
    for t, fields in CREDIT_FIELDS.items():
        assert EXTRA_FIELDS[t][:len(fields)] == fields  # stored with the type, listed first


def test_credits_are_saved_searchable_and_round_trip(cat, tmp_path):
    cat.add("book", "Watchmen", creator="Alan Moore", extra={"illustrator": "Dave Gibbons"})
    cat.add("movie", "Interstellar", creator="Christopher Nolan", extra={"composer": "Hans Zimmer"})
    cat.add("music", "Faces", creator="Mac Miller", extra={"performers": "Earl Sweatshirt, Rick Ross",
                                                          "label": "REMember Music"})
    cat.add("game", "Stardew Valley", creator="ConcernedApe", extra={"designer": "Eric Barone"})
    assert cat.search("gibbons")[0]["title"] == "Watchmen"
    assert cat.search("zimmer")[0]["title"] == "Interstellar"
    assert cat.search("earl sweatshirt")[0]["title"] == "Faces"
    assert cat.search("barone")[0]["title"] == "Stardew Valley"
    copy = Catalog(tmp_path / "copy.db")
    copy.import_csv(io.StringIO(cat.export_csv()))
    assert copy.search("watchmen")[0]["extra"]["illustrator"] == "Dave Gibbons"
    assert copy.search("faces")[0]["extra"]["label"] == "REMember Music"


def test_shared_credits_power_more_like_this(cat):
    dune2 = cat.search("part two")[0]
    cat.update(dune2["id"], extra={**dune2["extra"], "composer": "Hans Zimmer"})
    inter = cat.add("movie", "Interstellar", creator="Christopher Nolan", extra={"composer": "Hans Zimmer"})
    related = {r["title"]: r["reason"] for r in cat.similar(inter)}
    assert related["Dune: Part Two"] == "Also with Hans Zimmer"


def test_details_page_credits_section(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    tpab = cat.add("music", "To Pimp a Butterfly", creator="Kendrick Lamar",
                   extra={"performers": "Bilal, Anna Wise", "label": "Top Dawg"})
    page = client.get(f"/item/{tpab}").get_data(as_text=True)
    for text in ("<h2>Credits</h2>", "<dt>Artist</dt>", "<dt>Performers</dt>", "<dt>Record label</dt>",
                 'href="/?q=Bilal"', 'href="/?q=Anna+Wise"', "Top Dawg"):
        assert text in page, text
    assert 'href="/?q=Top+Dawg"' not in page        # the label isn't a person to search for
    assert page.count("Top Dawg") == 1              # not repeated in the details table

    # A game: the studio is the main credit, the designer sits beside it.
    hades = cat.search("hades")[0]
    cat.update(hades["id"], extra={**hades["extra"], "designer": "Greg Kasavin"})
    page = client.get(f"/item/{hades['id']}").get_data(as_text=True)
    assert "<dt>Game studio</dt>" in page and "<dt>Game designer</dt>" in page

    # With only the main credit, the header line covers it and no Credits section is shown.
    mythos = cat.search("mythos")[0]
    assert "<h2>Credits</h2>" not in client.get(f"/item/{mythos['id']}").get_data(as_text=True)


def test_form_shows_and_saves_each_types_credits(cat):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    page = client.get("/add?type=book").get_data(as_text=True)
    for name in ("illustrator", "performers", "label", "composer", "designer"):
        assert f'name="extra__{name}"' in page and page.count(f'name="extra__{name}"') == 1, name
    assert "Game studio" in page                        # creator label for games, used by the script
    # Only the submitted type's credits are kept, even if other types' boxes were filled.
    client.post("/add", data={"media_type": "movie", "title": "Arrival", "creator": "Denis Villeneuve",
                              "extra__composer": "Jóhann Jóhannsson", "extra__illustrator": "ignored"})
    arrival = cat.search("arrival")[0]
    assert arrival["extra"].get("composer") == "Jóhann Jóhannsson" and "illustrator" not in arrival["extra"]


def test_lookup_parsers_fill_credits():
    r = {"id": "x", "title": "Wesley's Theory",
         "artist-credit": [{"name": "Kendrick Lamar", "joinphrase": " feat. "},
                           {"name": "George Clinton", "joinphrase": " & "},
                           {"name": "Thundercat", "joinphrase": ""}]}
    c = metadata.parse_musicbrainz_release(r)
    assert c["creator"] == "Kendrick Lamar" and c["extra"]["performers"] == "George Clinton, Thundercat"
    joint = metadata.parse_musicbrainz_release({"id": "y", "title": "Watch the Throne", "artist-credit": [
        {"name": "JAY-Z", "joinphrase": " & "}, {"name": "Kanye West", "joinphrase": ""}]})
    assert joint["creator"] == "JAY-Z & Kanye West" and "performers" not in joint["extra"]
    m = metadata.parse_tmdb_movie({"id": 1, "title": "Dune: Part Two", "credits": {"crew": [
        {"job": "Director", "name": "Denis Villeneuve"},
        {"job": "Original Music Composer", "name": "Hans Zimmer"}]}})
    assert m["creator"] == "Denis Villeneuve" and m["extra"]["composer"] == "Hans Zimmer"


def test_cli_credit_options(tmp_path, capsys):
    from encyclomedia.__main__ import main
    db = str(tmp_path / "cli.db")
    main(["--db", db, "--no-semantic", "add", "book", "Watchmen", "--creator", "Alan Moore",
          "--illustrator", "Dave Gibbons"])
    main(["--db", db, "--no-semantic", "add", "music", "Faces", "--performers", "Rick Ross", "--label", "REMember"])
    with pytest.raises(SystemExit, match="--composer is for movies"):
        main(["--db", db, "--no-semantic", "add", "book", "Dune", "--composer", "Hans Zimmer"])
    main(["--db", db, "--no-semantic", "edit", "1", "--illustrator", "Dave Gibbons, John Higgins"])
    capsys.readouterr()
    main(["--db", db, "--no-semantic", "show", "1"])
    assert "Dave Gibbons, John Higgins" in capsys.readouterr().out


# --------------------------------------------------------------------------- per-type pages
@pytest.fixture
def web(cat):
    app = create_app(cat.db_path)
    app.testing = True
    return app.test_client()


@pytest.mark.parametrize("path, title, present, absent", [
    ("/books", "Books", ["Dune", "Mythos"], ["Dune: Part Two", "Abbey Road", "Hades"]),
    ("/music", "Music", ["Abbey Road"], ["Mythos", "Hades"]),
    ("/movies", "Movies", ["Dune: Part Two"], ["Mythos", "Abbey Road"]),
    ("/games", "Games", ["Hades"], ["Mythos", "Abbey Road"]),
])
def test_each_media_type_has_its_own_page(web, path, title, present, absent):
    page = web.get(path).get_data(as_text=True)
    assert f"<h1" in page and f">{title}</h1>" in page
    for t in present:
        assert f">{t}</a></h3>" in page, t
    for t in absent:
        assert f">{t}</a></h3>" not in page, t
    assert 'name="type"' not in page                       # no media-type picker on a type page
    assert "Import CSV" not in page


def test_all_page_still_lists_everything_and_tabs_show_counts(web):
    page = web.get("/").get_data(as_text=True)
    for t in ("Dune", "Dune: Part Two", "Abbey Road", "Hades", "Mythos"):
        assert f">{t}</a></h3>" in page
    assert re.search(r'class="tab active" href="/">All\s*<span class="n">5</span>', page)
    assert re.search(r'href="/books">\s*<span class="dot t-book"></span>Books <span class="n">2</span>', page)
    assert "Import CSV" in page


def test_search_and_genre_filter_stay_on_the_type_page(web, cat):
    cat.add("movie", "Dune", creator="David Lynch", genre="Science fiction")
    page = web.get("/books?q=dune").get_data(as_text=True)
    assert ">Dune</a></h3>" in page and "David Lynch" not in page and "Dune: Part Two" not in page
    assert 'action="/books"' in page
    page = web.get("/movies?genre=Science+fiction").get_data(as_text=True)
    assert "David Lynch" in page and 'href="/movies"' in page    # filter chip removes back to /movies
    assert "Frank Herbert" not in page


def test_old_type_links_redirect_to_type_pages(web):
    resp = web.get("/?type=book&genre=Fantasy&q=x")
    assert resp.status_code == 302 and resp.headers["Location"] == "/books?genre=Fantasy&q=x"
    assert web.get("/comics").status_code == 404


def test_sorting(web, cat):
    cat.add("book", "Neuromancer", creator="William Gibson", year=1984)
    titles = lambda html: re.findall(r'<h3><a href="/item/\d+">([^<]+)</a></h3>', html)
    assert titles(web.get("/books").get_data(as_text=True)) == ["Dune", "Mythos", "Neuromancer"]
    assert titles(web.get("/books?sort=newest").get_data(as_text=True)) == ["Neuromancer", "Dune", "Mythos"]  # no year -> last
    assert titles(web.get("/books?sort=creator").get_data(as_text=True)) == ["Mythos", "Neuromancer", "Dune"]  # Fry, Gibson, Herbert
    assert titles(web.get("/books?sort=added").get_data(as_text=True))[0] == "Neuromancer"
    page = web.get("/books").get_data(as_text=True)
    assert '<option value="creator" >Author</option>' in page or ">Author</option>" in page
    assert web.get("/books?sort=nonsense").status_code == 200   # unknown sort falls back to title


def test_detail_and_add_links_point_to_type_pages(web, cat):
    dune2 = cat.search("part two")[0]
    page = web.get(f"/item/{dune2['id']}").get_data(as_text=True)
    assert '<a href="/movies">Movies</a> ›' in page                   # breadcrumb
    assert 'class="tab active" href="/movies"' in page                 # its tab is highlighted
    books = web.get("/books").get_data(as_text=True)
    assert 'href="/add?type=book"' in books and "Add a book" in books
    assert "Add an album" in web.get("/music").get_data(as_text=True)


def test_empty_type_page(tmp_path):
    app = create_app(tmp_path / "empty.db")
    app.testing = True
    page = app.test_client().get("/games").get_data(as_text=True)
    assert "No games in your catalog yet." in page


def test_export_one_media_type(web, cat):
    text = web.get("/export.csv?type=music").get_data(as_text=True)
    lines = text.strip().splitlines()
    assert len(lines) == 2 and "Abbey Road" in lines[1]
    assert "isbn" not in lines[0] and "tracklist" in lines[0]          # only music's columns
    assert "encyclomedia-music.csv" in web.get("/export.csv?type=music").headers["Content-Disposition"]
    assert len(web.get("/export.csv").get_data(as_text=True).strip().splitlines()) == 6


# --------------------------------------------------------------------------- ISBN auto-fill
from encyclomedia import autofill  # noqa: E402


def test_normalize_isbn():
    n = metadata.normalize_isbn
    assert n("978-0-441-01359-3") == "9780441013593"
    assert n(" 0 441 17271 7 ") == "0441172717"
    assert n("080442957x") == "080442957X"                 # X check digit
    assert n("441172717") == "0441172717"                  # spreadsheet dropped the leading zero
    assert n("9780441013593.0") == "9780441013593"         # numeric spreadsheet column
    assert n("9.78044E+12") is None                         # Excel scientific notation: digits lost
    assert n("9780441013594") is None                       # wrong check digit (typo)
    assert n("") is None and n(None) is None


def fake_openlibrary(books: dict, years: dict = None, details: dict = None, calls: list = None):
    """A stand-in for metadata._http answering the three Open Library requests."""
    def fake(url, params=None, **kw):
        if calls is not None:
            calls.append((url, dict(params or {})))
        keys = (params or {}).get("bibkeys", "").split(",")
        if url.endswith("/api/books") and params["jscmd"] == "data":
            return {k: books[k[5:]] for k in keys if k[5:] in books}
        if url.endswith("/api/books"):
            return {k: {"details": details[k[5:]]} for k in keys if k[5:] in (details or {})}
        return {"docs": [{"isbn": [i], "first_publish_year": y} for i, y in (years or {}).items()]}
    return fake


DUNE_ED = {"title": "Dune", "authors": [{"name": "Frank Herbert"}], "publish_date": "2005",
           "publishers": [{"name": "Ace"}], "number_of_pages": 604,
           "cover": {"medium": "https://covers/m.jpg", "large": "https://covers/l.jpg"}}
HOBBIT_ED = {"title": "The Hobbit", "authors": [{"name": "J.R.R. Tolkien"}], "publish_date": "2012",
             "publishers": [{"name": "Houghton Mifflin"}], "number_of_pages": 300,
             "cover": {"medium": "https://covers/h.jpg"}}


def test_batch_lookup_uses_few_requests_and_reads_edition_details(monkeypatch):
    calls = []
    monkeypatch.setattr(metadata, "_http", fake_openlibrary(
        {"9780441013593": DUNE_ED, "9780547928227": HOBBIT_ED},
        years={"9780441013593": 1965, "9780547928227": 1937},
        details={"9780441013593": {"edition_name": "40th anniversary edition", "physical_format": "Paperback"}},
        calls=calls))
    found = metadata.books_by_isbns(["978-0-441-01359-3", "9780547928227", "9780000000002", "junk"], pause=0)
    assert set(found) == {"9780441013593", "9780547928227"}
    dune = found["9780441013593"]
    assert (dune["year"], dune["format"]) == (1965, "Paperback")
    assert dune["extra"]["edition"] == "40th anniversary edition" and dune["extra"]["edition_year"] == 2005
    assert dune["extra"]["cover_url"] == "https://covers/l.jpg"           # large cover preferred
    assert found["9780547928227"]["year"] == 1937
    assert len(calls) == 3                                                  # one batch: data, details, search
    assert metadata.books_by_isbns(["x"] * 3) == {}                         # nothing valid: no requests


def test_autofill_fills_only_blank_fields(cat, monkeypatch):
    monkeypatch.setattr(metadata, "_http", fake_openlibrary(
        {"9780441013593": DUNE_ED}, years={"9780441013593": 1965},
        details={"9780441013593": {"edition_name": "40th anniversary edition", "physical_format": "Paperback"}}))
    dune = cat.search("9780441013593")[0]          # fixture: creator, year, publisher "Ace", pages 604 set
    cat.update(dune["id"], format="hardcover")
    report = autofill.autofill_books(cat, [dune["id"]], lookup=lambda i: metadata.books_by_isbns(i, pause=0))
    after = cat.get(dune["id"])
    assert report.filled == ["Dune"]
    assert after["extra"]["cover_url"] == "https://covers/l.jpg"
    assert after["extra"]["edition"] == "40th anniversary edition" and after["extra"]["edition_year"] == 2005
    assert after["format"] == "hardcover"            # existing value kept
    assert after["year"] == 1965 and after["creator"] == "Frank Herbert"


def test_csv_import_autofills_books_with_isbns(tmp_path, monkeypatch):
    monkeypatch.setattr(metadata, "_http", fake_openlibrary(
        {"9780441013593": DUNE_ED, "9780547928227": HOBBIT_ED},
        years={"9780441013593": 1965, "9780547928227": 1937}))
    csv_text = ("media_type,title,creator,isbn,location\n"
                "book,Dune,,978-0-441-01359-3,Shelf A2\n"
                "book,My Hobbit,Tolkien (my spelling),9780547928227,\n"
                "book,Mystery Book,,9780000000002,\n"
                "book,Excel Victim,,9.78044E+12,\n"
                "book,No ISBN,Someone,,\n"
                "movie,Arrival,Denis Villeneuve,,\n")
    app = create_app(tmp_path / "imp.db")
    app.testing = True
    client = app.test_client()
    client.post("/import", content_type="multipart/form-data",
                data={"file": (io.BytesIO(csv_text.encode()), "c.csv"), "autofill": "1"})
    page = client.get("/").get_data(as_text=True)
    assert "Imported 6 items." in page
    assert "Filled in details for 2 books from their ISBNs." in page
    assert "Not found on Open Library: Mystery Book (9780000000002)." in page
    assert "Excel Victim (9.78044E+12)" in page

    cat = Catalog(tmp_path / "imp.db")
    dune = cat.search("shelf a2")[0]
    assert dune["creator"] == "Frank Herbert" and dune["year"] == 1965
    assert dune["extra"]["cover_url"] == "https://covers/l.jpg" and dune["extra"]["location"] == "Shelf A2"
    hobbit = cat.search("my hobbit")[0]
    assert hobbit["title"] == "My Hobbit" and hobbit["creator"] == "Tolkien (my spelling)"   # CSV wins
    assert hobbit["extra"]["cover_url"] == "https://covers/h.jpg"


def test_import_without_autofill_or_offline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(metadata, "_http", fake_openlibrary({}, calls=calls))
    app = create_app(tmp_path / "imp.db")
    app.testing = True
    client = app.test_client()
    csv_bytes = b"media_type,title,isbn\nbook,Dune,9780441013593\n"
    client.post("/import", content_type="multipart/form-data",
                data={"file": (io.BytesIO(csv_bytes), "c.csv")})          # box unticked
    assert calls == []

    def offline(*a, **kw):
        raise metadata.MetadataError("Could not reach openlibrary.org")
    monkeypatch.setattr(metadata, "_http", offline)
    client.post("/import", content_type="multipart/form-data",
                data={"file": (io.BytesIO(csv_bytes), "c.csv"), "autofill": "1"})
    page = client.get("/").get_data(as_text=True)
    assert "Imported 1 item." in page and "Could not reach openlibrary.org" in page
    assert len(Catalog(tmp_path / "imp.db").list("book")) == 2              # items still imported


def test_books_page_fill_in_button(cat, monkeypatch):
    app = create_app(cat.db_path)
    app.testing = True
    client = app.test_client()
    assert "Fill in from ISBNs (1)" in client.get("/books").get_data(as_text=True)  # Dune has an ISBN
    assert "Fill in from ISBNs" not in client.get("/movies").get_data(as_text=True)
    monkeypatch.setattr(metadata, "_http", fake_openlibrary(
        {"9780441013593": DUNE_ED},
        details={"9780441013593": {"edition_name": "Ace mass-market", "physical_format": "Paperback"}}))
    resp = client.post("/books/autofill")
    assert resp.headers["Location"] == "/books"
    page = client.get("/books").get_data(as_text=True)
    assert "Filled in details for 1 book from their ISBNs." in page
    assert "Fill in from ISBNs" not in page                 # nothing left to fill
    assert cat.search("9780441013593")[0]["extra"]["edition"] == "Ace mass-market"


def test_cli_import_autofill(tmp_path, monkeypatch, capsys):
    from encyclomedia.__main__ import main
    monkeypatch.setattr(metadata, "_http", fake_openlibrary({"9780441013593": DUNE_ED}))
    csv_path = tmp_path / "books.csv"
    csv_path.write_text("media_type,title,isbn\nbook,Dune,9780441013593\n", encoding="utf-8")
    db = str(tmp_path / "cli.db")
    main(["--db", db, "--no-semantic", "import", str(csv_path)])
    assert "Filled in details for 1 book" in capsys.readouterr().out
    main(["--db", db, "--no-semantic", "import", "--no-autofill", str(csv_path)])
    assert "Filled in" not in capsys.readouterr().out
    main(["--db", db, "--no-semantic", "autofill"])
    assert "Filled in details for 1 book" in capsys.readouterr().out


# --------------------------------------------------------------------------- several genres at once
@pytest.fixture
def rock(tmp_path):
    c = Catalog(tmp_path / "rock.db")
    c.add("music", "Abbey Road", genre="Rock")
    c.add("music", "The Piper at the Gates of Dawn", genre="Psychedelic rock, Rock")
    c.add("music", "Revolver", genre="Psychedelic rock")
    c.add("music", "Kind of Blue", genre="Jazz")
    c.add("movie", "Yellow Submarine", genre="Animation, Psychedelic rock")
    return c


def titles_of(items):
    return {i["title"] for i in items}


def test_catalog_any_and_all_genres(rock):
    both = ["Rock", "Psychedelic rock"]
    assert titles_of(rock.list(genre=both)) == {"Abbey Road", "The Piper at the Gates of Dawn", "Revolver",
                                               "Yellow Submarine"}
    assert titles_of(rock.list(genre=both, genre_match="all")) == {"The Piper at the Gates of Dawn"}
    assert titles_of(rock.list("music", both)) == {"Abbey Road", "The Piper at the Gates of Dawn", "Revolver"}
    assert titles_of(rock.list(genre="rock, psychedelic-rock")) == titles_of(rock.list(genre=both))  # spellings
    assert titles_of(rock.search("the", genre=both)) == {"The Piper at the Gates of Dawn"}
    assert titles_of(rock.list(genre=["Rock"], genre_match="all")) == {"Abbey Road", "The Piper at the Gates of Dawn"}


def test_web_select_several_genres(rock):
    app = create_app(rock.db_path)
    app.testing = True
    client = app.test_client()
    shown = lambda html: set(re.findall(r'<h3><a href="/item/\d+">([^<]+)</a></h3>', html))

    page = client.get("/music?genre=Rock&genre=Psychedelic+rock").get_data(as_text=True)
    assert shown(page) == {"Abbey Road", "The Piper at the Gates of Dawn", "Revolver"}
    assert "Rock ✕" in page and "Psychedelic rock ✕" in page and ">or<" in page
    # Each selected chip's ✕ removes just that genre.
    assert 'href="/music?genre=Psychedelic+rock" title="Remove Rock"' in page
    # Switching to "All" keeps both genres.
    assert 'href="/music?genre=Rock&amp;genre=Psychedelic+rock&amp;match=all"' in page
    page = client.get("/music?genre=Rock&genre=Psychedelic+rock&match=all").get_data(as_text=True)
    assert shown(page) == {"The Piper at the Gates of Dawn"} and ">and<" in page

    # A genre chip on a card adds that genre to the selection; clicking a selected one removes it.
    page = client.get("/music?genre=Rock").get_data(as_text=True)
    assert 'href="/music?genre=Rock&amp;genre=Psychedelic+rock"' in page
    assert re.search(r'href="/music"\s+title="Remove Rock from the selected genres"', page)
    # The dropdown adds another genre and no longer offers ones already selected.
    assert '<option value="Psychedelic rock">' in page and '<option value="Rock">' not in page
    assert '<input type="hidden" name="genre" value="Rock">' in page
    assert "+ Add a genre" in page


def test_genre_selection_survives_search_sort_and_redirect(rock):
    app = create_app(rock.db_path)
    app.testing = True
    client = app.test_client()
    page = client.get("/music?genre=Rock&genre=Jazz&sort=newest").get_data(as_text=True)
    assert page.count('<input type="hidden" name="genre" value="Rock">') == 2   # search form + sort form
    assert page.count('<input type="hidden" name="genre" value="Jazz">') == 2
    resp = client.get("/?type=music&genre=Rock&genre=Jazz")
    assert resp.headers["Location"] == "/music?genre=Rock&genre=Jazz"
    page = client.get("/?genre=Rock&genre=Animation").get_data(as_text=True)    # across all types
    assert {"Abbey Road", "Yellow Submarine"} <= set(re.findall(r'<h3><a href="/item/\d+">([^<]+)</a></h3>', page))
    assert client.get("/api/search?q=the&genre=Rock&genre=Psychedelic+rock&match=all").get_json()["results"][0][
        "title"] == "The Piper at the Gates of Dawn"


def test_cli_several_genres(rock, capsys):
    from encyclomedia.__main__ import main
    main(["--db", rock.db_path, "--no-semantic", "list", "--genre", "Rock", "--genre", "Psychedelic rock"])
    out = capsys.readouterr().out
    assert "Abbey Road" in out and "Revolver" in out and "Kind of Blue" not in out
    main(["--db", rock.db_path, "--no-semantic", "list", "--genre", "Rock", "--genre", "Psychedelic rock",
          "--all-genres"])
    out = capsys.readouterr().out
    assert "Piper" in out and "Abbey Road" not in out


# --------------------------------------------------------------------------- "The" in artist sorting
def test_sort_name_ignores_leading_the():
    from encyclomedia.catalog import sort_name
    assert sort_name("The Beatles") == "beatles"
    assert sort_name("  THE   Kinks ") == "kinks"
    assert sort_name("The The") == "the"                 # the band called The The
    assert sort_name("Theory of a Deadman") == "theory of a deadman"   # only the whole word "The"
    assert sort_name("Thelonious Monk") == "thelonious monk"
    assert sort_name(None) == ""


def test_artists_sort_ignoring_the(tmp_path):
    cat = Catalog(tmp_path / "bands.db")
    for artist in ("The Kinks", "Pink Floyd", "The Beatles", "ABBA", "The Cranberries", "Coldplay",
                   "Thelonious Monk", "Fleetwood Mac", "Kate Bush"):
        cat.add("music", f"Album by {artist}", creator=artist)
    order = [i["creator"] for i in cat.list("music", sort="creator")]
    assert order == ["ABBA", "The Beatles", "Coldplay", "The Cranberries", "Fleetwood Mac",
                     "Kate Bush", "The Kinks", "Pink Floyd", "Thelonious Monk"]
    # The web page's "Sort by Artist" uses the same order, and names still display in full.
    app = create_app(cat.db_path)
    app.testing = True
    page = app.test_client().get("/music?sort=creator").get_data(as_text=True)
    shown = re.findall(r'<h3><a href="/item/\d+">Album by ([^<]+)</a></h3>', page)
    assert shown == order


# --------------------------------------------------------------------------- titles and surnames
def test_surname_first():
    from encyclomedia.catalog import surname_first as s
    assert s("Kate Bush") == "Bush, Kate"
    assert s("Martin Luther King Jr.") == "King, Martin Luther, Jr."
    assert s("Ludwig van Beethoven") == "Beethoven, Ludwig van"     # lowercase particle moves
    assert s("Ursula K. Le Guin") == "Le Guin, Ursula K."           # capitalized one stays
    assert s("Neil Gaiman, Terry Pratchett") == "Gaiman, Neil"      # first credited person decides
    assert s("Madonna") == "Madonna"


def test_titles_ignore_leading_the(tmp_path):
    cat = Catalog(tmp_path / "t.db")
    for title in ("The Shining", "Spirited Away", "Arrival", "The Thing", "Theodora"):
        cat.add("movie", title)
    assert [i["title"] for i in cat.list("movie", sort="title")] == [
        "Arrival", "The Shining", "Spirited Away", "Theodora", "The Thing"]
    assert [i["title"] for i in cat.list("movie")] == [i["title"] for i in cat.list("movie", sort="title")]


def test_people_file_by_surname_bands_as_written(tmp_path):
    cat = Catalog(tmp_path / "s.db")
    for artist in ("Kate Bush", "The Beatles", "Pink Floyd", "Fleetwood Mac", "Miles Davis", "ABBA"):
        cat.add("music", f"Album by {artist}", creator=artist)
    order = lambda: [i["creator"] for i in cat.list("music", sort="creator")]
    # Without settings, artists file as written (bands can't be told from people by name alone).
    assert order() == ["ABBA", "The Beatles", "Fleetwood Mac", "Kate Bush", "Miles Davis", "Pink Floyd"]
    cat.set_file_under("Kate Bush", "Bush, Kate", "music")
    cat.set_file_under("miles  davis", "Davis, Miles", "music")       # matched however it's typed
    assert order() == ["ABBA", "The Beatles", "Kate Bush", "Miles Davis", "Fleetwood Mac", "Pink Floyd"]
    # Authors and directors file by surname automatically.
    for author in ("Stephen Fry", "Frank Herbert", "William Gibson", "Ursula K. Le Guin"):
        cat.add("book", f"Book by {author}", creator=author)
    assert [i["creator"] for i in cat.list("book", sort="creator")] == [
        "Stephen Fry", "William Gibson", "Frank Herbert", "Ursula K. Le Guin"]
    # Setting the automatic value (or blank) removes the setting; MusicBrainz-style "Beatles, The" counts as automatic.
    cat.set_file_under("Kate Bush", "", "music")
    cat.set_file_under("The Beatles", "Beatles, The", "music")
    assert cat.file_under_setting("Kate Bush") is None and cat.file_under_setting("The Beatles") is None
    cat.set_file_under("Frank Herbert", "Herbert, Frank", "book")
    assert cat.file_under_setting("Frank Herbert") is None


def test_web_file_under_and_solo_artist_api(tmp_path):
    app = create_app(tmp_path / "w.db")
    app.testing = True
    client = app.test_client()
    info = client.get("/api/file-under?type=music&name=Kate+Bush").get_json()
    assert info == {"setting": None, "automatic": "Kate Bush", "surname": "Bush, Kate"}
    assert client.get("/api/file-under?type=book&name=Frank+Herbert").get_json()["automatic"] == "Herbert, Frank"

    client.post("/add", data={"media_type": "music", "title": "Hounds of Love", "creator": "Kate Bush",
                              "file_under": "Bush, Kate"})
    client.post("/add", data={"media_type": "music", "title": "Abbey Road", "creator": "The Beatles",
                              "file_under": ""})
    cat = Catalog(tmp_path / "w.db")
    assert cat.file_under_setting("Kate Bush") == "Bush, Kate"
    # Adding another album with the box left blank keeps how the artist is filed.
    client.post("/add", data={"media_type": "music", "title": "The Kick Inside", "creator": "Kate Bush",
                              "file_under": ""})
    assert cat.file_under_setting("Kate Bush") == "Bush, Kate"
    titles = re.findall(r'<h3><a href="/item/\d+">([^<]+)</a></h3>',
                        client.get("/music?sort=creator").get_data(as_text=True))
    assert titles == ["Abbey Road", "Hounds of Love", "The Kick Inside"]   # Beatles, then Bush's albums by title

    hounds = cat.search("hounds")[0]
    edit_page = client.get(f"/item/{hounds['id']}/edit").get_data(as_text=True)
    assert 'name="file_under" value="Bush, Kate"' in edit_page and "Solo artist: use surname" in edit_page
    # Clearing the box while editing the same artist goes back to automatic.
    client.post(f"/item/{hounds['id']}/edit", data={"media_type": "music", "title": "Hounds of Love",
                                                    "creator": "Kate Bush", "file_under": ""})
    assert cat.file_under_setting("Kate Bush") is None


def test_musicbrainz_sort_name_is_used():
    r = {"id": "x", "title": "Hounds of Love", "artist-credit": [
        {"name": "Kate Bush", "joinphrase": "", "artist": {"name": "Kate Bush", "sort-name": "Bush, Kate"}}]}
    assert metadata.parse_musicbrainz_release(r)["creator_sort"] == "Bush, Kate"
    duo = {"id": "y", "title": "Watch the Throne", "artist-credit": [
        {"name": "JAY-Z", "joinphrase": " & ", "artist": {"sort-name": "JAY-Z"}},
        {"name": "Kanye West", "joinphrase": "", "artist": {"sort-name": "West, Kanye"}}]}
    assert metadata.parse_musicbrainz_release(duo)["creator_sort"] is None


def test_cli_file_under(tmp_path, capsys):
    from encyclomedia.__main__ import main
    db = str(tmp_path / "c.db")
    main(["--db", db, "--no-semantic", "add", "music", "Hounds of Love", "--creator", "Kate Bush",
          "--file-under", "Bush, Kate"])
    main(["--db", db, "--no-semantic", "add", "music", "Abbey Road", "--creator", "The Beatles"])
    capsys.readouterr()
    main(["--db", db, "--no-semantic", "list", "--type", "music", "--sort", "creator"])
    out = capsys.readouterr().out
    assert out.index("The Beatles") < out.index("Kate Bush")
    main(["--db", db, "--no-semantic", "edit", "1", "--file-under", ""])
    assert Catalog(db).file_under_setting("Kate Bush") is None


# --------------------------------------------------------------------------- Artists page
@pytest.fixture
def band_cat(tmp_path):
    c = Catalog(tmp_path / "artists.db")
    for title, artist in (("Abbey Road", "The Beatles"), ("Revolver", "The Beatles"),
                          ("Hounds of Love", "Kate Bush"), ("Kind of Blue", "Miles Davis"),
                          ("Rumours", "Fleetwood Mac"), ("Ray of Light", "Madonna"),
                          ("Swimming", "Mac Miller"), ("Dune", "Frank Herbert")):
        c.add("book" if title == "Dune" else "music", title, creator=artist)
    return c


def test_creators_listing(band_cat):
    band_cat.set_file_under("Kate Bush", "Bush, Kate", "music")
    band_cat.set_file_under("Mac Miller", "Miller, Mac (rapper)", "music")
    rows = {a["name"]: a for a in band_cat.creators("music")}
    assert "Frank Herbert" not in rows                                     # only music
    assert rows["The Beatles"]["count"] == 2 and rows["The Beatles"]["kind"] == "band"
    assert rows["Kate Bush"]["kind"] == "solo" and rows["Kate Bush"]["letter"] == "B"
    assert rows["Mac Miller"]["kind"] == "custom" and rows["Mac Miller"]["letter"] == "M"
    assert rows["Madonna"]["one_word"]
    assert [a["name"] for a in band_cat.creators("music")] == [
        "The Beatles", "Kate Bush", "Fleetwood Mac", "Madonna", "Miles Davis", "Mac Miller"]  # Mac Miller is custom-filed


def test_artists_page_marks_several_at_once(band_cat):
    app = create_app(band_cat.db_path)
    app.testing = True
    client = app.test_client()
    page = client.get("/artists").get_data(as_text=True)
    flat = " ".join(page.split())
    assert "<h1>Artists</h1>" in page and "6 artists · 0 filed by surname" in flat
    assert re.search(r'class="tab active" href="/music"', page)              # lives under Music
    names = dict((m[1], int(m[0])) for m in re.findall(r'name="name_(\d+)" value="([^"]+)"', page))
    assert set(names) == {"The Beatles", "Kate Bush", "Fleetwood Mac", "Madonna", "Mac Miller", "Miles Davis"}
    assert re.search(rf'name="kind_{names["Madonna"]}" value="solo"\s+disabled', page)   # no surname to use

    form = {"count": str(len(names))}
    for name, i in names.items():
        form[f"name_{i}"] = name
        form[f"kind_{i}"] = "band"
    form[f"kind_{names['Kate Bush']}"] = "solo"
    form[f"kind_{names['Miles Davis']}"] = "solo"
    form[f"kind_{names['Mac Miller']}"] = "custom"
    form[f"custom_{names['Mac Miller']}"] = "Miller, Mac"
    resp = client.post("/artists", data=form)
    assert resp.headers["Location"] == "/artists"
    page = client.get("/artists").get_data(as_text=True)
    assert "Saved. 3 artists re-filed." in page
    assert "6 artists · 3 filed by surname" in " ".join(page.split())       # "Miller, Mac" is his surname form
    assert band_cat.file_under_setting("Kate Bush") == "Bush, Kate"
    assert band_cat.file_under_setting("Miles Davis") == "Davis, Miles"
    assert band_cat.file_under_setting("Mac Miller") == "Miller, Mac"
    assert band_cat.file_under_setting("The Beatles") is None
    order = [i["creator"] for i in band_cat.list("music", sort="creator")]
    assert order == ["The Beatles", "The Beatles", "Kate Bush", "Miles Davis", "Fleetwood Mac", "Madonna",
                     "Mac Miller"]

    # Switching an artist back to Band removes the setting.
    form[f"kind_{names['Kate Bush']}"] = "band"
    client.post("/artists", data=form)
    assert band_cat.file_under_setting("Kate Bush") is None


def test_artists_links(band_cat, tmp_path):
    app = create_app(band_cat.db_path)
    app.testing = True
    client = app.test_client()
    assert 'href="/artists">Artists</a>' in client.get("/music").get_data(as_text=True)
    assert 'href="/artists">Artists</a>' in client.get("/books").get_data(as_text=True)  # header link
    empty = create_app(tmp_path / "none.db")
    empty.testing = True
    assert "No artists yet." in empty.test_client().get("/artists").get_data(as_text=True)


# --------------------------------------------------------------------------- punctuation and accents
def test_sort_name_skips_leading_punctuation_and_accents():
    from encyclomedia.catalog import sort_name
    assert sort_name('"Experience" Jimi Hendrix - Original Motion Picture Soundtrack').startswith("experience")
    assert sort_name("...Somewhere More Familiar") == "somewhere more familiar"
    assert sort_name("(What's the Story) Morning Glory?").startswith("what's")
    assert sort_name('"The Wall"') == 'wall"'                 # quote, then "The", both skipped
    assert sort_name("¡Viva!") == "viva!" and sort_name("'Til Tuesday") == "til tuesday"
    assert sort_name("Élan") == "elan"                         # accents don't push titles past Z
    assert sort_name("!!!") == "!!!"                           # all punctuation: kept as is


def test_titles_with_leading_punctuation_sort_into_place(tmp_path):
    cat = Catalog(tmp_path / "p.db")
    titles = ['"Experience" Jimi Hendrix - Original Motion Picture Soundtrack', "...Somewhere More Familiar",
              "Abbey Road", "(What's the Story) Morning Glory?", "Élan", "Kind of Blue", "The Wall", "Rumours",
              "Electric Ladyland", "Sound of Silver"]
    for t in titles:
        cat.add("music", t)
    expected = ["Abbey Road", "Élan", "Electric Ladyland",
                '"Experience" Jimi Hendrix - Original Motion Picture Soundtrack', "Kind of Blue", "Rumours",
                "...Somewhere More Familiar", "Sound of Silver", "The Wall", "(What's the Story) Morning Glory?"]
    assert [i["title"] for i in cat.list("music", sort="title")] == expected
    app = create_app(cat.db_path)
    app.testing = True
    page = app.test_client().get("/music").get_data(as_text=True)
    shown = re.findall(r'<h3><a href="/item/\d+">([^<]+)</a></h3>', page)
    assert [html.unescape(t) for t in shown] == expected       # shown exactly as written, in this order
