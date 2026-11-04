"""Run with:  python -m pytest -q"""
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
    assert calls[1][1]["q"] == "isbn:9780547928227"

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
    resp = client.post(item_url, data={"media_type": "game", "title": "Celeste", "extra__platform": "Steam Deck"})
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
    assert "Swimming" in page and "Kind of Blue" not in page and "Jazz rap (2)" in page
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
