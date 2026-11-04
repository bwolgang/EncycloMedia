# EncycloMedia

Catalog your books, music, movies and games, and find anything in it with ranked search.

## Setup

```bash
pip install -r requirements.txt
python -m encyclomedia serve          # then open http://127.0.0.1:5000
```

Your catalog lives in `~/.encyclomedia.db`. Use `--db PATH` to choose another file. If you used the
original single-file script, run with `--db ~/.media_catalog.db` and that database is upgraded in place.

## What's in it

| Module | Does |
|---|---|
| `catalog.py` | SQLite storage, type-specific fields, multi-genre tagging, BM25 keyword search, hybrid search, CSV import/export |
| `semantic.py` | Optional sentence-embedding model for meaning-based search |
| `metadata.py` | Auto-fill from Open Library, MusicBrainz, TMDB, IGDB |
| `web.py` + `templates/` | Flask web interface |
| `__main__.py` | Command line interface |

### Type-specific fields
Every item has title, creator, year, genre, format, tags and notes. Each type also has its own fields,
stored as JSON and included in search:

- **book**: isbn, edition, edition_year, publisher, pages, series
- **music**: barcode, edition_year, label, tracklist, discs
- **movie**: upc, runtime, rating, cast
- **game**: platform, publisher, players, barcode

`year` is always the **original** release: when the book was first published or the album first
came out. `edition_year` is when your particular copy was printed or pressed, for example a 2012
printing of a 1937 novel or a 2019 vinyl reissue. An ISBN or barcode lookup fills in both.

All four also have `cover_url`. To add a field, add its name to `EXTRA_FIELDS` in `catalog.py`.
The form, CSV import/export and search pick it up automatically.

### Item pages
Clicking an item in the catalog opens its **details page**, which shows the cover, credits, every saved field,
the cast or track list, tags and notes. From there you can:
- **Edit details**: opens the edit form at `/item/<id>/edit`. Saving returns you to the catalog.
- **Add another copy**: opens the add form pre-filled with this item's details, for a second edition,
  format or platform. Nothing is saved until you submit the form.
- **View on Open Library / MusicBrainz / TMDB / IGDB**: opens the item on the matching site. Books with an
  ISBN go straight to that edition's page.
- **Delete** (asks you to confirm first).
- **Browse related items**: click a genre, tag, cast member or the creator to search your catalog.
  **More like this** lists related items, based on the same creator, related titles (the Dune novel ↔
  the Dune films), shared genres and tags, and similar subject matter when semantic search is on.

### Cover images
Each item can have a cover, either fetched by auto-fill or uploaded by you. A photo of your own copy works well.
- **Edit form:** use the **Cover** panel. Click **Upload image** or drop a file onto the box, and the preview
  updates immediately. You can also paste an image link, or click **Remove**.
- **Details page:** click **Add a cover image** / **Change cover**, or drop an image straight onto the cover.
- JPG, PNG, WebP and GIF files up to 20 MB are accepted. Files are checked by their content, not their name.
  With Pillow installed, phone photos are turned the right way up and shrunk to at most 1200×1800.
- Uploads are stored in a folder next to your database: `~/.encyclomedia_covers/` for the default catalog, or
  `demo_covers/` for `--db demo.db`. Replaced and removed covers are deleted automatically. A cover shared by
  an item and its copy is kept until neither of them uses it.
- Back up that folder along with your `.db` file. CSV exports record uploaded covers as `/covers/<file>`.

### Genres
An item can have any number of genres. Swimming can be both **Hip-Hop** and **Jazz rap**, for example.

- **Web:** add genres as chips in the item form. Autocomplete suggests genres you've already used.
  Click a genre on any item, or use the genre dropdown, to see only items in that genre.
  The **Genres** page shows counts per media type and lets you rename, merge or remove genres.
- **Spelling variants are one genre.** "Hip-Hop", "hip hop" and "hiphop" all resolve to the first spelling
  you used. To change that spelling everywhere, rename the genre.
- **Merging:** renaming "Rap" to an existing "Hip-Hop" moves every Rap item into Hip-Hop.
- **CSV / CLI:** separate genres with commas, e.g. `--genre "Hip-Hop, Jazz rap"`.
- The genre filter is exact. `--genre "jazz rap"` returns only Jazz rap items, while a keyword search for
  "jazz rap" also matches plain jazz albums and anything mentioning rap.

### Auto-fill
| Type | Source | Look up by | Setup |
|---|---|---|---|
| book | Open Library | ISBN or title | none |
| music | MusicBrainz | barcode or title | optional `ENCYCLOMEDIA_CONTACT` (your email/URL, which MusicBrainz asks for) |
| movie | TMDB | title, optionally `Title (Year)` | free key → `TMDB_API_KEY` (themoviedb.org → Settings → API) |
| game | IGDB | title | free Twitch app → `IGDB_CLIENT_ID`, `IGDB_CLIENT_SECRET` (dev.twitch.tv/console) |

Movies and games can't be looked up by barcode because there's no free UPC database for them.

### Search
- **Keyword:** SQLite FTS5 with BM25 ranking, prefix matching and stemming. Title matches weigh most.
- **Semantic** (needs `sentence-transformers`): each item is embedded with `all-MiniLM-L6-v2`. The model
  downloads about 90 MB on first use, then runs offline. A query like "space opera about politics" can
  then find *Dune*.
- **Hybrid:** `score = 0.5 × keyword + 0.5 × similarity`. Items can surface through either signal,
  and items that match both rank highest. Tune `alpha` and `min_similarity` in `Catalog.search`.
- After installing sentence-transformers, run `python -m encyclomedia reindex` once to embed existing items.
  New and edited items are embedded automatically.

## Command line

```bash
python -m encyclomedia add book 9780441013593 --lookup          # auto-fill from ISBN
python -m encyclomedia add game "Hades" --field platform=Switch --tags "greek mythology"
python -m encyclomedia lookup music "Abbey Road"                 # list matches
python -m encyclomedia lookup music "Abbey Road" --add 1         # add match #1
python -m encyclomedia search "space politics" --type book
python -m encyclomedia search "herbert" --keyword-only
python -m encyclomedia add music "Swimming" --creator "Mac Miller" --genre "Hip-Hop, Jazz rap"
python -m encyclomedia list --type movie
python -m encyclomedia list --genre "jazz rap"
python -m encyclomedia search "mac" --genre hip-hop
python -m encyclomedia genres --type music
python -m encyclomedia genre-rename "Rap" "Hip-Hop"           # merge
python -m encyclomedia genre-remove "Misc"
python -m encyclomedia show 3
python -m encyclomedia edit 3 --format vinyl --field label="Apple Records"
python -m encyclomedia remove 3
python -m encyclomedia import collection.csv
python -m encyclomedia export collection.csv
python -m encyclomedia reindex
python -m encyclomedia serve --port 8000
```

CSV columns: `media_type`, `title`, plus any core or type-specific field (`isbn`, `platform`, …).

## Tests

```bash
python -m pytest -q
```

The tests use a fake embedder and canned API responses, so they run offline.
