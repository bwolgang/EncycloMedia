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
| `duplicates.py` | Decides when two items are the same (for the add-form warning and CSV import) |
| `autofill.py` | Fills in books' blank details from their ISBNs (used on CSV import) |
| `web.py` + `templates/` | Flask web interface |
| `__main__.py` | Command line interface |

### Browsing
The tabs along the top switch between **All** and a page for each media type:

| Page | Address |
|---|---|
| All items | `/` |
| Books | `/books` |
| Music | `/music` |
| Movies | `/movies` |
| Games | `/games` |

- Each tab shows how many items it holds.
- A type page searches and filters within that type only. Its genre menu lists only that type's genres,
  and its **Add** button starts a new item of that type.
- **Sort by** title, creator (Author / Artist / Director / Game studio), year (newest or oldest), or recently
  added. Search results are always sorted by best match.
- **Alphabetical order skips what comes before the first real word** in titles and names: a leading "The",
  and leading punctuation such as quotes, periods, brackets and dashes. So The Shining files with the S's,
  "Experience" with the E's, ...Somewhere More Familiar with the S's and (What's the Story) Morning Glory?
  with the W's. Accents are ignored too (Élan files with the E's). Titles and names are still shown exactly
  as written.
- **Sorting by creator:**
  - Authors and directors file by **surname** automatically (Herbert, Frank). Lowercase particles go after
    the given name (Beethoven, Ludwig van), capitalized ones stay with the surname (Le Guin, Ursula K.), and
    "Jr." is set aside.
  - Artists and game studios file as written, because a band's name can't be told from a person's.
  - The **File under** box in the form changes how any creator is filed, and the change applies to all of their
    items. For music, **Solo artist: use surname** fills it in (Kate Bush → Bush, Kate). MusicBrainz
    auto-fill sets it for you. Leave it blank for automatic filing.
  - Within one creator, items are listed oldest first.
- **Artists page** (`/artists`, linked in the header and on the Music page): every artist in your music
  collection, grouped by the letter they're filed under, with an album count. For each artist, choose
  **Band** (as written), **Solo artist** (by surname) or **Custom**, then click **Save** once. Use the
  **Find an artist** box to filter a long list. Unsaved changes are highlighted, and the page warns you before
  you leave without saving.
- **Export CSV** on a type page exports just that type (`/export.csv?type=music`).
- An item's details page highlights its tab, and its breadcrumb and genre chips lead back to its type page.
- Older links such as `/?type=book` redirect to the matching page.

### Duplicates
EncycloMedia notices when you're adding something you already have.

- **What counts as the same item** (same media type, and):
  - the same ISBN or barcode. An ISBN-10 and its ISBN-13 count as the same; or
  - when there's no code to compare, the same title and creator, ignoring capitals, accents, punctuation and a
    leading "The". A blank creator matches any creator.
  - A **different format, edition or game platform** marks a separate copy, not a duplicate: Abbey Road on vinyl
    and on CD, or Hades on Switch and on PC. So does a different ISBN or barcode (another edition).
- **Adding by hand:** a warning appears in the form as you type, with a link to the copy you have. The button
  changes to **Add anyway** if you do want another copy. Without JavaScript, the warning appears when you click
  Add. **Add another copy** on a details page never warns.
- **CSV import:** **Skip items I already have** is ticked by default. Rows matching your catalog, or an earlier
  row of the same file, are skipped and listed afterwards, so importing the same file twice adds nothing.
  Untick it to import everything. Command line: `import --allow-duplicates`.
- The rules are in `duplicates.py` if you want to adjust them.

### Credits
Each media type has its own credit fields. The main credit is shown first:

| Type | Main credit | Other credits |
|---|---|---|
| book | Author | Illustrator (for comics and graphic novels) |
| movie | Director | Composer (of the score) |
| music | Artist | Performers (featured performers), Record label |
| game | Game studio | Game designer |

- Separate several names with commas, e.g. `Bilal, Anna Wise`.
- In the form, these fields sit next to the main credit and change when you switch the media type.
- The details page shows a **Credits** section. Clicking a person lists everything in your catalog that
  credits them. People are also searchable, and a shared composer, performer or designer counts toward
  **More like this**.
- Auto-fill fills in a movie's composer from TMDB, and an album's featured performers ("feat.") and record
  label from MusicBrainz.
- Earlier versions labelled games' main credit "Developer"; those studio names now show as **Game studio**.

### Type-specific fields
Every item has title, creator, year, genre, format, tags and notes. Each type also has its own fields,
stored as JSON and included in search:

- **book**: illustrator, isbn, edition_year, publisher, pages, series
- **music**: performers, label (record label), barcode, edition_year, tracklist, discs
- **movie**: composer, upc, runtime, rating, cast
- **game**: designer, platform, publisher, players, barcode

`year` is always the **original** release: when the book was first published or the album first
came out. `edition_year` is when your particular copy was printed or pressed, for example a 2012
printing of a 1937 novel or a 2019 vinyl reissue. An ISBN or barcode lookup fills in both.

All four also have `cover_url`. To add a field, add its name to `EXTRA_FIELDS` in `catalog.py`.
The form, CSV import/export and search pick it up automatically.

### Your copy: edition, location, origin, condition
Every item, of any type, has four fields describing your physical copy:

| Field | For | Example |
|---|---|---|
| **Edition** | which version you own | First edition, Director's cut, Collector's edition |
| **Location** | where it's stored | Living room shelf A2, Storage box 3 |
| **Origin** | where you got it | Barnes & Noble, eBay, Gift from Chris |
| **Condition notes** | its current state | Spine creased, disc lightly scratched |

- These fields are in the **Your copy** section of the add/edit form. Location and Origin suggest places
  you've used before, so "Shelf A2" stays spelled one way.
- The details page shows them under **Your copy**. Clicking a location or origin lists everything stored
  there or bought there. Browse cards also show each item's location.
- All four fields are included in search ("scratched" finds items with that in their condition notes).
- **Add another copy** leaves these four fields blank, since they describe one particular copy.
- CSV import and export use the columns `edition`, `location`, `origin` and `condition`.

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
- **Several genres at once:** on a browse page, click genre tags on the items (or use the **+ Add a genre**
  menu) to select more than one genre. Click a selected tag again, or its ✕, to remove it.
  - **Any** (the default) shows items that have *at least one* of the selected genres: Rock + Psychedelic rock
    lists every rock album and every psychedelic rock album.
  - **All** shows only items that have *every* selected genre.
  - The address keeps your selection (`/music?genre=Rock&genre=Psychedelic+rock`), so you can bookmark it.
  - Command line: repeat `--genre`, and add `--all-genres` for the "All" behaviour.
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

#### Filling in books from their ISBNs
When you import a CSV, every book with an `isbn` is looked up on Open Library, and its **blank** fields are
filled in: cover, author, first-published year, format, edition, edition year, publisher and page count.
Values from your CSV are always kept. Genres aren't filled in, because Open Library's subject lists are
long and mixed.

- On the web, the import form has a **Fill in books' details from their ISBNs** box, ticked by default.
  On the command line, add `--no-autofill` to skip the lookups.
- Afterwards, a message says how many books were filled in and lists any ISBN Open Library doesn't know.
- ISBNs can have hyphens or spaces. A leading zero that a spreadsheet dropped is restored. ISBNs that Excel
  turned into numbers like `9.78044E+12` have lost their digits, so they're listed as invalid. Format the
  ISBN column as **Text** before typing them in.
- Books are looked up 20 at a time, so a few hundred books take under a minute.
- If you're offline, or for books already in your catalog, use **Fill in from ISBNs** on the Books page
  (`python -m encyclomedia autofill`). It appears when books with an ISBN are missing details.

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
python -m encyclomedia list --type movie --sort newest   # sorts: type, title, creator, newest, oldest, added
python -m encyclomedia list --genre "jazz rap"
python -m encyclomedia add movie "Arrival" --edition "4K Steelbook" --location "Media cabinet" \
    --origin "Best Buy" --condition "Sealed"
python -m encyclomedia edit 3 --location "Storage box 3"
python -m encyclomedia list --location "media cabinet"
python -m encyclomedia add book "Watchmen" --creator "Alan Moore" --illustrator "Dave Gibbons"
python -m encyclomedia add movie "Interstellar" --creator "Christopher Nolan" --composer "Hans Zimmer"
python -m encyclomedia add music "Faces" --creator "Mac Miller" --performers "Rick Ross" --label "REMember"
python -m encyclomedia add game "Stardew Valley" --creator "ConcernedApe" --designer "Eric Barone"
python -m encyclomedia add music "Hounds of Love" --creator "Kate Bush" --file-under "Bush, Kate"
python -m encyclomedia locations                                # or: locations --origins
python -m encyclomedia search "mac" --genre hip-hop
python -m encyclomedia list --genre Rock --genre "Psychedelic rock"           # either genre
python -m encyclomedia list --genre Rock --genre "Psychedelic rock" --all-genres  # both genres
python -m encyclomedia genres --type music
python -m encyclomedia genre-rename "Rap" "Hip-Hop"           # merge
python -m encyclomedia genre-remove "Misc"
python -m encyclomedia show 3
python -m encyclomedia edit 3 --format vinyl --field label="Apple Records"
python -m encyclomedia remove 3
python -m encyclomedia import collection.csv                 # add --no-autofill to skip ISBN lookups
python -m encyclomedia autofill                              # fill in books that have an ISBN
python -m encyclomedia export collection.csv
python -m encyclomedia export books.csv --type book
python -m encyclomedia reindex
python -m encyclomedia serve --port 8000
```

CSV columns: `media_type`, `title`, plus any core or type-specific field (`isbn`, `platform`, …).

## Tests

```bash
python -m pytest -q
```

The tests use a fake embedder and canned API responses, so they run offline.
