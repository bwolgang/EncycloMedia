"""
Auto-fill item details from free online databases.

    book   Open Library   by ISBN or title          no key needed
    music  MusicBrainz    by barcode or title       no key needed (1 request/second limit)
    movie  TMDB           by title                  free API key: TMDB_API_KEY
    game   IGDB           by title                  free Twitch app: IGDB_CLIENT_ID + IGDB_CLIENT_SECRET

Every lookup returns a list of candidate dicts shaped like catalog items:
    {"media_type", "title", "creator", "year", "genre", "format", "notes", "extra": {...}, "source"}
so the caller can pick one and pass it straight to Catalog.add().

Fetching and parsing are separate functions so the parsers can be tested offline.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

# MusicBrainz asks clients to identify themselves with an app name and contact.
# Set ENCYCLOMEDIA_CONTACT to an email or URL of your choosing.
_contact = os.environ.get("ENCYCLOMEDIA_CONTACT", "").strip()
USER_AGENT = "EncycloMedia/2.0" + (f" ( {_contact} )" if _contact else "")


class MetadataError(Exception):
    """A lookup could not be completed (network, missing key, bad response)."""


# --------------------------------------------------------------------------- http
def _http(url, params=None, headers=None, data=None, method=None, timeout=12):
    if params:
        url += ("&" if "?" in url else "?") + urlencode(params)
    if isinstance(data, str):
        data = data.encode()
    req = Request(url, data=data, method=method,
                  headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except HTTPError as e:
        raise MetadataError(f"{urlparse(url).netloc} returned HTTP {e.code}") from e
    except URLError as e:
        raise MetadataError(f"Could not reach {urlparse(url).netloc}: {e.reason}") from e
    except (TimeoutError, json.JSONDecodeError) as e:
        raise MetadataError(f"Bad response from {urlparse(url).netloc}: {e}") from e


def _year(text):
    m = re.search(r"\b(\d{4})\b", str(text or ""))
    return int(m.group(1)) if m else None


def _candidate(media_type, title, source, extra=None, **fields):
    return {"media_type": media_type, "title": title, "source": source,
            "creator": fields.get("creator"), "year": fields.get("year"),
            "genre": fields.get("genre"), "format": fields.get("format"),
            "notes": fields.get("notes"),
            "extra": {k: v for k, v in (extra or {}).items() if v not in (None, "", [])}}


def looks_like_code(text: str) -> bool:
    """True for an ISBN-10/13 or a UPC/EAN barcode."""
    digits = re.sub(r"[\s-]", "", text or "")
    return bool(re.fullmatch(r"\d{9}[\dXx]|\d{12,13}", digits))


# --------------------------------------------------------------------------- books
OPENLIBRARY = "https://openlibrary.org"


def parse_openlibrary_book(d: dict, isbn=None, first_published=None) -> dict:
    """An ISBN identifies one edition, so its publish_date is the edition's year.
    `first_published` is the work's original year, looked up separately."""
    subjects = [s["name"] if isinstance(s, dict) else s for s in d.get("subjects", [])]
    publishers = [p["name"] if isinstance(p, dict) else p for p in d.get("publishers", [])]
    edition_year = _year(d.get("publish_date"))
    return _candidate(
        "book", d.get("title"), "Open Library",
        creator=", ".join(a["name"] for a in d.get("authors", [])) or None,
        year=first_published or edition_year,
        genre=", ".join(subjects[:3]) or None,
        extra={"isbn": isbn, "edition_year": edition_year,
               "publisher": publishers[0] if publishers else None,
               "pages": d.get("number_of_pages"),
               "cover_url": (d.get("cover") or {}).get("medium")},
    )


def parse_openlibrary_search(data: dict) -> list[dict]:
    out = []
    for doc in data.get("docs", []):
        isbns = doc.get("isbn") or []
        isbn = next((i for i in isbns if len(i) == 13), isbns[0] if isbns else None)
        out.append(_candidate(
            "book", doc.get("title"), "Open Library",
            creator=", ".join(doc.get("author_name", [])) or None,
            year=doc.get("first_publish_year"),
            genre=", ".join(doc.get("subject", [])[:3]) or None,
            extra={"isbn": isbn,
                   "publisher": (doc.get("publisher") or [None])[0],
                   "pages": doc.get("number_of_pages_median"),
                   "cover_url": f"https://covers.openlibrary.org/b/id/{doc['cover_i']}-M.jpg"
                   if doc.get("cover_i") else None},
        ))
    return out


def books_by_isbn(isbn: str) -> list[dict]:
    isbn = re.sub(r"[^0-9Xx]", "", isbn)
    key = f"ISBN:{isbn}"
    data = _http(f"{OPENLIBRARY}/api/books", {"bibkeys": key, "format": "json", "jscmd": "data"})
    if key not in data:
        return []
    return [parse_openlibrary_book(data[key], isbn, first_published=_first_published(isbn))]


def _first_published(isbn: str) -> int | None:
    """Original publication year of the work this ISBN's edition belongs to."""
    try:
        docs = _http(f"{OPENLIBRARY}/search.json",
                     {"q": f"isbn:{isbn}", "fields": "first_publish_year", "limit": 1}).get("docs", [])
    except MetadataError:
        return None  # not worth failing the whole lookup over
    return docs[0].get("first_publish_year") if docs else None


def books_by_title(query: str, limit=5) -> list[dict]:
    fields = "title,author_name,first_publish_year,isbn,subject,publisher,number_of_pages_median,cover_i"
    return parse_openlibrary_search(
        _http(f"{OPENLIBRARY}/search.json", {"q": query, "limit": limit, "fields": fields}))


# --------------------------------------------------------------------------- music
MUSICBRAINZ = "https://musicbrainz.org/ws/2"
_last_mb_call = 0.0


def _musicbrainz(path, params):
    global _last_mb_call
    wait = 1.05 - (time.monotonic() - _last_mb_call)  # their limit is 1 request/second
    if wait > 0:
        time.sleep(wait)
    _last_mb_call = time.monotonic()
    return _http(f"{MUSICBRAINZ}/{path}", {**params, "fmt": "json"})


def _artist_credit(credits) -> str | None:
    text = "".join(c.get("name", "") + c.get("joinphrase", "") for c in credits or [])
    return text or None


def parse_musicbrainz_release(r: dict) -> dict:
    """A MusicBrainz *release* is one pressing (e.g. a 2019 vinyl reissue); its
    *release group* holds the album's original release date."""
    media = r.get("media") or []
    label_info = r.get("label-info") or []
    tracks = [t.get("title") for m in media for t in m.get("tracks", []) if t.get("title")]
    return _candidate(
        "music", r.get("title"), "MusicBrainz",
        creator=_artist_credit(r.get("artist-credit")),
        year=_year((r.get("release-group") or {}).get("first-release-date")),
        genre=", ".join(g["name"] for g in (r.get("genres") or [])[:3]) or None,
        format=media[0].get("format") if media else None,
        extra={"barcode": r.get("barcode"),
               "edition_year": _year(r.get("date")),
               "label": (label_info[0].get("label") or {}).get("name") if label_info else None,
               "discs": len(media) or None,
               "tracklist": "; ".join(tracks) or None,
               "musicbrainz_id": r.get("id"),
               "cover_url": f"https://coverartarchive.org/release/{r['id']}/front-250" if r.get("id") else None},
    )


def music_search(query="", barcode="", limit=5, with_tracklist=True) -> list[dict]:
    digits = re.sub(r"\D", "", barcode)
    q = f"barcode:{digits}" if barcode else query
    data = _musicbrainz("release/", {"query": q, "limit": limit})
    results = [parse_musicbrainz_release(r) for r in data.get("releases", [])]
    if with_tracklist and results and results[0]["extra"].get("musicbrainz_id"):
        # Search results don't include tracks or genres; look up the best match in full.
        full = _musicbrainz(f"release/{results[0]['extra']['musicbrainz_id']}",
                            {"inc": "recordings+artist-credits+labels+genres+release-groups"})
        results[0] = parse_musicbrainz_release(full)
    return results


# --------------------------------------------------------------------------- movies
TMDB = "https://api.themoviedb.org/3"


def _tmdb(path, params=None):
    key = os.environ.get("TMDB_API_KEY", "").strip()
    if not key:
        raise MetadataError("Movie lookup needs a free TMDB key: set the TMDB_API_KEY environment variable.")
    params, headers = dict(params or {}), {}
    if key.startswith("eyJ"):  # v4 "read access token"
        headers["Authorization"] = f"Bearer {key}"
    else:                      # v3 "API key"
        params["api_key"] = key
    return _http(f"{TMDB}{path}", params, headers)


def parse_tmdb_movie(d: dict) -> dict:
    credits = d.get("credits") or {}
    directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    rating = None
    for country in (d.get("release_dates") or {}).get("results", []):
        if country.get("iso_3166_1") == "US":
            rating = next((r["certification"] for r in country.get("release_dates", []) if r.get("certification")), None)
    return _candidate(
        "movie", d.get("title"), "TMDB",
        creator=", ".join(directors) or None,
        year=_year(d.get("release_date")),
        genre=", ".join(g["name"] for g in d.get("genres", [])) or None,
        notes=d.get("overview") or None,
        extra={"runtime": f"{d['runtime']} min" if d.get("runtime") else None,
               "rating": rating,
               "cast": ", ".join(c["name"] for c in credits.get("cast", [])[:6]) or None,
               "tmdb_id": d.get("id"),
               "cover_url": f"https://image.tmdb.org/t/p/w342{d['poster_path']}" if d.get("poster_path") else None},
    )


def movie_search(query: str, limit=5) -> list[dict]:
    year = None
    m = re.search(r"\((\d{4})\)\s*$", query)  # allow "Dune (2021)"
    if m:
        year, query = m.group(1), query[:m.start()].strip()
    params = {"query": query, **({"year": year} if year else {})}
    hits = _tmdb("/search/movie", params).get("results", [])[:limit]
    return [parse_tmdb_movie(_tmdb(f"/movie/{h['id']}", {"append_to_response": "credits,release_dates"}))
            for h in hits]


# --------------------------------------------------------------------------- games
IGDB = "https://api.igdb.com/v4"
_igdb_token = {"value": None, "expires": 0.0}


def _igdb_auth():
    cid = os.environ.get("IGDB_CLIENT_ID", "").strip()
    secret = os.environ.get("IGDB_CLIENT_SECRET", "").strip()
    if not (cid and secret):
        raise MetadataError("Game lookup needs a free Twitch developer app: set IGDB_CLIENT_ID and IGDB_CLIENT_SECRET.")
    if not _igdb_token["value"] or time.time() > _igdb_token["expires"]:
        tok = _http("https://id.twitch.tv/oauth2/token",
                    {"client_id": cid, "client_secret": secret, "grant_type": "client_credentials"},
                    data=b"", method="POST")
        _igdb_token.update(value=tok["access_token"], expires=time.time() + tok.get("expires_in", 3600) - 60)
    return {"Client-ID": cid, "Authorization": f"Bearer {_igdb_token['value']}", "Content-Type": "text/plain"}


def parse_igdb_game(g: dict) -> dict:
    companies = g.get("involved_companies") or []
    devs = [c["company"]["name"] for c in companies if c.get("developer") and c.get("company")]
    pubs = [c["company"]["name"] for c in companies if c.get("publisher") and c.get("company")]
    ts = g.get("first_release_date")
    cover = (g.get("cover") or {}).get("image_id")
    return _candidate(
        "game", g.get("name"), "IGDB",
        creator=", ".join(devs) or None,
        year=datetime.fromtimestamp(ts, timezone.utc).year if ts else None,
        genre=", ".join(x["name"] for x in g.get("genres", [])) or None,
        notes=g.get("summary") or None,
        extra={"platform": ", ".join(p["name"] for p in g.get("platforms", [])) or None,
               "publisher": ", ".join(pubs) or None,
               "igdb_id": g.get("id"),
               "cover_url": f"https://images.igdb.com/igdb/image/upload/t_cover_big/{cover}.jpg" if cover else None},
    )


def game_search(query: str, limit=5) -> list[dict]:
    body = (f'search "{query.replace(chr(34), "")}"; '
            "fields name,first_release_date,summary,genres.name,platforms.name,cover.image_id,"
            "involved_companies.developer,involved_companies.publisher,involved_companies.company.name; "
            f"limit {int(limit)};")
    data = _http(f"{IGDB}/games", headers=_igdb_auth(), data=body, method="POST")
    return [parse_igdb_game(g) for g in data]


# --------------------------------------------------------------------------- entry point
def lookup(media_type: str, query: str = "", code: str = "", limit: int = 5) -> list[dict]:
    """Look up candidates for an item. `code` is an ISBN or barcode; a query that
    looks like one is treated as one."""
    query, code = (query or "").strip(), (code or "").strip()
    if not code and looks_like_code(query):
        code, query = query, ""
    if not (query or code):
        raise MetadataError("Enter a title, or an ISBN/barcode.")

    if media_type == "book":
        return books_by_isbn(code) if code else books_by_title(query, limit)
    if media_type == "music":
        return music_search(query=query, barcode=code, limit=limit)
    if media_type in ("movie", "game"):
        if code and not query:
            raise MetadataError(f"There's no free barcode database for {media_type}s; search by title instead.")
        return movie_search(query, limit) if media_type == "movie" else game_search(query, limit)
    raise MetadataError(f"Unknown media type {media_type!r}")
