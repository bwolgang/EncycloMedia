"""
Cover images uploaded from your computer or phone.

Uploaded files are checked to be real images (by their content, not their name),
turned upright using the photo's EXIF orientation, shrunk to a sensible size,
and saved in a folder next to the catalog database:

    ~/.encyclomedia.db  ->  ~/.encyclomedia_covers/
    demo.db             ->  demo_covers/

An item's cover_url is then "/covers/<file name>", which the web app serves.
Resizing needs Pillow (pip install pillow); without it, images are stored as-is.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

URL_PREFIX = "/covers/"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024        # reject anything bigger than 20 MB
MAX_SIZE = (1200, 1800)                    # longest edges after resizing (px)
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+\.(jpg|png|gif|webp)$")


class CoverError(ValueError):
    """The upload isn't a usable image."""


def covers_dir_for(db_path) -> Path | None:
    """Folder that holds a database's uploaded covers (None for in-memory databases)."""
    if not db_path or str(db_path) == ":memory:":
        return None
    db = Path(db_path).expanduser()
    return db.with_name(f"{db.stem}_covers")


def detect_type(data: bytes) -> str | None:
    """Image type from the file's first bytes: jpg, png, gif, webp, or None."""
    if data[:3] == b"\xff\xd8\xff":
        return "jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def prepare(data: bytes) -> tuple[bytes, str]:
    """Validate an upload and return (image bytes to store, file extension)."""
    if not data:
        raise CoverError("The file was empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise CoverError(f"Images must be under {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    kind = detect_type(data)
    if kind is None:
        raise CoverError("That file isn't a supported image. Use JPG, PNG, WebP or GIF.")
    if kind == "gif":
        return data, kind  # keep animation intact

    try:
        from PIL import Image, ImageOps
    except ImportError:
        return data, kind  # Pillow not installed: store the original

    try:
        with Image.open(io.BytesIO(data)) as img:
            img = ImageOps.exif_transpose(img)  # phone photos: rotate to how they were taken
            img.thumbnail(MAX_SIZE)
            out = io.BytesIO()
            if img.mode in ("RGBA", "LA", "P") and kind in ("png", "webp"):
                img.save(out, "PNG", optimize=True)
                return out.getvalue(), "png"
            img.convert("RGB").save(out, "JPEG", quality=88, optimize=True)
            return out.getvalue(), "jpg"
    except Exception as e:  # corrupt or truncated file
        raise CoverError("That image couldn't be read; it may be damaged.") from e


def filename_from_url(url) -> str | None:
    """'/covers/12-ab34.jpg' -> '12-ab34.jpg' for uploaded covers, else None."""
    if isinstance(url, str) and url.startswith(URL_PREFIX):
        name = url[len(URL_PREFIX):]
        if _SAFE_NAME.match(name):
            return name
    return None
