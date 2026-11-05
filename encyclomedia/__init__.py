"""EncycloMedia: catalog books, music, movies and games with smart search."""
from .catalog import COPY_FIELDS, CORE_FIELDS, EXTRA_FIELDS, MEDIA_TYPES, Catalog

__all__ = ["Catalog", "MEDIA_TYPES", "CORE_FIELDS", "COPY_FIELDS", "EXTRA_FIELDS"]
__version__ = "2.0.0"
