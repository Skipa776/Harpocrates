"""tomllib on Python 3.11+, the tomli package (same API) on 3.10."""

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

__all__ = ["tomllib"]
