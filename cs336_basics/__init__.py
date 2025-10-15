import importlib.metadata

try:
    from importlib.metadata import version, PackageNotFoundError  # py>=3.8
except Exception:
    from importlib_metadata import version, PackageNotFoundError  # backport

try:
    __version__ = version("cs336_basics")
except PackageNotFoundError:
    __version__ = "0+local"

