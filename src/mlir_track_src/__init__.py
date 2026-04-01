from importlib.metadata import PackageNotFoundError, version

try:
    dist_name = "mlir-track-src"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
