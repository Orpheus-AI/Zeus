"""
Clear Zeus validator cache using the same paths as the validator.
Uses zeus.validator.constants so Path.home() and cache locations stay in sync.
Run from project root with PYTHONPATH set (e.g. by start_validator.sh).
"""
import shutil
import sys

from zeus.validator.constants import ERA5_CACHE_DIR, OLD_METADATA_DATABASE_LOCATION

# ERA5 variable cache subdirs to remove (same names as used by cache layout)
ERA5_SUBDIRS = (
    "2m_dewpoint_temperature",
    "surface_pressure",
    "total_precipitation",
)


def main() -> None:
    print("Clearing Zeus cache (paths from zeus.validator.constants)...")

    for name in ERA5_SUBDIRS:
        path = ERA5_CACHE_DIR / name
        if path.exists():
            shutil.rmtree(path)
            print(f"  removed: {path}")
        else:
            print(f"  (skip, missing): {path}")
  
    if OLD_METADATA_DATABASE_LOCATION.exists():
        OLD_METADATA_DATABASE_LOCATION.unlink()
        print(f"  removed: {OLD_METADATA_DATABASE_LOCATION}")
    else:
        print(f"  (skip, missing): {OLD_METADATA_DATABASE_LOCATION}")

    print("Zeus cache clear done.")
    sys.exit(0)


if __name__ == "__main__":
    main()
