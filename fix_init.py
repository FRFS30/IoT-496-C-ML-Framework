"""
fix_inits.py
============
Run this once from the project root to strip all eager imports out of every
iotids __init__.py.  After this, `python random_forest.py` will work.

Usage (from project root):
    python fix_inits.py
"""

from pathlib import Path

# ── Root of the iotids Python library ────────────────────────────────────────
IOTIDS_ROOT = Path("python") / "iotids"

# ── Every __init__.py that needs to be made lazy ─────────────────────────────
#    key   = relative path from IOTIDS_ROOT
#    value = the ONLY content the file should contain after the fix
INIT_CONTENTS: dict[str, str] = {

    "__init__.py": '''\
# python/iotids/__init__.py
#
# Kept minimal on purpose.  Eager "from . import ..." lines here cause
# circular imports because Python tries to initialize every subpackage at
# the same time.  Let callers import directly from submodule paths instead:
#
#   from python.iotids.data.csv_reader import read_csv          # correct
#   from python.iotids import data                              # also fine
#   from python.iotids import core, utils, data, ...            # WRONG — circular

__version__ = "0.1.0"
''',

    "core/__init__.py": '''\
# core/__init__.py — lazy, no eager imports
''',

    "data/__init__.py": '''\
# data/__init__.py — lazy, no eager imports
''',

    "nn/__init__.py": '''\
# nn/__init__.py — lazy, no eager imports
''',

    "forest/__init__.py": '''\
# forest/__init__.py — lazy, no eager imports
''',

    "federated/__init__.py": '''\
# federated/__init__.py — lazy, no eager imports
''',

    "prune/__init__.py": '''\
# prune/__init__.py — lazy, no eager imports
''',

    "quantize/__init__.py": '''\
# quantize/__init__.py — lazy, no eager imports
''',

    "metrics/__init__.py": '''\
# metrics/__init__.py — lazy, no eager imports
''',

    "utils/__init__.py": '''\
# utils/__init__.py — lazy, no eager imports
''',
}


def fix():
    if not IOTIDS_ROOT.exists():
        print(f"ERROR: {IOTIDS_ROOT} not found. "
              f"Run this script from the project root "
              f"(the folder that contains the 'python/' directory).")
        return

    fixed = 0
    skipped = 0

    for rel_path, new_content in INIT_CONTENTS.items():
        target = IOTIDS_ROOT / rel_path

        if not target.exists():
            # Create it if missing (e.g. subpackage dir exists but no __init__)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_content, encoding="utf-8")
            print(f"  CREATED  {target}")
            fixed += 1
            continue

        old_content = target.read_text(encoding="utf-8")

        # Only rewrite if the file actually has eager imports
        if "from . import" in old_content or (
            "import" in old_content and "__version__" not in old_content
        ):
            # Preserve any __version__ / __author__ lines that were already there
            preserved = [
                line for line in old_content.splitlines()
                if line.startswith("__version__") or line.startswith("__author__")
            ]
            final = new_content
            if preserved:
                final = final.rstrip("\n") + "\n" + "\n".join(preserved) + "\n"

            target.write_text(final, encoding="utf-8")
            print(f"  FIXED    {target}")
            fixed += 1
        else:
            print(f"  OK       {target}  (no eager imports found, skipped)")
            skipped += 1

    print(f"\nDone. {fixed} file(s) fixed, {skipped} already clean.")
    print("\nYou can now run:")
    print("    python random_forest.py")
    print("    python dnn.py")


if __name__ == "__main__":
    fix()