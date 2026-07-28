# Makes `tests` a package so the smoke tests run as `python -m tests.<name>` from the
# repo root (which puts the root on sys.path, so `from qem import ...` resolves with no
# path hacks). See README "Running the smoke tests".
