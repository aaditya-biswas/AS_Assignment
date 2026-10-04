"""Root conftest: ensures the project root is importable during tests.

The modules are top-level (``config.py``, ``world.py`` ...), so tests import
them directly (``from world import World``). Having this file at the project
root makes pytest add the root directory to ``sys.path`` even when the tests
live in ``tests/``.
"""
