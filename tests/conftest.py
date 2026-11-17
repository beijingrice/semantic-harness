"""Test isolation: offline tests must not depend on the developer's .env or shell.

``app.config`` calls ``load_dotenv()`` at import time, which copies a local
``.env`` into ``os.environ``. Because ``Settings()`` reads the environment on
every construction, removing those variables per test restores the documented
defaults. Tests that need a value pass it explicitly or use ``monkeypatch.setenv``.
"""
import os

import pytest

_CONFIG_PREFIXES = ("LM_",)
_CONFIG_NAMES = {"MAX_KEYWORDS"}


@pytest.fixture(autouse=True)
def _isolate_config_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(_CONFIG_PREFIXES) or name in _CONFIG_NAMES:
            monkeypatch.delenv(name, raising=False)
