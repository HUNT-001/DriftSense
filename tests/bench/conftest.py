import pytest

pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def _no_imds(monkeypatch):
    """Never touch the EC2 metadata endpoint from tests (unless a test opts back in)."""
    monkeypatch.setenv("BENCH_NO_IMDS", "1")
    monkeypatch.setenv("ENGINE_GIT_SHA", "test-sha")
    from bench.cool import common as C

    C._IMDS_CACHE.clear()
    C._IMDS_TOKEN.clear()
    yield
    C._IMDS_CACHE.clear()
    C._IMDS_TOKEN.clear()
