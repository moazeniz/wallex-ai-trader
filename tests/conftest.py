import pytest


@pytest.fixture(autouse=True)
def _fresh_circuit_breakers():
    """Circuit breakers are process-wide; give every test a clean registry."""
    try:
        from experiments.wallex_arena import resilience
    except ImportError:
        yield
        return
    resilience._breakers.clear()
    yield
    resilience._breakers.clear()
