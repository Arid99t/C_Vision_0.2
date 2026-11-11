"""Allow the standalone browser parity tests to run under pytest as well."""
import pytest

from tests.webharness import browser_page


@pytest.fixture(scope="module")
def page():
    with browser_page() as (page, _):
        yield page
