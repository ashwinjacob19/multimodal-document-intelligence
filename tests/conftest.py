import asyncio
from collections.abc import Generator

import pytest


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Overrides the default event_loop fixture to have session scope.

    This ensures that all async tests share the same event loop,
    preventing SQLAlchemy engine/connection pool attachment errors.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()
