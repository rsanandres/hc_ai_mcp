"""Concurrency smoke tests."""

from __future__ import annotations

import asyncio


def test_async_gather_smoke() -> None:
    async def _work(i: int) -> int:
        await asyncio.sleep(0.01)
        return i * 2

    async def _runner():
        return await asyncio.gather(*[_work(i) for i in range(3)])

    results = asyncio.run(_runner())
    assert results == [0, 2, 4]
