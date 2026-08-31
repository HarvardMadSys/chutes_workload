from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .models import SessionId


@dataclass(frozen=True)
class CacheLookupResult:
    hit: bool
    reusable_tokens: int


@dataclass(frozen=True)
class CacheUpdateResult:
    evicted_entries: int
    evicted_tokens: int


class KVCache:
    """Per-instance KV cache keyed by session ID with token-based LRU eviction.

    capacity_tokens=None means infinite capacity (no eviction).

    Maintains two parallel records:
      - ``_entries`` (live, LRU-evicted): drives cache hits and eviction.
      - ``_ever_entries`` (shadow, never evicted): max ``it+ot`` ever stored
        for each session on this instance. Used by metrics so the
        replication ratio under finite cache reflects what an infinite
        cache would have held — i.e., the true spread the policy created,
        not what's left after the LRU evicted the evidence.
    """

    def __init__(self, capacity_tokens: int | None) -> None:
        self.capacity_tokens: int | None = capacity_tokens
        self._entries: OrderedDict[SessionId, int] = OrderedDict()
        self._ever_entries: dict[SessionId, int] = {}
        self._current_size_tokens: int = 0
        self._total_evicted_entries: int = 0
        self._total_evicted_tokens: int = 0

    @property
    def current_size_tokens(self) -> int:
        return self._current_size_tokens

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def total_evicted_tokens(self) -> int:
        return self._total_evicted_tokens

    @property
    def total_evicted_entries(self) -> int:
        return self._total_evicted_entries

    def contains(self, session_id: SessionId) -> bool:
        return session_id in self._entries

    def size_of(self, session_id: SessionId) -> int:
        return self._entries.get(session_id, 0)

    def lookup(self, session_id: SessionId, input_tokens: int) -> CacheLookupResult:
        """Compute reusable token count without mutating state."""
        cached_size = self._entries.get(session_id)
        if cached_size is None:
            return CacheLookupResult(hit=False, reusable_tokens=0)
        reusable = min(cached_size, input_tokens)
        return CacheLookupResult(hit=reusable > 0, reusable_tokens=reusable)

    def update(self, session_id: SessionId, input_tokens: int) -> CacheUpdateResult:
        """Insert/refresh the session entry and evict LRU entries if over capacity."""
        prev_ever = self._ever_entries.get(session_id, 0)
        if input_tokens > prev_ever:
            self._ever_entries[session_id] = input_tokens

        existing = self._entries.get(session_id)
        new_size = input_tokens if existing is None else max(existing, input_tokens)

        if existing is not None:
            self._current_size_tokens -= existing
            del self._entries[session_id]

        self._entries[session_id] = new_size
        self._current_size_tokens += new_size

        evicted_entries = 0
        evicted_tokens = 0
        if self.capacity_tokens is not None:
            while self._current_size_tokens > self.capacity_tokens and self._entries:
                victim_id, victim_size = self._entries.popitem(last=False)
                if victim_id == session_id:
                    # We just inserted this entry; if it alone exceeds capacity,
                    # keep it but stop evicting — the request still ran on this instance.
                    self._entries[session_id] = victim_size
                    self._entries.move_to_end(session_id)
                    break
                self._current_size_tokens -= victim_size
                evicted_entries += 1
                evicted_tokens += victim_size

        self._total_evicted_entries += evicted_entries
        self._total_evicted_tokens += evicted_tokens
        return CacheUpdateResult(evicted_entries=evicted_entries, evicted_tokens=evicted_tokens)

    def snapshot_sessions(self) -> dict[SessionId, int]:
        """Live cache contents (post-eviction). Drives cache hits."""
        return dict(self._entries)

    def ever_snapshot_sessions(self) -> dict[SessionId, int]:
        """Max ``it+ot`` ever stored per session on this instance.

        Equals ``snapshot_sessions()`` under infinite cache; preserves
        evicted entries under finite cache so replication metrics reflect
        the true spread the policy created.
        """
        return dict(self._ever_entries)
