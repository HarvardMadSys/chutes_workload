from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .models import Request, SessionId


@dataclass
class SessionState:
    last_instance_id: int | None = None
    last_request_time: datetime | None = None
    instances_touched: set[int] = field(default_factory=set)
    request_count_per_instance: dict[int, int] = field(default_factory=dict)
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    simulated_cached_tokens: int = 0
    cache_hits: int = 0


class SimulationState:
    def __init__(self) -> None:
        self.sessions: dict[SessionId, SessionState] = {}

    def get_or_create(self, session_id: SessionId) -> SessionState:
        existing = self.sessions.get(session_id)
        if existing is not None:
            return existing
        created = SessionState()
        self.sessions[session_id] = created
        return created

    def update_after_request(
        self,
        request: Request,
        instance_id: int,
        simulated_cached_tokens: int,
        cache_hit: bool,
    ) -> None:
        session = self.get_or_create(request.session_id)
        session.last_instance_id = instance_id
        session.last_request_time = request.started_at
        session.instances_touched.add(instance_id)
        session.request_count_per_instance[instance_id] = (
            session.request_count_per_instance.get(instance_id, 0) + 1
        )
        session.request_count += 1
        session.input_tokens += request.input_tokens
        session.output_tokens += request.output_tokens
        session.simulated_cached_tokens += simulated_cached_tokens
        if cache_hit:
            session.cache_hits += 1
