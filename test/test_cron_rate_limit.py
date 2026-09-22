"""Fire-time per-agent concurrency enforcement in the cron scheduler.

Covers the two pieces added for the ``cron_rate_limit.max_concurrent_per_agent``
knob (FEAT-003):

- :func:`cron_job_agent_key` — the effective agent identity a job dispatches,
  reusing :func:`agent_sequence_dispatches` so keying tracks real dispatch, and
  returning ``None`` (exempt) for script / command / agentless jobs.
- :meth:`CronService._rate_limit_deferred` — the stateless partition of a due
  list into ``(admitted, deferred)`` by per-agent concurrency, mirroring the
  critical-posture admission deferral (a deferred job is not fired and not
  mutated, so it is simply due again next tick).

These are tested directly for determinism; a final test drives ``_on_timer``
end-to-end and asserts only the admitted job's task is created.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kiro_crew.cron import (
    CronJob,
    CronSchedule,
    CronService,
    cron_job_agent_key,
)


def _job(
    jid: str,
    *,
    agent_id: str = "",
    agent_sequence: list[str] | None = None,
    script: str = "",
    command: str = "",
) -> CronJob:
    """A minimal every-60s CronJob for keying / partition tests."""
    return CronJob(
        id=jid,
        name=jid,
        message="m",
        schedule=CronSchedule(kind="every", every_secs=60),
        agent_id=agent_id,
        agent_sequence=list(agent_sequence or []),
        script=script,
        command=command,
    )


def _svc(tmp_path: Path, limit: int) -> CronService:
    """A CronService with the per-agent limit set, no timer armed."""
    svc = CronService(base_dir=tmp_path)
    svc._load()
    svc._max_concurrent_per_agent = limit
    return svc


# ── cron_job_agent_key (case 1) ──


class TestAgentKey:
    def test_plain_agent_job_keys_on_agent_id(self) -> None:
        assert cron_job_agent_key(_job("a", agent_id="alice")) == "alice"

    def test_multi_agent_sequence_keys_on_primary(self) -> None:
        # A sequence of >1 dispatches and takes precedence over agent_id.
        job = _job("a", agent_id="alice", agent_sequence=["bob", "carol"])
        assert cron_job_agent_key(job) == "bob"

    def test_single_element_sequence_falls_through_to_agent_id(self) -> None:
        # A one-element sequence is dormant; dispatch falls through to agent_id.
        job = _job("a", agent_id="alice", agent_sequence=["bob"])
        assert cron_job_agent_key(job) == "alice"

    def test_script_job_is_exempt(self) -> None:
        assert cron_job_agent_key(_job("a", agent_id="alice", script="x.py:f")) is None

    def test_command_job_is_exempt(self) -> None:
        assert cron_job_agent_key(_job("a", agent_id="alice", command="echo hi")) is None

    def test_agentless_job_is_exempt(self) -> None:
        assert cron_job_agent_key(_job("a")) is None


# ── _rate_limit_deferred (cases 2-7) ──


class TestRateLimitDeferred:
    def test_same_agent_over_limit_defers_extra(self, tmp_path: Path) -> None:
        # limit=1, two due jobs for the SAME agent: one admitted, one deferred.
        svc = _svc(tmp_path, limit=1)
        j1 = _job("j1", agent_id="alice")
        j2 = _job("j2", agent_id="alice")
        admitted, deferred = svc._rate_limit_deferred([j1, j2])
        assert admitted == [j1]
        assert deferred == [j2]

    def test_different_agents_limited_independently(self, tmp_path: Path) -> None:
        # limit=1, two due jobs for DIFFERENT agents: both admitted.
        svc = _svc(tmp_path, limit=1)
        j1 = _job("j1", agent_id="alice")
        j2 = _job("j2", agent_id="bob")
        admitted, deferred = svc._rate_limit_deferred([j1, j2])
        assert admitted == [j1, j2]
        assert deferred == []

    def test_already_executing_run_counts_against_limit(self, tmp_path: Path) -> None:
        # A run already in flight for an agent occupies the limit, so a due job
        # for that same agent is deferred.
        svc = _svc(tmp_path, limit=1)
        running = _job("running", agent_id="alice")
        due = _job("due", agent_id="alice")
        svc._jobs = [running, due]
        svc._executing = {"running"}
        admitted, deferred = svc._rate_limit_deferred([due])
        assert admitted == []
        assert deferred == [due]

    def test_script_and_command_never_deferred(self, tmp_path: Path) -> None:
        # Even past the limit, script/command jobs (key None) always admit.
        svc = _svc(tmp_path, limit=1)
        occupant = _job("occupant", agent_id="alice")
        svc._jobs = [occupant]
        svc._executing = {"occupant"}
        s = _job("s", script="x.py:f")
        c = _job("c", command="echo hi")
        admitted, deferred = svc._rate_limit_deferred([s, c])
        assert admitted == [s, c]
        assert deferred == []

    def test_limit_zero_admits_everything(self, tmp_path: Path) -> None:
        # 0 = unlimited: short-circuit, nothing deferred.
        svc = _svc(tmp_path, limit=0)
        jobs = [_job(f"j{i}", agent_id="alice") for i in range(5)]
        admitted, deferred = svc._rate_limit_deferred(jobs)
        assert admitted == jobs
        assert deferred == []

    def test_deferral_is_stateless(self, tmp_path: Path) -> None:
        # A deferred job is not fired and not mutated: last_run_ts,
        # run_never_started, fire_time_denied unchanged; not in _executing.
        svc = _svc(tmp_path, limit=1)
        j1 = _job("j1", agent_id="alice")
        j2 = _job("j2", agent_id="alice")
        admitted, deferred = svc._rate_limit_deferred([j1, j2])
        assert deferred == [j2]
        assert j2.last_run_ts is None
        assert j2.run_never_started is False
        assert j2.fire_time_denied is False
        assert "j2" not in svc._executing


# ── _on_timer end-to-end (only the admitted job fires) ──


class TestOnTimerRateLimit:
    def test_on_timer_fires_only_admitted_job(self, tmp_path: Path, monkeypatch) -> None:
        svc = _svc(tmp_path, limit=1)
        # Two same-agent jobs. Backdate last_run_ts so the every=60s interval
        # has elapsed and both are due on this tick.
        j1 = svc.add_job(name="j1", message="m", every_secs=60, agent_id="alice")
        j2 = svc.add_job(name="j2", message="m", every_secs=60, agent_id="alice")

        # Force both jobs due (the every=60s interval math depends on
        # persisted last_run_ts / created_ts; pin the verdict deterministically
        # so this test exercises only the rate-limit partition, not scheduling).
        monkeypatch.setattr(svc, "_is_due", lambda job, now: True)

        # Neutralize the actual run so no gateway/agent machinery is exercised.
        async def _noop_run(job: CronJob) -> None:
            return None

        monkeypatch.setattr(svc, "_run_job_isolated", _noop_run)

        async def _drive() -> None:
            await svc._on_timer()

        asyncio.run(_drive())

        # Exactly one of the two same-agent jobs fired this tick.
        fired = set(svc._running_tasks)
        assert len(fired) == 1
        assert fired <= {j1.id, j2.id}
        # The deferred job stayed due: not executing, last_run_ts untouched.
        deferred_id = ({j1.id, j2.id} - fired).pop()
        assert deferred_id not in svc._executing
        deferred_job = {j.id: j for j in svc._jobs}[deferred_id]
        assert deferred_job.last_run_ts is None
