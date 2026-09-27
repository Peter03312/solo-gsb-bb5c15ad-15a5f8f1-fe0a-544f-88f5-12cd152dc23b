"""End-to-end tests for per-target maintenance pauses."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from app import db
from tests.test_webhooks import _get_event, _post_event_obj
from .conftest import wait_for


def _put_pause(base_url: str, target: str, paused: bool) -> httpx.Response:
    return httpx.put(
        f"{base_url}/targets/pause",
        json={"target": target, "paused": paused},
    )


def _get_pauses(base_url: str, **params) -> dict:
    return httpx.get(f"{base_url}/targets/pauses", params=params).json()


def _wait_event(base_url: str, event_id: str, status: str, timeout: float = 15.0) -> dict:
    return wait_for(
        lambda: (lambda e: e if e["status"] == status else None)(
            _get_event(base_url, event_id)
        ),
        timeout=timeout,
    )


def _wait_breaker(target: str, state: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        rows = db.list_breakers(target=target)
        last = rows[0] if rows else None
        if last is not None and last["state"] == state:
            return last
        time.sleep(0.05)
    raise AssertionError(f"breaker for {target} never reached {state}; last={last!r}")


# -------------------------------------------------- API shape and normalisation

def test_pause_api_normalises_idempotent_and_lists(server):
    # A target never configured has no row and behaves as unpaused.
    target = f"{server.base_url}/t/record?pause-shape={uuid.uuid4().hex}"
    assert _get_pauses(server.base_url, target=target)["pauses"] == []

    # Uppercase scheme/host is normalised exactly as event ingest normalises
    # targets (str(HttpUrl)), so this matches stored targets byte-for-byte.
    upper = target.replace("http://", "HTTP://").replace("127.0.0.1", "127.0.0.1")
    resp = _put_pause(server.base_url, upper, True)
    assert resp.status_code == 200
    row = resp.json()
    assert set(row) == {"target", "paused", "updated_at"}
    assert row["target"] == target  # normalised form
    assert row["paused"] is True
    datetime.fromisoformat(row["updated_at"])  # parseable RFC 3339 stamp

    # Listing the same normalised target (exact match) finds the one row.
    listed = _get_pauses(server.base_url, target=target)["pauses"]
    assert [r["target"] for r in listed] == [target]

    # Repeating the same setting is idempotent: updated_at does not move.
    time.sleep(0.01)
    again = _put_pause(server.base_url, target, True).json()
    assert again == row

    # Filtering by paused state works, and resuming flips only the flag.
    assert len(_get_pauses(server.base_url, paused=True)["pauses"]) >= 1
    resumed = _put_pause(server.base_url, target, False).json()
    assert resumed["paused"] is False
    assert resumed["target"] == target
    assert resumed["updated_at"] != row["updated_at"]
    assert _get_pauses(server.base_url, target=target, paused=True)["pauses"] == []
    unpaused = _get_pauses(server.base_url, target=target, paused=False)["pauses"]
    assert [r["target"] for r in unpaused] == [target]

    # List envelope carries the paging fields; invalid input is a 422.
    body = _get_pauses(server.base_url)
    assert set(body) == {"pauses", "limit", "offset"}
    assert httpx.put(
        f"{server.base_url}/targets/pause", json={"target": target}
    ).status_code == 422
    assert httpx.put(
        f"{server.base_url}/targets/pause",
        json={"target": target, "paused": "maybe"},
    ).status_code == 422
    assert httpx.put(
        f"{server.base_url}/targets/pause",
        json={"target": "not-a-url", "paused": True},
    ).status_code == 422


# --------------------------------------------- paused target frozen, others run

def test_paused_target_not_claimed_others_deliver_and_resume_drains(server):
    base = server.base_url
    target = f"{base}/t/record?paused-drain={uuid.uuid4().hex}"
    other = f"{base}/t/record?other={uuid.uuid4().hex}"
    try:
        assert _put_pause(base, target, True).status_code == 200

        first = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        second = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        healthy = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": other, "payload": {},
        }).json()

        # Other targets are completely unaffected.
        _wait_event(base, healthy["id"], "delivered")
        # The paused target's events sit untouched: queued, zero attempts,
        # even though their delivery time has long passed.
        time.sleep(0.6)
        for event_id in (first["id"], second["id"]):
            event = _get_event(base, event_id)
            assert event["status"] == "queued"
            assert event["attempts_count"] == 0

        # Resume: the queue drains in strict head order on the original
        # schedule (nothing was reset).
        assert _put_pause(base, target, False).json()["paused"] is False
        f1 = _wait_event(base, first["id"], "delivered")
        f2 = _wait_event(base, second["id"], "delivered")
        assert f1["attempts_count"] == 1 and f2["attempts_count"] == 1
        assert f1["delivered_at"] <= f2["delivered_at"]
    finally:
        _put_pause(base, target, False)


# --------------------------------------------- in-flight attempt not interrupted

def test_inflight_request_completes_while_target_is_paused(server):
    from .conftest import _gates
    token = f"pause-inflight-{uuid.uuid4().hex}"
    _gates[token] = threading.Event()
    target = f"{server.base_url}/t/block?token={token}"
    try:
        head = _post_event_obj(server.base_url, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        _wait_event(server.base_url, head["id"], "processing")

        # Pause (twice, exercising idempotency) after the claim: the request
        # already in flight must still finish normally.
        assert _put_pause(server.base_url, target, True).status_code == 200
        _put_pause(server.base_url, target, True)

        tail = _post_event_obj(server.base_url, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        time.sleep(0.5)
        assert _get_event(server.base_url, tail["id"])["status"] == "queued"
        assert _get_event(server.base_url, tail["id"])["attempts_count"] == 0

        # Release the in-flight request: it completes as delivered even while
        # the target stays paused.
        httpx.post(f"{server.base_url}/t/release", params={"token": token})
        _wait_event(server.base_url, head["id"], "delivered")

        # The tail remains frozen until resume.
        time.sleep(0.3)
        assert _get_event(server.base_url, tail["id"])["status"] == "queued"
        _put_pause(server.base_url, target, False)
        _wait_event(server.base_url, tail["id"], "delivered")
    finally:
        httpx.post(f"{server.base_url}/t/release", params={"token": token})
        _put_pause(server.base_url, target, False)


# --------------------------------- backoff / attempts not reset by pause/resume

def test_pause_and_resume_do_not_reset_attempt_counter(server):
    base = server.base_url
    # Keeps failing: the event works through 1/2/4s backoff and dies after
    # the 4th failed attempt. Pausing in the middle must not restart that
    # count.
    target = f"{base}/t/probe?fail_times=10&noreset={uuid.uuid4().hex}"
    event = _post_event_obj(base, {
        "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
    }).json()
    event_id = event["id"]
    try:
        # Let at least two attempts land, then pause while retries continue
        # to be scheduled behind the gate.
        wait_for(lambda: _get_event(base, event_id)["attempts_count"] >= 2)
        _put_pause(base, target, True)
        time.sleep(2.5)  # longer than the remaining 2s/4s backoff slots
        paused_event = _get_event(base, event_id)
        assert paused_event["attempts_count"] < 4
        attempts_while_paused = paused_event["attempts_count"]

        # Resume: the existing round continues from the same attempt numbers;
        # the event still dies on exactly the 4th global attempt — no reset
        # ever granted it a fresh round.
        _put_pause(base, target, False)
        dead = _wait_event(base, event_id, "dead", timeout=20)
        assert dead["attempts_count"] == 4
        assert [a["attempt_no"] for a in dead["attempts"]] == [1, 2, 3, 4]
        assert all(a["status_code"] == 500 for a in dead["attempts"])
        assert attempts_while_paused <= 4
    finally:
        _put_pause(base, target, False)


# ------------------------------------------- breaker state untouched by the switch

def test_pause_preserves_breaker_state_and_blocks_the_probe(server):
    base = server.base_url
    gate = f"pause-brk-{uuid.uuid4().hex}"
    # 3 fast failures open the breaker; the next call would block on the gate
    # as the half-open probe.
    target = f"{base}/t/probe?fail_times=3&gate={gate}"
    try:
        head = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        wait_for(lambda: _get_event(base, head["id"])["attempts_count"] == 3, timeout=10)
        breaker = _wait_breaker(target, "open")

        # Pause keeps every breaker field exactly as it was: state, counter,
        # deadline — nothing is reset or advanced.
        _put_pause(base, target, True)
        # Wait past BOTH the 1s open window (test config) and the head's 4s
        # backoff: with the target unpaused the half-open probe would be
        # claimable now, but the pause must suppress it completely.
        time.sleep(5.0)
        still_open = db.list_breakers(target=target)[0]
        assert still_open["state"] == "open"
        assert still_open["consecutive_failures"] == 3
        assert still_open["open_until"] == breaker["open_until"]
        assert still_open["probe_event_id"] is None
        head_event = _get_event(base, head["id"])
        assert head_event["attempts_count"] == 3
        assert head_event["status"] == "queued"

        # Resume: the original head immediately becomes the single half-open
        # probe and blocks on the gate — the breaker resumes its own
        # state machine where it left off.
        _put_pause(base, target, False)
        half = _wait_breaker(target, "half-open", timeout=5)
        assert half["probe_event_id"] == head["id"]

        httpx.post(f"{base}/t/probe-release", params={"gate": gate})
        closed = _wait_breaker(target, "closed", timeout=10)
        assert closed["consecutive_failures"] == 0
        _wait_event(base, head["id"], "delivered")
    finally:
        httpx.post(f"{base}/t/probe-release", params={"gate": gate})
        _put_pause(base, target, False)


# ------------------------------------ queued expiry sweep continues while paused

def test_expired_queued_events_are_swept_while_paused(server):
    base = server.base_url
    target = f"{base}/t/record?pause-expiry={uuid.uuid4().hex}"
    try:
        _put_pause(base, target, True)
        head = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex,
            "target": target,
            "payload": {},
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        }).json()
        tail = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()

        # The head expires with zero attempts even though nothing is being
        # delivered to the paused target.
        expired = _wait_event(base, head["id"], "expired", timeout=10)
        assert expired["attempts_count"] == 0
        assert expired["attempts"] == []
        # The tail becomes the head but stays frozen by the pause.
        time.sleep(0.3)
        assert _get_event(base, tail["id"])["status"] == "queued"
        assert _get_event(base, tail["id"])["attempts_count"] == 0

        _put_pause(base, target, False)
        delivered = _wait_event(base, tail["id"], "delivered")
        assert delivered["attempts_count"] == 1
    finally:
        _put_pause(base, target, False)


# ------------------------------ pause write and claim are serialised by the DB

def test_claim_and_pause_are_serialized_in_database_transactions(server):
    # Ten *different* targets, whose events are independent heads and so are
    # all claimable in one tick (the pool only parallelises distinct
    # endpoints). Pausing one of them once the claim critical section has
    # started cannot roll that claim back.
    targets = [
        f"{server.base_url}/dev/sink?serial={uuid.uuid4().hex}" for _ in range(10)
    ]
    paused_target = targets[0]
    committed = threading.Event()

    def pause():
        db.set_target_pause(paused_target, True)
        committed.set()

    ids: list[str] = []
    with db.db_lock:
        conn = db.get_conn()
        now = time.time()
        for target in targets:
            event_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO events (id, idempotency_key, target, payload, status, "
                "attempts_count, created_at, next_attempt_at, deliver_after) "
                "VALUES (?, ?, ?, '{}', 'queued', 0, 'x', ?, 'x')",
                (event_id, uuid.uuid4().hex, target, now - 1),
            )
            ids.append(event_id)
        conn.commit()

        writer = threading.Thread(target=pause)
        writer.start()
        assert not committed.wait(0.3), "pause commit must wait for the claim critical section"

        # Same-thread re-entry into the RLock: this claim is part of one
        # serialised sequence with the pending pause write, and commits first.
        claimed = db.claim_due_events()
        assert {c["id"] for c in claimed} == set(ids)

    assert committed.wait(2.0)
    writer.join(timeout=2)

    # The live worker never submitted our hand-claimed rows to its pool, so
    # finalise them directly (they would otherwise stay 'processing' and, by
    # the head-ordering rule, block the target behind them).
    with db.db_lock:
        conn = db.get_conn()
        conn.executemany(
            "UPDATE events SET status = 'delivered', next_attempt_at = NULL "
            "WHERE id = ?",
            [(event_id,) for event_id in ids],
        )
        conn.commit()

    # After the pause committed, nothing queued for the paused target can be
    # claimed — including a brand-new due event — while another target goes.
    with db.db_lock:
        extra_id = uuid.uuid4().hex
        db.get_conn().execute(
            "INSERT INTO events (id, idempotency_key, target, payload, status, "
            "attempts_count, created_at, next_attempt_at, deliver_after) "
            "VALUES (?, ?, ?, '{}', 'queued', 0, 'x', ?, 'x')",
            (extra_id, uuid.uuid4().hex, paused_target, time.time() - 1),
        )
        db.get_conn().commit()
        assert db.claim_due_events() == []

    # Resuming clears only the gate; the event claims immediately.
    db.set_target_pause(paused_target, False)
    with db.db_lock:
        resumed_claims = db.claim_due_events()
        assert [c["id"] for c in resumed_claims] == [extra_id]

    # Tidy up: finalise the extra row too.
    with db.db_lock:
        conn = db.get_conn()
        conn.execute(
            "UPDATE events SET status = 'delivered', next_attempt_at = NULL "
            "WHERE id = ?",
            (extra_id,),
        )
        conn.commit()


# -------------------------------------- old databases auto-init; state survives

def test_pause_table_auto_initialises_on_old_database_and_survives_restart():
    db_path = tempfile.mktemp(prefix="webhook-pause-restart-", suffix=".db")
    try:
        # A "legacy" database with only the original events table, no
        # target_pauses table at all.
        legacy = sqlite3.connect(db_path)
        legacy.execute(
            "CREATE TABLE events (id TEXT PRIMARY KEY, idempotency_key TEXT, "
            "target TEXT, payload TEXT, status TEXT, attempts_count INTEGER, "
            "created_at TEXT, next_attempt_at REAL, delivered_at TEXT)"
        )
        legacy.commit()
        legacy.close()

        common_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "WEBHOOK_SECRET": "x",
            "WEBHOOK_DB_PATH": db_path,
        }
        first = subprocess.run(
            [sys.executable, "-c",
             "import json; from app import db; db.init_db(); "
             "print(json.dumps(db.list_target_pauses())); "
             "db.set_target_pause('http://example.com/legacy', True); "
             "print(json.dumps(db.list_target_pauses(target='http://example.com/legacy')))"],
            cwd="/workspace", env=common_env, capture_output=True, text=True,
        )
        assert first.returncode == 0, first.stderr
        lines = [json.loads(line) for line in first.stdout.splitlines()]
        assert lines[0] == []  # table was created empty on the old database
        assert len(lines[1]) == 1
        assert lines[1][0]["target"] == "http://example.com/legacy"
        assert lines[1][0]["paused"] is True
        assert lines[1][0]["updated_at"]

        # A fresh process ("restart") re-runs init on the same file and sees
        # the persisted pause; init_db() stays idempotent.
        second = subprocess.run(
            [sys.executable, "-c",
             "import json; from app import db; db.init_db(); "
             "print(json.dumps(db.list_target_pauses()))"],
            cwd="/workspace", env=common_env, capture_output=True, text=True,
        )
        assert second.returncode == 0, second.stderr
        rows = json.loads(second.stdout.strip())
        assert len(rows) == 1
        assert rows[0]["target"] == "http://example.com/legacy"
        assert rows[0]["paused"] is True
        assert rows[0]["updated_at"] == lines[1][0]["updated_at"]
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except OSError:
                pass
