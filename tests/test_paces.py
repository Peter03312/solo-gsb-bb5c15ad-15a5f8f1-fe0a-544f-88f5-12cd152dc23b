"""End-to-end tests for per-target delivery pacing (rate limiting)."""
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


def _put_pace(base_url: str, target: str, interval_seconds: float) -> httpx.Response:
    return httpx.put(
        f"{base_url}/targets/pace",
        json={"target": target, "interval_seconds": interval_seconds},
    )


def _get_paces(base_url: str, **params) -> dict:
    return httpx.get(f"{base_url}/targets/paces", params=params).json()


def _wait_event(base_url: str, event_id: str, status: str, timeout: float = 15.0) -> dict:
    return wait_for(
        lambda: (lambda e: e if e["status"] == status else None)(
            _get_event(base_url, event_id)
        ),
        timeout=timeout,
    )


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


# -------------------------------------------------- API shape and normalisation

def test_pace_api_normalises_idempotent_lists_and_validates(server):
    target = f"{server.base_url}/t/record?pace-shape={uuid.uuid4().hex}"

    # Never configured: no row, envelope still carries the paging fields.
    body = _get_paces(server.base_url)
    assert set(body) == {"paces", "limit", "offset"}
    assert _get_paces(server.base_url, target=target)["paces"] == []

    # Uppercase scheme normalises exactly as event ingest normalises targets.
    upper = target.replace("http://", "HTTP://")
    resp = _put_pace(server.base_url, upper, 5)
    assert resp.status_code == 200
    row = resp.json()
    assert set(row) == {
        "target", "interval_seconds", "last_claimed_at",
        "next_claimable_at", "updated_at",
    }
    assert row["target"] == target  # normalised form
    assert row["interval_seconds"] == 5
    assert row["last_claimed_at"] is None       # no claim yet
    assert row["next_claimable_at"] is None     # nothing scheduled
    datetime.fromisoformat(row["updated_at"])

    # Exact-match listing and active filter find the one row.
    assert [r["target"] for r in _get_paces(server.base_url, target=target)["paces"]] == [target]
    assert len(_get_paces(server.base_url, active=True)["paces"]) >= 1
    assert _get_paces(server.base_url, target=target, active=False)["paces"] == []

    # Repeating the same interval is idempotent: the whole row is unchanged.
    time.sleep(0.01)
    assert _put_pace(server.base_url, target, 5).json() == row

    # Disabling keeps the row but flips it to the inactive filter; fractional
    # intervals are accepted and echoed as floats.
    disabled = _put_pace(server.base_url, target, 0).json()
    assert disabled["interval_seconds"] == 0
    assert disabled["updated_at"] != row["updated_at"]
    assert [r["target"] for r in _get_paces(server.base_url, active=False)["paces"]] == [target]
    frac = _put_pace(server.base_url, target, 0.25).json()
    assert frac["interval_seconds"] == 0.25

    # Boundary value 3600 is accepted.
    assert _put_pace(server.base_url, target, 3600).status_code == 200

    # Invalid input -> 422.
    for bad in (-1, 3601, "soon", None):
        assert httpx.put(
            f"{server.base_url}/targets/pace",
            json={"target": target, "interval_seconds": bad},
        ).status_code == 422, bad
    assert httpx.put(
        f"{server.base_url}/targets/pace", json={"target": target}
    ).status_code == 422
    assert httpx.put(
        f"{server.base_url}/targets/pace",
        json={"target": "not-a-url", "interval_seconds": 5},
    ).status_code == 422


# ------------------------------------- first delivery and retries both pace out

def test_first_delivery_immediate_but_next_event_waits_interval(server):
    base = server.base_url
    target = f"{base}/t/flaky?fail_times=1&pace-gap={uuid.uuid4().hex}"
    try:
        assert _put_pace(base, target, 1).status_code == 200

        first = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        # The very first delivery for a configured target goes immediately:
        # no prior claim means no interval to wait out.
        delivered1 = _wait_event(base, first["id"], "delivered", timeout=10)
        first_attempt_at = _parse(delivered1["attempts"][0]["attempted_at"])
        assert (_parse(delivered1["created_at"]) - first_attempt_at).total_seconds() < 1.0
        # The first attempt failed once, so the retry claim also consumed an
        # interval; the event still finishes with the 1s pace on top of the
        # 1s backoff (the two coincide here).
        assert delivered1["attempts_count"] == 2

        # Second event: its head is due immediately but the target's pace was
        # consumed by the retry claim, so it waits about one interval. (The
        # flaky sink fails every event's first attempt, so this one also
        # needs two attempts — what matters is when the first one starts.)
        second = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        delivered2 = _wait_event(base, second["id"], "delivered", timeout=10)
        gap = (
            _parse(delivered2["attempts"][0]["attempted_at"])
            - _parse(delivered1["attempts"][-1]["attempted_at"])
        ).total_seconds()
        assert 0.85 <= gap <= 2.5, gap

        listed = _get_paces(base, target=target)["paces"][0]
        assert listed["interval_seconds"] == 1
        assert listed["last_claimed_at"] is not None
        assert listed["next_claimable_at"] is not None
    finally:
        _put_pace(base, target, 0)


def test_retry_claims_also_consume_the_interval(server):
    base = server.base_url
    # Always-500 target, pace 2s: with no pace the claims land at 0/1/3/7s
    # (1/2/4 backoff); with the pace they land at 0/2/4/8s, so the gaps prove
    # retries occupy intervals too.
    target = f"{base}/t/probe?fail_times=10&pace-retry={uuid.uuid4().hex}"
    try:
        assert _put_pace(base, target, 2).status_code == 200
        event = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        dead = _wait_event(base, event["id"], "dead", timeout=30)
        assert [a["status_code"] for a in dead["attempts"]] == [500, 500, 500, 500]
        times = [_parse(a["attempted_at"]) for a in dead["attempts"]]
        gaps = [(times[i] - times[i - 1]).total_seconds() for i in range(1, 4)]
        assert 1.8 <= gaps[0] <= 3.2, gaps
        assert 1.8 <= gaps[1] <= 3.5, gaps
        assert 3.5 <= gaps[2] <= 5.5, gaps
    finally:
        _put_pace(base, target, 0)


# ------------------------------ changing the interval recomputes from last claim

def test_changing_interval_recomputes_next_claim_from_last_claim(server):
    base = server.base_url
    target = f"{base}/t/record?pace-adjust={uuid.uuid4().hex}"
    try:
        assert _put_pace(base, target, 30).status_code == 200
        first = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        _wait_event(base, first["id"], "delivered")
        second = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()

        # Under the 30s interval the second event is frozen at the head.
        time.sleep(0.8)
        blocked = _get_event(base, second["id"])
        assert blocked["status"] == "queued"
        assert blocked["attempts_count"] == 0

        # Shorten to 1s WITHOUT resetting the last-claim moment: the
        # next-claimable deadline is recomputed from the existing stamp, so it
        # is already due (roughly a second passed since the first claim).
        adjusted = _put_pace(base, target, 1).json()
        assert adjusted["interval_seconds"] == 1
        assert adjusted["last_claimed_at"] is not None
        assert adjusted["next_claimable_at"] is not None

        delivered = _wait_event(base, second["id"], "delivered", timeout=5)
        gap = (
            _parse(delivered["attempts"][0]["attempted_at"])
            - _parse(_get_event(base, first["id"])["delivered_at"])
        ).total_seconds()
        # Not the old 30s wait, and not instant: recomputed as last+1s.
        assert 0.8 <= gap <= 2.5, gap

        # Widening the interval also takes effect on the next claim at once.
        assert _put_pace(base, target, 30).status_code == 200
        third = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        time.sleep(0.6)
        assert _get_event(base, third["id"])["status"] == "queued"
    finally:
        _put_pace(base, target, 0)


# --------------------------------------------- interval 0 lifts the gate at once

def test_setting_interval_zero_lifts_pace_immediately(server):
    base = server.base_url
    target = f"{base}/t/record?pace-off={uuid.uuid4().hex}"
    try:
        _put_pace(base, target, 30)
        first = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        _wait_event(base, first["id"], "delivered")
        second = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        time.sleep(0.6)
        assert _get_event(base, second["id"])["status"] == "queued"

        # Disabling drains the blocked head immediately.
        assert _put_pace(base, target, 0).json()["interval_seconds"] == 0
        _wait_event(base, second["id"], "delivered", timeout=5)

        # And further events flow without any interval wait.
        third = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        delivered3 = _wait_event(base, third["id"], "delivered", timeout=5)
        gap = (
            _parse(delivered3["delivered_at"])
            - _parse(_get_event(base, second["id"])["delivered_at"])
        ).total_seconds()
        assert gap < 1.0, gap

        # The disabled row stays listed but inactive; its last stamp remains
        # and there is no next-claimable deadline.
        row = _get_paces(base, target=target)["paces"][0]
        assert row["interval_seconds"] == 0
        assert row["last_claimed_at"] is not None
        assert row["next_claimable_at"] is None
    finally:
        _put_pace(base, target, 0)


# --------------------------------------- pace only blocks that target's own head

def test_paced_head_blocked_but_other_targets_deliver(server):
    base = server.base_url
    paced = f"{base}/t/record?pace-own={uuid.uuid4().hex}"
    other = f"{base}/t/record?pace-other={uuid.uuid4().hex}"
    try:
        _put_pace(base, paced, 30)
        first = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": paced, "payload": {},
        }).json()
        _wait_event(base, first["id"], "delivered")
        second = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": paced, "payload": {},
        }).json()
        healthy = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": other, "payload": {},
        }).json()

        # The unpaced target is completely unaffected and delivers now.
        _wait_event(base, healthy["id"], "delivered")
        time.sleep(0.4)
        assert _get_event(base, second["id"])["status"] == "queued"
        assert _get_event(base, second["id"])["attempts_count"] == 0
    finally:
        _put_pace(base, paced, 0)


# --------------------------------------------- expiry sweep ignores the pace

def test_expired_queued_events_swept_while_paced(server):
    base = server.base_url
    target = f"{base}/t/record?pace-expiry={uuid.uuid4().hex}"
    try:
        _put_pace(base, target, 30)
        head = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        _wait_event(base, head["id"], "delivered")
        middle = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex,
            "target": target,
            "payload": {},
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        }).json()
        tail = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()

        # The middle event expires with zero attempts even though the pace
        # would have blocked it anyway; the sweep deliberately ignores pace.
        expired = _wait_event(base, middle["id"], "expired", timeout=10)
        assert expired["attempts_count"] == 0
        # The tail becomes the head and stays pace-blocked.
        time.sleep(0.3)
        assert _get_event(base, tail["id"])["status"] == "queued"
        assert _get_event(base, tail["id"])["attempts_count"] == 0

        _put_pace(base, target, 0)
        delivered = _wait_event(base, tail["id"], "delivered")
        assert delivered["attempts_count"] == 1
    finally:
        _put_pace(base, target, 0)


# ------------------------------------------------- pace stacks with the pause

def test_pause_and_pace_stack_and_first_claim_after_resume_is_immediate(server):
    base = server.base_url
    target = f"{base}/t/record?pace-pause={uuid.uuid4().hex}"
    try:
        httpx.put(f"{base}/targets/pause", json={"target": target, "paused": True})
        _put_pace(base, target, 30)
        event = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        time.sleep(0.6)
        assert _get_event(base, event["id"])["status"] == "queued"
        assert _get_event(base, event["id"])["attempts_count"] == 0

        # Resume: the pace has never stamped a claim, so the first delivery
        # for the target still goes out immediately.
        httpx.put(f"{base}/targets/pause", json={"target": target, "paused": False})
        _wait_event(base, event["id"], "delivered", timeout=5)
    finally:
        httpx.put(f"{base}/targets/pause", json={"target": target, "paused": False})
        _put_pace(base, target, 0)


# ------------------------------- pace write and claim are serialised by the DB

def test_claim_and_pace_are_serialized_in_database_transactions(server):
    targets = [
        f"{server.base_url}/dev/sink?pace-serial={uuid.uuid4().hex}" for _ in range(10)
    ]
    paced_target = targets[0]
    committed = threading.Event()

    def pace():
        db.set_target_pace(paced_target, 30)
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

        writer = threading.Thread(target=pace)
        writer.start()
        assert not committed.wait(0.3), "pace commit must wait for the claim critical section"

        # Same-thread re-entry into the RLock: this claim is one serialised
        # sequence with the pending pace write and commits first. First claims
        # for every fresh target go out immediately (no prior stamp).
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

    # The claim stamped last_claim_ts for every claimed target; the now-paced
    # target's fresh due event is blocked for 30s, the others untouched.
    with db.db_lock:
        conn = db.get_conn()
        extra_id = uuid.uuid4().hex
        conn.execute(
            "INSERT INTO events (id, idempotency_key, target, payload, status, "
            "attempts_count, created_at, next_attempt_at, deliver_after) "
            "VALUES (?, ?, ?, '{}', 'queued', 0, 'x', ?, 'x')",
            (extra_id, uuid.uuid4().hex, paced_target, time.time() - 1),
        )
        conn.commit()
        assert db.claim_due_events() == []

    # Disabling lifts the gate within the same serialised transactions.
    db.set_target_pace(paced_target, 0)
    with db.db_lock:
        resumed = db.claim_due_events()
        assert [c["id"] for c in resumed] == [extra_id]

    # Tidy up the extra row too.
    with db.db_lock:
        conn = db.get_conn()
        conn.execute(
            "UPDATE events SET status = 'delivered', next_attempt_at = NULL "
            "WHERE id = ?",
            (extra_id,),
        )
        conn.commit()


# ------------------------------ old databases auto-init; state survives restart

def test_pace_table_auto_initialises_on_old_database_and_survives_restart():
    db_path = tempfile.mktemp(prefix="webhook-pace-restart-", suffix=".db")
    try:
        # A "legacy" database with only the original events table.
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
             "print(json.dumps(db.list_target_paces())); "
             "db.set_target_pace('http://example.com/legacy', 5); "
             "db.create_event('legacy-key', 'http://example.com/legacy', {}); "
             "assert len(db.claim_due_events()) == 1; "
             "print(json.dumps(db.list_target_paces(target='http://example.com/legacy')))"],
            cwd="/workspace", env=common_env, capture_output=True, text=True,
        )
        assert first.returncode == 0, first.stderr
        lines = [json.loads(line) for line in first.stdout.splitlines()]
        assert lines[0] == []  # table created empty on the old database
        assert len(lines[1]) == 1
        row = lines[1][0]
        assert row["target"] == "http://example.com/legacy"
        assert row["interval_seconds"] == 5
        assert row["last_claimed_at"]          # the claim stamped it
        assert row["next_claimable_at"]

        # A fresh process ("restart") re-runs init on the same file and keeps
        # both the configured interval and the last-claim stamp.
        second = subprocess.run(
            [sys.executable, "-c",
             "import json; from app import db; db.init_db(); "
             "print(json.dumps(db.list_target_paces()))"],
            cwd="/workspace", env=common_env, capture_output=True, text=True,
        )
        assert second.returncode == 0, second.stderr
        rows = json.loads(second.stdout.strip())
        assert len(rows) == 1
        assert rows[0]["target"] == "http://example.com/legacy"
        assert rows[0]["interval_seconds"] == 5
        assert rows[0]["last_claimed_at"] == row["last_claimed_at"]
        assert rows[0]["next_claimable_at"] == row["next_claimable_at"]
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except OSError:
                pass
