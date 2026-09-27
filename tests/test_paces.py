"""End-to-end tests for per-target delivery pacing."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone

import httpx

from app import db
from tests.test_webhooks import _get_event, _post_event_obj
from .conftest import wait_for


def _put_pace(base_url: str, target: str, interval_seconds) -> httpx.Response:
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

def test_pace_api_normalises_idempotent_and_validates(server):
    # A target never configured has no row and behaves as unpaced.
    target = f"{server.base_url}/t/record?pace-shape={uuid.uuid4().hex}"
    assert _get_paces(server.base_url, target=target)["paces"] == []

    # Uppercase scheme/host is normalised exactly as event ingest normalises
    # targets (str(HttpUrl)), so this matches stored targets byte-for-byte.
    upper = target.replace("http://", "HTTP://")
    resp = _put_pace(server.base_url, upper, 5)
    assert resp.status_code == 200
    row = resp.json()
    assert set(row) == {
        "target", "interval_seconds", "last_claim_at", "next_claim_at",
        "updated_at",
    }
    assert row["target"] == target  # normalised form
    assert row["interval_seconds"] == 5
    assert row["last_claim_at"] is None  # never claimed yet
    assert row["next_claim_at"] is None
    datetime.fromisoformat(row["updated_at"])  # parseable RFC 3339 stamp

    # Listing the same normalised target (exact match) finds the one row.
    listed = _get_paces(server.base_url, target=target)["paces"]
    assert [r["target"] for r in listed] == [target]

    # Repeating the same setting is idempotent: updated_at does not move.
    time.sleep(0.01)
    again = _put_pace(server.base_url, target, 5).json()
    assert again == row

    # Changing the interval refreshes updated_at; disabling keeps the row.
    changed = _put_pace(server.base_url, target, 2).json()
    assert changed["interval_seconds"] == 2
    assert changed["updated_at"] != row["updated_at"]
    disabled = _put_pace(server.base_url, target, 0).json()
    assert disabled["interval_seconds"] == 0
    assert disabled["next_claim_at"] is None
    assert [r["target"] for r in _get_paces(server.base_url, target=target)["paces"]] == [target]

    # List envelope carries the paging fields.
    body = _get_paces(server.base_url)
    assert set(body) == {"paces", "limit", "offset"}

    # Invalid input is a 422: missing field, out-of-range interval, wrong
    # types, bad URL.
    assert httpx.put(
        f"{server.base_url}/targets/pace", json={"target": target}
    ).status_code == 422
    assert _put_pace(server.base_url, target, -1).status_code == 422
    assert _put_pace(server.base_url, target, 3601).status_code == 422
    assert _put_pace(server.base_url, target, "fast").status_code == 422
    assert _put_pace(server.base_url, "not-a-url", 5).status_code == 422
    # Bounds are inclusive: 0 and 3600 are both accepted.
    assert _put_pace(server.base_url, target, 3600).status_code == 200
    assert _put_pace(server.base_url, target, 0).status_code == 200


# ----------------------------------- pacing spaces claims; other targets run

def test_paced_target_spacing_and_other_targets_unaffected(server):
    base = server.base_url
    target = f"{base}/t/record?paced={uuid.uuid4().hex}"
    other = f"{base}/t/record?unpaced={uuid.uuid4().hex}"
    try:
        assert _put_pace(base, target, 2).status_code == 200

        first = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        # The first claim is immediate (the target was never claimed).
        f1 = _wait_event(base, first["id"], "delivered")
        assert f1["attempts_count"] == 1

        second = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        third = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        healthy = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": other, "payload": {},
        }).json()

        # The unpaced target is delivered immediately even though the paced
        # target has a backlog.
        _wait_event(base, healthy["id"], "delivered", timeout=5)
        # The paced head waits for the interval: still queued well inside it.
        time.sleep(0.8)
        assert _get_event(base, second["id"])["status"] == "queued"
        assert _get_event(base, second["id"])["attempts_count"] == 0

        # Both drain eventually, in head order, spaced by the interval.
        f2 = _wait_event(base, second["id"], "delivered", timeout=10)
        f3 = _wait_event(base, third["id"], "delivered", timeout=10)
        assert f2["attempts_count"] == 1 and f3["attempts_count"] == 1
        gap12 = _parse(f2["delivered_at"]) - _parse(f1["delivered_at"])
        gap23 = _parse(f3["delivered_at"]) - _parse(f2["delivered_at"])
        assert gap12 >= timedelta(seconds=1.5)
        assert gap23 >= timedelta(seconds=1.5)

        # The pace row exposes the last claim and the recomputed next slot.
        pace = _get_paces(base, target=target)["paces"][0]
        assert pace["interval_seconds"] == 2
        assert pace["last_claim_at"] is not None
        assert _parse(pace["next_claim_at"]) - _parse(pace["last_claim_at"]) \
            == timedelta(seconds=2)
    finally:
        _put_pace(base, target, 0)


# ------------------------------------------- retries consume the interval too

def test_pace_applies_to_retries_not_just_first_attempts(server):
    base = server.base_url
    # Fails the first attempt, then succeeds: without pacing the retry would
    # go out after the 1s backoff; pacing stretches it to the interval.
    target = f"{base}/t/flaky?fail_times=1&paced-retry={uuid.uuid4().hex}"
    try:
        assert _put_pace(base, target, 3).status_code == 200
        event = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()

        delivered = _wait_event(base, event["id"], "delivered", timeout=15)
        assert delivered["attempts_count"] == 2
        first, second = delivered["attempts"]
        assert first["status_code"] == 500
        assert second["status_code"] == 200
        gap = _parse(second["attempted_at"]) - _parse(first["attempted_at"])
        # max(1s backoff, 3s pace) — clearly above the bare 1s backoff.
        assert gap >= timedelta(seconds=2.5)
    finally:
        _put_pace(base, target, 0)


# ------------------------- adjust recomputes from last claim; disable lifts

def test_adjust_recomputes_from_last_claim_and_disable_lifts(server):
    base = server.base_url
    target = f"{base}/t/record?pace-adjust={uuid.uuid4().hex}"
    try:
        assert _put_pace(base, target, 60).status_code == 200
        head = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        _wait_event(base, head["id"], "delivered")

        queued = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        time.sleep(0.5)
        assert _get_event(base, queued["id"])["status"] == "queued"

        # Shortening the interval recomputes the next slot from the existing
        # last-claim stamp: 60s -> 1s makes the event claimable right away.
        assert _put_pace(base, target, 1).status_code == 200
        _wait_event(base, queued["id"], "delivered", timeout=5)

        # Re-enable a long interval: the next event is frozen again ...
        assert _put_pace(base, target, 60).status_code == 200
        frozen = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        time.sleep(0.8)
        assert _get_event(base, frozen["id"])["status"] == "queued"

        # ... and setting 0 lifts the limit immediately.
        assert _put_pace(base, target, 0).status_code == 200
        _wait_event(base, frozen["id"], "delivered", timeout=5)
    finally:
        _put_pace(base, target, 0)


# ------------------------------------ expiry sweep is not affected by pacing

def test_expired_queued_events_are_swept_despite_pacing(server):
    base = server.base_url
    target = f"{base}/t/record?pace-expiry={uuid.uuid4().hex}"
    try:
        assert _put_pace(base, target, 60).status_code == 200
        # Claim the slot so the expiring head is genuinely pace-blocked.
        first = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()
        _wait_event(base, first["id"], "delivered")

        expiring = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex,
            "target": target,
            "payload": {},
            "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
        }).json()
        tail = _post_event_obj(base, {
            "idempotency_key": uuid.uuid4().hex, "target": target, "payload": {},
        }).json()

        # The pace-blocked head still expires on its deadline, with zero
        # attempts, and stops blocking the queue.
        expired = _wait_event(base, expiring["id"], "expired", timeout=10)
        assert expired["attempts_count"] == 0
        assert expired["attempts"] == []
        # The tail is now the head but remains frozen by the pace.
        time.sleep(0.3)
        assert _get_event(base, tail["id"])["status"] == "queued"

        _put_pace(base, target, 0)
        delivered = _wait_event(base, tail["id"], "delivered")
        assert delivered["attempts_count"] == 1
    finally:
        _put_pace(base, target, 0)


# -------------------------------------- old databases auto-init; state survives

def test_pace_table_auto_initialises_on_old_database_and_survives_restart():
    db_path = tempfile.mktemp(prefix="webhook-pace-restart-", suffix=".db")
    try:
        # A "legacy" database with only the original events table, no
        # target_paces table at all.
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
             "import json, time; from app import db; db.init_db(); "
             "print(json.dumps(db.list_target_paces())); "
             "db.set_target_pace('http://example.com/legacy', 30); "
             "conn = db.get_conn(); "
             "conn.execute(\"INSERT INTO events (id, idempotency_key, target, "
             "payload, status, attempts_count, created_at, next_attempt_at, "
             "deliver_after) VALUES ('e1', 'k1', 'http://example.com/legacy', "
             "'{}', 'queued', 0, 'x', ?, 'x')\", (time.time() - 1,)); "
             "conn.commit(); "
             "claimed = db.claim_due_events(); "
             "print(json.dumps([c['id'] for c in claimed])); "
             "print(json.dumps(db.list_target_paces("
             "target='http://example.com/legacy')))"],
            cwd="/workspace", env=common_env, capture_output=True, text=True,
        )
        assert first.returncode == 0, first.stderr
        lines = [json.loads(line) for line in first.stdout.splitlines()]
        assert lines[0] == []  # table was created empty on the old database
        assert lines[1] == ["e1"]  # first claim is immediate
        row = lines[2][0]
        assert row["target"] == "http://example.com/legacy"
        assert row["interval_seconds"] == 30
        assert row["last_claim_at"] is not None
        assert row["next_claim_at"] is not None
        assert row["updated_at"]

        # A fresh process ("restart") re-runs init on the same file and sees
        # the persisted configuration *and* the last-claim stamp; init_db()
        # stays idempotent.
        second = subprocess.run(
            [sys.executable, "-c",
             "import json; from app import db; db.init_db(); "
             "print(json.dumps(db.list_target_paces()))"],
            cwd="/workspace", env=common_env, capture_output=True, text=True,
        )
        assert second.returncode == 0, second.stderr
        rows = json.loads(second.stdout.strip())
        assert rows == lines[2]
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except OSError:
                pass
