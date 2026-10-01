"""SQLite persistence layer.

A single connection is shared between the API threads and the worker
threads and is protected by ``db_lock``. The database is the source of
truth for the delivery state machine, so no delivery state lives only in
memory and a crash can never permanently lose or duplicate an event.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone

from . import config

logger = logging.getLogger("webhook.db")

# Event lifecycle:
#   queued     - waiting for the first attempt or a retry
#   processing - an HTTP attempt is in flight (at most one per endpoint)
#   delivered  - got a 2xx response (terminal)
#   dead       - all attempts of the current round exhausted (dead-letter);
#                terminal unless replayed, which starts a fresh backoff round
#   expired    - still queued when its ``expires_at`` deadline was reached
#                (terminal); never receives an attempt and stops blocking
#                its endpoint so the next event can proceed. Events already
#                in ``processing`` are never interrupted by their deadline.
TERMINAL_STATUSES = ("delivered", "dead", "expired")

# Per-target circuit-breaker lifecycle:
#   closed    - deliveries proceed; ``consecutive_failures`` counts the
#               current run of failed attempts and resets to 0 on any 2xx
#   open      - the target failed BREAKER_FAILURE_THRESHOLD times in a row;
#               no event for the target may be claimed until
#               ``open_until``. Other targets are unaffected
#   half-open - ``open_until`` has elapsed; exactly one event (the due head
#               event) has been claimed as a probe and is in flight. While
#               it runs, nothing else for the target is claimable
# A 2xx probe closes the breaker; any other outcome reopens it for another
# full open window. All breaker state lives in the ``breakers`` table, so a
# restart keeps the deadline counting instead of resetting it.


# Per-target maintenance pause:
#   A row exists only for targets explicitly configured via PUT
#   /targets/pause. ``paused=1`` gates scheduling: no *new* event for the
#   target may be claimed (claim_due_events LEFT JOINs this table in its
#   single claim transaction), while an event already ``processing`` is left
#   untouched and its in-flight request completes normally. Pause never
#   resets attempt counts, backoff or breaker state; queued events still
#   cross their ``expires_at`` deadline and are swept to ``expired`` while a
#   target is paused. Setting the same flag repeatedly is idempotent. State
#   lives only in SQLite, so it survives restarts and is initialised on old
#   databases by ``CREATE TABLE IF NOT EXISTS``.


# Per-target delivery pace (rate limit):
#   A row exists only for targets explicitly configured via PUT
#   /targets/pace. ``interval_seconds > 0`` gates scheduling in the same
#   claim transaction as everything else: the due head event is claimable
#   only once ``last_claim_ts + interval_seconds`` has passed, where
#   ``last_claim_ts`` is stamped at every claim (first delivery *and* every
#   retry claim). ``interval_seconds = 0`` disables the gate immediately.
#   Changing the interval does not touch ``last_claim_ts`` — the next
#   claimable moment is recomputed from the last claim. The gate only holds
#   back that target's own head event (other targets claim independently)
#   and is stacked on top of the deliver_after/backoff, breaker and pause
#   gates; the expiry sweep deliberately ignores it. Like pauses, the
#   configuration and the last-claim stamp persist in SQLite across
#   restarts and the table is auto-created on old databases. Claims also
#   stamp *implicit* rows (``configured = 0``, interval 0) for targets
#   never explicitly configured; they stay invisible to the listing and a
#   later ``PUT`` adopts their stamp, so configuring a pace right after a
#   claim cannot smuggle a second delivery through.


class ReplayNotAllowedError(Exception):
    """Raised when a replay is requested for an event that is not dead."""

    def __init__(self, event_id: str, status: str) -> None:
        self.event_id = event_id
        self.status = status
        super().__init__(
            f"event {event_id} is '{status}'; only dead-letter events can be replayed"
        )


class InvalidDeliveryWindowError(ValueError):
    """Raised when ``expires_at`` is not strictly later than ``deliver_after``."""

db_lock = threading.RLock()
_conn: sqlite3.Connection | None = None


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _iso_utc(dt: datetime) -> str:
    """Normalise an aware datetime to the UTC ISO-8601 format used everywhere."""
    return dt.astimezone(timezone.utc).isoformat(timespec="microseconds")


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(os.path.abspath(config.DB_PATH)), exist_ok=True)
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _conn.execute("PRAGMA busy_timeout=5000")
    return _conn


def init_db() -> None:
    with db_lock:
        conn = get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id              TEXT PRIMARY KEY,
                idempotency_key TEXT NOT NULL UNIQUE,
                target          TEXT NOT NULL,
                payload         TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'queued',
                attempts_count  INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL,
                next_attempt_at REAL,
                delivered_at    TEXT,
                replay_count    INTEGER NOT NULL DEFAULT 0,
                last_replayed_at TEXT,
                round_base_attempts INTEGER NOT NULL DEFAULT 0,
                deliver_after   TEXT NOT NULL,
                expires_at      TEXT,
                expires_at_ts   REAL
            );

            CREATE TABLE IF NOT EXISTS attempts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id     TEXT NOT NULL REFERENCES events(id),
                attempt_no   INTEGER NOT NULL,
                status_code  INTEGER,
                error        TEXT,
                attempted_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS breakers (
                target               TEXT PRIMARY KEY,
                state                TEXT NOT NULL DEFAULT 'closed',
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                open_until_ts        REAL,
                probe_event_id       TEXT
            );

            CREATE TABLE IF NOT EXISTS target_pauses (
                target     TEXT PRIMARY KEY,
                paused     INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS target_paces (
                target           TEXT PRIMARY KEY,
                interval_seconds REAL NOT NULL DEFAULT 0,
                configured       INTEGER NOT NULL DEFAULT 0,
                last_claim_ts    REAL,
                updated_at       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_status
                ON events(status, next_attempt_at);
            CREATE INDEX IF NOT EXISTS idx_events_target ON events(target);
            CREATE INDEX IF NOT EXISTS idx_attempts_event ON attempts(event_id);
            """
        )
        _migrate_events(conn)
        _migrate_target_paces(conn)
        conn.commit()


def _migrate_target_paces(conn: sqlite3.Connection) -> None:
    """Backfill the ``configured`` flag on databases created earlier.

    Databases from the first pace revision got the ``target_paces`` table
    without the ``configured`` column that distinguishes explicit PUT rows
    from implicit claim stamps; ``CREATE TABLE IF NOT EXISTS`` never alters
    it. Existing rows were always explicit PUT rows, hence the default 1.
    """
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(target_paces)")}
    if cols and "configured" not in cols:
        conn.execute(
            "ALTER TABLE target_paces ADD COLUMN configured INTEGER NOT NULL DEFAULT 1"
        )


def _migrate_events(conn: sqlite3.Connection) -> None:
    """Add replay and delivery-window columns to older databases.

    ``CREATE TABLE IF NOT EXISTS`` leaves pre-existing files untouched, so
    the new columns are added idempotently here. Existing rows receive the
    defaults (``replay_count=0``, ``round_base_attempts=0``,
    ``deliver_after=created_at``, no ``expires_at``), which keeps their
    behaviour identical to the pre-window state machine: deliver
    immediately, never expire.
    """
    existing = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
    for column, ddl in (
        ("replay_count", "replay_count INTEGER NOT NULL DEFAULT 0"),
        ("last_replayed_at", "last_replayed_at TEXT"),
        ("round_base_attempts", "round_base_attempts INTEGER NOT NULL DEFAULT 0"),
        ("deliver_after", "deliver_after TEXT"),
        ("expires_at", "expires_at TEXT"),
        ("expires_at_ts", "expires_at_ts REAL"),
    ):
        if column not in existing:
            conn.execute(f"ALTER TABLE events ADD COLUMN {ddl}")
    # Rows created before delivery windows existed are due immediately and
    # never expire, exactly as before.
    conn.execute(
        "UPDATE events SET deliver_after = created_at WHERE deliver_after IS NULL"
    )


def reset_stuck_events() -> None:
    """Re-queue events left in 'processing' by a hard crash/restart.

    Such an event becomes the blocking head for its endpoint again and is
    safely redelivered. Delivery is at-least-once, which is why outbound
    requests carry an idempotency header.

    If the interrupted attempt was a half-open breaker probe, the breaker
    cannot keep waiting on a probe that will never return: it reopens with an
    already-elapsed deadline, so the recovered head event is probed again on
    the next tick instead of stalling that target forever. Open deadlines are
    left untouched, so the open window survives the restart exactly as it
    would have without one.
    """
    with db_lock:
        conn = get_conn()
        conn.execute(
            "UPDATE events SET status = 'queued', next_attempt_at = ? "
            "WHERE status = 'processing'",
            (time.time(),),
        )
        conn.execute(
            "UPDATE breakers SET state = 'open', open_until_ts = ?, "
            "probe_event_id = NULL WHERE state = 'half-open'",
            (time.time(),),
        )
        conn.commit()


def create_event(
    idempotency_key: str,
    target: str,
    payload: dict | list,
    deliver_after: datetime | None = None,
    expires_at: datetime | None = None,
) -> tuple[dict, bool]:
    """Insert an event. Returns ``(event, created)``.

    If the idempotency key already exists the stored event is returned with
    ``created=False``; nothing is inserted, nothing is enqueued again and
    the delivery-window fields of the new request are ignored entirely.

    For new events ``deliver_after`` defaults to the creation moment and
    ``expires_at`` (when given) must be strictly later than it, otherwise
    ``InvalidDeliveryWindowError`` is raised and nothing is stored.
    """
    event_id = uuid.uuid4().hex
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    with db_lock:
        conn = get_conn()
        existing = conn.execute(
            "SELECT * FROM events WHERE idempotency_key = ?", (idempotency_key,)
        ).fetchone()
        if existing is not None:
            return dict(existing), False

        now_dt = datetime.now(timezone.utc)
        deliver_after = deliver_after or now_dt
        if expires_at is not None and expires_at <= deliver_after:
            raise InvalidDeliveryWindowError(
                "expires_at must be later than deliver_after "
                f"(deliver_after={_iso_utc(deliver_after)}, "
                f"expires_at={_iso_utc(expires_at)})"
            )
        conn.execute(
            "INSERT INTO events "
            "(id, idempotency_key, target, payload, status, attempts_count, "
            " created_at, next_attempt_at, deliver_after, expires_at, expires_at_ts) "
            "VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?)",
            (
                event_id,
                idempotency_key,
                target,
                payload_json,
                _iso_utc(now_dt),
                deliver_after.timestamp(),
                _iso_utc(deliver_after),
                _iso_utc(expires_at) if expires_at is not None else None,
                expires_at.timestamp() if expires_at is not None else None,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return dict(row), True


def get_events_by_keys(keys: list[str]) -> list[dict]:
    """Return the stored events for the given idempotency keys (any order)."""
    if not keys:
        return []
    with db_lock:
        conn = get_conn()
        placeholders = ", ".join("?" for _ in keys)
        rows = conn.execute(
            f"SELECT * FROM events WHERE idempotency_key IN ({placeholders})",
            tuple(keys),
        ).fetchall()
        return [_hydrate(r) for r in rows]


def create_events_batch(items: list[dict]) -> list[tuple[dict, bool]]:
    """Insert a batch of new events in a single SQLite transaction.

    ``items`` carry ``idempotency_key``/``target``/``payload`` plus optional
    ``deliver_after``/``expires_at`` datetimes (already shape-validated by
    the API layer). Returns one ``(event, created)`` pair per item, in input
    order.

    The batch is all-or-nothing: every delivery window is validated before
    the first INSERT and any failure (invalid window, write error) rolls the
    transaction back, so a batch never lands partially. A key that already
    exists when its turn comes yields the stored event with
    ``created=False`` — re-sending a batch never enqueues anything twice.
    All events of one batch share a single creation timestamp, and inserts
    happen in input order so rowid (creation) ordering — which drives the
    per-endpoint delivery order — matches the caller's array order.
    """
    now_dt = datetime.now(timezone.utc)
    with db_lock:
        conn = get_conn()
        try:
            staged: list[dict] = []
            for item in items:
                existing = conn.execute(
                    "SELECT * FROM events WHERE idempotency_key = ?",
                    (item["idempotency_key"],),
                ).fetchone()
                if existing is not None:
                    staged.append({"existing": dict(existing)})
                    continue
                deliver_after = item["deliver_after"] or now_dt
                expires_at = item["expires_at"]
                if expires_at is not None and expires_at <= deliver_after:
                    raise InvalidDeliveryWindowError(
                        "expires_at must be later than deliver_after "
                        f"(idempotency_key={item['idempotency_key']!r}, "
                        f"deliver_after={_iso_utc(deliver_after)}, "
                        f"expires_at={_iso_utc(expires_at)})"
                    )
                staged.append({"params": (
                    uuid.uuid4().hex,
                    item["idempotency_key"],
                    item["target"],
                    json.dumps(
                        item["payload"], separators=(",", ":"), ensure_ascii=False
                    ),
                    _iso_utc(now_dt),
                    deliver_after.timestamp(),
                    _iso_utc(deliver_after),
                    _iso_utc(expires_at) if expires_at is not None else None,
                    expires_at.timestamp() if expires_at is not None else None,
                )})
            results: list[tuple[dict, bool]] = []
            for entry in staged:
                if "existing" in entry:
                    results.append((entry["existing"], False))
                    continue
                params = entry["params"]
                conn.execute(
                    "INSERT INTO events "
                    "(id, idempotency_key, target, payload, status, attempts_count, "
                    " created_at, next_attempt_at, deliver_after, expires_at, "
                    " expires_at_ts) "
                    "VALUES (?, ?, ?, ?, 'queued', 0, ?, ?, ?, ?, ?)",
                    params,
                )
                row = conn.execute(
                    "SELECT * FROM events WHERE id = ?", (params[0],)
                ).fetchone()
                results.append((dict(row), True))
            conn.commit()
            return results
        except Exception:
            # Nothing of this batch may survive a failed validation or write.
            conn.rollback()
            raise


def _hydrate(row: sqlite3.Row) -> dict:
    event = dict(row)
    try:
        event["payload"] = json.loads(event["payload"])
    except (TypeError, ValueError):
        pass
    return event


def claim_due_events() -> list[dict]:
    """Atomically claim the head event of every endpoint that is due.

    One transaction, three steps:

    1. Every ``queued`` event whose ``expires_at`` has passed is flipped to
       ``expired``. It never receives an attempt and stops blocking its
       endpoint, so the next event for the same target can proceed. Events
       already ``processing`` are deliberately not touched — an in-flight
       attempt is never interrupted by its deadline.
    2. An event is claimable only when it is ``queued``, its backoff and
       ``deliver_after`` (both folded into ``next_attempt_at``) have
       elapsed, and no earlier (lower rowid = earlier creation) event for
       the same target is still ``queued``/``processing``. Claimed events
       flip to ``processing`` so another tick can never double-claim them.
       This is the ordering guarantee: a later event cannot be picked while
       an earlier one is pending — even if that earlier event is merely
       waiting for its ``deliver_after`` — in flight, or backing off.
    3. The per-target breaker gates step 2: while it is ``open`` nothing for
       that target is claimable until ``open_until`` elapses; while it is
       ``half-open`` a probe is already in flight, so nothing else goes.
       Once an open deadline has elapsed, the single due head event is
       claimed and the breaker atomically flips to ``half-open`` with that
       event recorded as its probe. Different targets stay independent, so
       a breaker on one endpoint never blocks anyone else.
    4. A target under maintenance pause (``target_pauses.paused = 1``) gates
       step 2 exactly like the breaker: nothing new for it is claimable and
       other targets are unaffected. Events already ``processing`` are not
       touched, so an in-flight request always completes. The gate and the
       claim run in this one transaction under ``db_lock``, which is the same
       lock the pause write takes, so a committed pause is visible to the
       very next claim and a claim that committed first is never rolled
       back. The expiry sweep in step 1 deliberately ignores the pause.
    5. A target with a configured delivery pace
       (``target_paces.interval_seconds > 0``) gates step 2 as well: the due
       head event is claimable only when ``last_claim_ts`` is NULL (first
       delivery) or ``last_claim_ts + interval_seconds`` has elapsed.
       Every claim in step 2 stamps ``last_claim_ts`` in this same
       transaction, so first deliveries and retry claims alike consume an
       interval and the gate holds across restarts. The stamp and the pace
       write share ``db_lock`` and single transactions, so a committed
       interval change takes effect on the very next claim, recomputed from
       the unchanged last-claim moment; an interval of 0 disables the gate
       immediately. Only that target's own head event is held back — other
       targets still claim — and the expiry sweep in step 1 ignores the
       pace, so queued events expire on their deadline regardless.
    """
    now = time.time()
    with db_lock:
        conn = get_conn()
        conn.execute(
            "UPDATE events SET status = 'expired', next_attempt_at = NULL "
            "WHERE status = 'queued' "
            "AND expires_at_ts IS NOT NULL AND expires_at_ts <= ?",
            (now,),
        )
        rows = conn.execute(
            """
            SELECT e.* FROM events e
            LEFT JOIN breakers b ON b.target = e.target
            LEFT JOIN target_pauses p ON p.target = e.target
            LEFT JOIN target_paces pc ON pc.target = e.target
            WHERE e.status = 'queued'
              AND e.next_attempt_at <= ?
              AND (p.target IS NULL OR p.paused = 0)
              AND (
                    pc.target IS NULL
                    OR pc.interval_seconds = 0
                    OR pc.last_claim_ts IS NULL
                    OR pc.last_claim_ts + pc.interval_seconds <= ?
              )
              AND (
                    b.target IS NULL
                    OR b.state = 'closed'
                    OR (b.state = 'open' AND b.open_until_ts IS NOT NULL
                        AND b.open_until_ts <= ?)
              )
              AND NOT EXISTS (
                    SELECT 1 FROM events e2
                    WHERE e2.target = e.target
                      AND e2.rowid < e.rowid
                      AND e2.status IN ('queued', 'processing')
              )
            ORDER BY e.rowid
            """,
            (now, now, now),
        ).fetchall()
        if rows:
            conn.executemany(
                "UPDATE events SET status = 'processing' WHERE id = ?",
                [(r["id"],) for r in rows],
            )
            # An expired open deadline makes this claim the one and only
            # half-open probe; closed targets simply keep their state. The
            # old deadline is cleared: open_until is only meaningful while
            # the breaker is actually open.
            conn.executemany(
                "UPDATE breakers SET state = 'half-open', probe_event_id = ?, "
                "open_until_ts = NULL WHERE target = ? AND state = 'open'",
                [(r["id"], r["target"]) for r in rows],
            )
            # Every claim — first delivery and every retry — consumes a pace
            # interval by stamping the target's last-claim time in this same
            # transaction. The upsert also records implicit rows (interval 0,
            # unconfigured) for targets that have never been configured: if a
            # pace is committed afterwards, it keeps this stamp, so a claim
            # that serialised ahead of the pace write is still the target's
            # last claim. Rows with interval 0 keep stamping too, so
            # re-enabling the pace later recomputes from the true last claim.
            # Head ordering makes each claimed target unique; DISTINCT only
            # guards against duplicated input.
            # INSERT carries interval 0 / configured 0 for new implicit rows;
            # on conflict the UPDATE names only last_claim_ts, so an existing
            # explicit configuration (its own interval/configured columns) is
            # never touched.
            conn.executemany(
                "INSERT INTO target_paces (target, interval_seconds, configured, "
                "last_claim_ts, updated_at) VALUES (?, 0, 0, ?, '') "
                "ON CONFLICT(target) DO UPDATE SET last_claim_ts = "
                "excluded.last_claim_ts",
                [(target, now) for target in {r["target"] for r in rows}],
            )
        conn.commit()
        return [_hydrate(r) for r in rows]


def record_attempt(
    event_id: str, status_code: int | None, error: str | None
) -> dict:
    """Persist one finished attempt and advance the event state machine.

    2xx          -> delivered
    other / none -> queued with the next backoff of the current round
                    (1s, 2s, 4s after round attempts 1, 2, 3), or dead
                    after the 4th failed attempt of the round. Replaying
                    a dead event starts a new round: ``attempt_no`` keeps
                    counting up globally, but the backoff schedule and the
                    dead-letter cut-off are relative to the round.

    In the same transaction the target's circuit breaker is advanced: a 2xx
    closes it and zeroes the consecutive-failure counter; any other outcome
    increments the counter, opens it for ``BREAKER_OPEN_SECONDS`` at the
    ``BREAKER_FAILURE_THRESHOLD`` consecutive failure, and — when the attempt
    was a half-open probe — reopens it immediately for another full window.
    """
    with db_lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            raise KeyError(event_id)
        event = dict(row)
        attempt_no = event["attempts_count"] + 1
        round_attempt = attempt_no - event["round_base_attempts"]

        conn.execute(
            "INSERT INTO attempts "
            "(event_id, attempt_no, status_code, error, attempted_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (event_id, attempt_no, status_code, error, _iso()),
        )

        success = status_code is not None and 200 <= status_code < 300
        if success:
            conn.execute(
                "UPDATE events SET status = 'delivered', attempts_count = ?, "
                "next_attempt_at = NULL, delivered_at = ? WHERE id = ?",
                (attempt_no, _iso(), event_id),
            )
        elif round_attempt >= config.MAX_ATTEMPTS:
            conn.execute(
                "UPDATE events SET status = 'dead', attempts_count = ?, "
                "next_attempt_at = NULL WHERE id = ?",
                (attempt_no, event_id),
            )
        else:
            delay = config.RETRY_BACKOFF[round_attempt - 1]
            conn.execute(
                "UPDATE events SET status = 'queued', attempts_count = ?, "
                "next_attempt_at = ? WHERE id = ?",
                (attempt_no, time.time() + delay, event_id),
            )
        _record_breaker_outcome(conn, event["target"], event_id, success)
        conn.commit()
        return get_event(event_id)


def _record_breaker_outcome(
    conn: sqlite3.Connection, target: str, event_id: str, success: bool
) -> None:
    """Advance one target's breaker after an attempt finished.

    Caller holds ``db_lock`` and the surrounding transaction. At most one
    attempt per target is ever in flight (head-event ordering), so the
    read-modify-write below cannot race with another attempt to the same
    target.
    """
    row = conn.execute(
        "SELECT * FROM breakers WHERE target = ?", (target,)
    ).fetchone()
    current = dict(row) if row is not None else None
    was_probe = current is not None and (
        current["state"] == "half-open" and current["probe_event_id"] == event_id
    )

    if success:
        if current is not None and current["state"] != "closed":
            logger.info("breaker for %s closed after successful probe", target)
        conn.execute(
            "INSERT INTO breakers (target, state, consecutive_failures, "
            "open_until_ts, probe_event_id) VALUES (?, 'closed', 0, NULL, NULL) "
            "ON CONFLICT(target) DO UPDATE SET state = 'closed', "
            "consecutive_failures = 0, open_until_ts = NULL, probe_event_id = NULL",
            (target,),
        )
        return

    failures = (current["consecutive_failures"] if current is not None else 0) + 1
    # A failed half-open probe reopens regardless of the running counter;
    # ordinary failures open once the threshold is reached.
    should_open = was_probe or failures >= config.BREAKER_FAILURE_THRESHOLD
    if should_open:
        open_until = time.time() + config.BREAKER_OPEN_SECONDS
        conn.execute(
            "INSERT INTO breakers (target, state, consecutive_failures, "
            "open_until_ts, probe_event_id) VALUES (?, 'open', ?, ?, NULL) "
            "ON CONFLICT(target) DO UPDATE SET state = 'open', "
            "consecutive_failures = excluded.consecutive_failures, "
            "open_until_ts = excluded.open_until_ts, probe_event_id = NULL",
            (target, failures, open_until),
        )
        if was_probe:
            logger.warning(
                "breaker for %s reopened for %.0fs after failed probe "
                "(%d consecutive failures)",
                target,
                config.BREAKER_OPEN_SECONDS,
                failures,
            )
        else:
            logger.warning(
                "breaker for %s opened for %.0fs after %d consecutive failures",
                target,
                config.BREAKER_OPEN_SECONDS,
                failures,
            )
    else:
        # Sub-threshold failure (or a closed breaker with no row yet): keep
        # counting while the ordinary retry/backoff schedule proceeds.
        conn.execute(
            "INSERT INTO breakers (target, state, consecutive_failures, "
            "open_until_ts, probe_event_id) VALUES (?, 'closed', ?, NULL, NULL) "
            "ON CONFLICT(target) DO UPDATE SET state = 'closed', "
            "consecutive_failures = excluded.consecutive_failures, "
            "open_until_ts = NULL, probe_event_id = NULL",
            (target, failures),
        )


def replay_event(event_id: str) -> dict:
    """Re-queue a dead-letter event for one fresh, full backoff round.

    The ``dead -> queued`` flip is a single conditional UPDATE (one
    transaction): concurrent replays — even from another process — cannot
    both win, because the loser updates zero rows and gets
    ``ReplayNotAllowedError``. A replay can therefore never enqueue or
    deliver the event twice.

    Attempt history is kept untouched: ``attempt_no`` keeps counting up
    (5, 6, …), while ``round_base_attempts`` is lifted to the current
    ``attempts_count`` so the new round restarts the 1s/2s/4s schedule and
    grants a full set of attempts again. The idempotency key, event id and
    creation order (rowid) are not touched, so per-endpoint ordering and
    outbound idempotency headers are unchanged.
    """
    with db_lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            raise KeyError(event_id)
        if row["status"] != "dead":
            raise ReplayNotAllowedError(event_id, row["status"])
        cur = conn.execute(
            "UPDATE events SET status = 'queued', next_attempt_at = ?, "
            "replay_count = replay_count + 1, last_replayed_at = ?, "
            "round_base_attempts = attempts_count "
            "WHERE id = ? AND status = 'dead'",
            (time.time(), _iso(), event_id),
        )
        if cur.rowcount != 1:
            # Lost a cross-process replay race between SELECT and UPDATE.
            conn.rollback()
            raise ReplayNotAllowedError(event_id, "queued")
        conn.commit()
        return get_event(event_id)


def _attempts_for(conn: sqlite3.Connection, event_id: str) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT attempt_no, status_code, error, attempted_at "
            "FROM attempts WHERE event_id = ? ORDER BY attempt_no",
            (event_id,),
        ).fetchall()
    ]


def get_event(event_id: str) -> dict | None:
    with db_lock:
        conn = get_conn()
        row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        event = _hydrate(row)
        event["attempts"] = _attempts_for(conn, event_id)
        return event


def get_event_by_idempotency_key(key: str) -> dict | None:
    with db_lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM events WHERE idempotency_key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        event = _hydrate(row)
        event["attempts"] = _attempts_for(conn, event["id"])
        return event


def list_events(status: str | None = None, limit: int = 50, offset: int = 0) -> list[dict]:
    limit = max(1, min(limit, 200))
    with db_lock:
        conn = get_conn()
        if status:
            rows = conn.execute(
                "SELECT * FROM events WHERE status = ? ORDER BY rowid "
                "LIMIT ? OFFSET ?",
                (status, limit, offset),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY rowid LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        events = []
        for row in rows:
            event = _hydrate(row)
            event["attempts"] = _attempts_for(conn, event["id"])
            events.append(event)
        return events


def _hydrate_breaker(row: sqlite3.Row) -> dict:
    """Public breaker representation with a UTC ISO deadline."""
    open_until_ts = row["open_until_ts"]
    return {
        "target": row["target"],
        "state": row["state"],
        "consecutive_failures": row["consecutive_failures"],
        "open_until": (
            datetime.fromtimestamp(open_until_ts, tz=timezone.utc)
            .isoformat(timespec="microseconds")
            if open_until_ts is not None
            else None
        ),
        "probe_event_id": row["probe_event_id"],
    }


def list_breakers(
    state: str | None = None, target: str | None = None,
    limit: int = 200, offset: int = 0,
) -> list[dict]:
    """Return breaker rows, optionally filtered by state and/or exact target.

    A row exists for every target that has seen at least one delivery
    attempt, including closed breakers recovering from earlier failures.
    """
    limit = max(1, min(limit, 500))
    clauses: list[str] = []
    params: list = []
    if state:
        clauses.append("state = ?")
        params.append(state)
    if target:
        clauses.append("target = ?")
        params.append(target)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with db_lock:
        conn = get_conn()
        rows = conn.execute(
            f"SELECT * FROM breakers {where} ORDER BY target LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_hydrate_breaker(r) for r in rows]


# ------------------------------------------------------------ target pauses

def set_target_pause(target: str, paused: bool) -> dict:
    """Upsert one target's maintenance flag in a single transaction.

    The target string is matched exactly against the stored events' target
    column (callers normalise the URL with the same rule used at ingest, so
    the keys line up). Setting the same value repeatedly is idempotent: an
    already-correct row is returned untouched and ``updated_at`` does not
    move. The write takes ``db_lock`` — the very same lock and serialisation
    point as ``claim_due_events`` — so once the commit returns, every later
    claim transaction sees the flag (a paused target cannot be claimed),
    while a claim that committed first keeps its ``processing`` event, whose
    request runs to completion regardless of the flag.
    """
    with db_lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM target_pauses WHERE target = ?", (target,)
        ).fetchone()
        flag = 1 if paused else 0
        if row is not None and row["paused"] == flag:
            return _hydrate_pause(row)
        conn.execute(
            "INSERT INTO target_pauses (target, paused, updated_at) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(target) DO UPDATE SET paused = excluded.paused, "
            "updated_at = excluded.updated_at",
            (target, flag, _iso()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM target_pauses WHERE target = ?", (target,)
        ).fetchone()
        return _hydrate_pause(row)


def _hydrate_pause(row: sqlite3.Row) -> dict:
    return {
        "target": row["target"],
        "paused": bool(row["paused"]),
        "updated_at": row["updated_at"],
    }


def list_target_pauses(
    target: str | None = None,
    paused: bool | None = None,
    limit: int = 200, offset: int = 0,
) -> list[dict]:
    """Return configured maintenance rows, optionally filtered.

    Only targets ever set through ``PUT /targets/pause`` appear (a row
    without an explicit setting does not exist and behaves unpaused).
    """
    limit = max(1, min(limit, 500))
    clauses: list[str] = []
    params: list = []
    if target is not None:
        clauses.append("target = ?")
        params.append(target)
    if paused is not None:
        clauses.append("paused = ?")
        params.append(1 if paused else 0)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with db_lock:
        conn = get_conn()
        rows = conn.execute(
            f"SELECT * FROM target_pauses {where} ORDER BY target "
            "LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_hydrate_pause(r) for r in rows]


# ------------------------------------------------------------- target paces

def set_target_pace(target: str, interval_seconds: float) -> dict:
    """Upsert one target's delivery interval in a single transaction.

    The target string is matched exactly against the stored events' target
    column (callers normalise the URL with the same rule used at ingest, so
    the keys line up). ``interval_seconds`` of 0 disables the pace. Setting
    the same interval repeatedly is idempotent: an already-correct row is
    returned untouched, ``updated_at`` does not move and the persisted
    ``last_claim_ts`` is never reset. Changing to a different interval also
    leaves ``last_claim_ts`` alone — the next claimable moment is recomputed
    by the claim gate as ``last_claim_ts + interval_seconds``. The write
    takes ``db_lock`` — the very same lock and serialisation point as
    ``claim_due_events`` — so once the commit returns, the next claim
    transaction immediately applies the new interval (and an interval of 0
    lifts the gate at once), while a claim that committed first keeps its
    ``processing`` event and its last-claim stamp.
    """
    with db_lock:
        conn = get_conn()
        row = conn.execute(
            "SELECT * FROM target_paces WHERE target = ? AND configured = 1",
            (target,),
        ).fetchone()
        if row is not None and float(row["interval_seconds"]) == float(interval_seconds):
            return _hydrate_pace(row)
        # On an implicit (stamp-only) row created by a claim that ran before
        # any configuration, the upsert adopts its last_claim_ts instead of
        # resetting it: that claim genuinely was the target's last claim.
        conn.execute(
            "INSERT INTO target_paces (target, interval_seconds, configured, "
            "last_claim_ts, updated_at) VALUES (?, ?, 1, NULL, ?) "
            "ON CONFLICT(target) DO UPDATE SET interval_seconds = "
            "excluded.interval_seconds, configured = 1, "
            "updated_at = excluded.updated_at",
            (target, float(interval_seconds), _iso()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM target_paces WHERE target = ?", (target,)
        ).fetchone()
        return _hydrate_pace(row)


def _hydrate_pace(row: sqlite3.Row) -> dict:
    """Public pace representation with ISO UTC timestamps."""
    interval = float(row["interval_seconds"])
    last_claim_ts = row["last_claim_ts"]
    # Echo integer-valued intervals as integers, matching JSON requests such
    # as ``5``; fractional intervals keep their float form.
    interval_out = int(interval) if interval.is_integer() else interval
    last_claimed_at = (
        datetime.fromtimestamp(last_claim_ts, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        if last_claim_ts is not None
        else None
    )
    next_claimable_at = None
    if interval > 0 and last_claim_ts is not None:
        next_claimable_at = (
            datetime.fromtimestamp(last_claim_ts + interval, tz=timezone.utc)
            .isoformat(timespec="microseconds")
        )
    return {
        "target": row["target"],
        "interval_seconds": interval_out,
        "last_claimed_at": last_claimed_at,
        "next_claimable_at": next_claimable_at,
        "updated_at": row["updated_at"],
    }


def list_target_paces(
    target: str | None = None,
    active: bool | None = None,
    limit: int = 200, offset: int = 0,
) -> list[dict]:
    """Return configured pace rows, optionally filtered.

    Only targets ever set through ``PUT /targets/pace`` appear (a row
    without an explicit setting does not exist and behaves as unpaced).
    ``active=True`` returns rows with a positive interval; ``active=False``
    returns explicitly disabled (interval 0) rows.
    """
    limit = max(1, min(limit, 500))
    clauses: list[str] = ["configured = 1"]
    params: list = []
    if target is not None:
        clauses.append("target = ?")
        params.append(target)
    if active is not None:
        clauses.append("interval_seconds > 0" if active else "interval_seconds = 0")
    where = "WHERE " + " AND ".join(clauses)
    with db_lock:
        conn = get_conn()
        rows = conn.execute(
            f"SELECT * FROM target_paces {where} ORDER BY target "
            "LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [_hydrate_pace(r) for r in rows]
