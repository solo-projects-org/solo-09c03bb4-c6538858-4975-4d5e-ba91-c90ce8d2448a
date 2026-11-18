"""SQLite helpers and schema for the carryover review app."""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL,
    high_threshold  REAL NOT NULL CHECK (high_threshold >= 0),
    blank_limit     REAL NOT NULL CHECK (blank_limit >= 0),
    data_version    INTEGER NOT NULL DEFAULT 0,
    result_state    TEXT NOT NULL DEFAULT 'pending'
                    CHECK (result_state IN ('pending', 'current', 'stale')),
    result_version  INTEGER,
    result_error    TEXT,
    computed_at     TEXT,
    rsd_limit       REAL CHECK (rsd_limit IS NULL OR rsd_limit >= 0),
    group_version   INTEGER NOT NULL DEFAULT 0,
    precision_state TEXT NOT NULL DEFAULT 'none'
                    CHECK (precision_state IN ('none', 'pending', 'current')),
    precision_data_version  INTEGER,
    precision_group_version INTEGER,
    precision_error TEXT,
    precision_computed_at TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS injections (
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    injection_no    TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('sample', 'blank')),
    peak_area       REAL CHECK (peak_area IS NULL OR peak_area >= 0),
    revision        INTEGER NOT NULL CHECK (revision >= 1),
    position        INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (batch_id, injection_no),
    CHECK (kind = 'blank' OR peak_area IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS injection_revisions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    injection_no    TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('sample', 'blank')),
    peak_area       REAL CHECK (peak_area IS NULL OR peak_area >= 0),
    revision        INTEGER NOT NULL CHECK (revision >= 1),
    accepted_at     TEXT NOT NULL,
    UNIQUE (batch_id, injection_no, revision)
);

CREATE TABLE IF NOT EXISTS results (
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    injection_no    TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    peak_area       REAL,
    revision        INTEGER NOT NULL,
    status          TEXT NOT NULL,
    trigger_injection TEXT,
    note            TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (batch_id, injection_no)
);

CREATE TABLE IF NOT EXISTS upload_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    accepted_count  INTEGER NOT NULL,
    skipped_count   INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

-- Current grouping assignment.  The primary key guarantees that one injection
-- can belong to only one sample group.
CREATE TABLE IF NOT EXISTS sample_groups (
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    injection_no    TEXT NOT NULL,
    group_name      TEXT NOT NULL,
    position        INTEGER NOT NULL DEFAULT 0,
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (batch_id, injection_no)
);

CREATE TABLE IF NOT EXISTS grouping_uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    filename        TEXT NOT NULL,
    injection_count INTEGER NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS precision_results (
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    group_name      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    status          TEXT NOT NULL
                    CHECK (status IN ('pass', 'fail', 'unverified')),
    group_size      INTEGER NOT NULL,
    sample_count    INTEGER NOT NULL,
    mean_area       REAL,
    stdev_area      REAL,
    rsd_percent     REAL,
    reason          TEXT NOT NULL DEFAULT '',
    note            TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (batch_id, group_name)
);

CREATE INDEX IF NOT EXISTS idx_revisions_batch
    ON injection_revisions(batch_id, injection_no, revision);
CREATE INDEX IF NOT EXISTS idx_upload_log_batch
    ON upload_log(batch_id, id);
"""


def connect(path: str) -> sqlite3.Connection:
    """Open an autocommit connection; transactions are started explicitly."""
    if path != ":memory:":
        Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


@contextmanager
def snapshot(conn: sqlite3.Connection):
    """Read several tables from one consistent SQLite snapshot."""
    conn.execute("BEGIN")
    try:
        yield
    finally:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass


@contextmanager
def immediate(conn: sqlite3.Connection):
    """Serialize a read/write unit across threads and processes."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    else:
        conn.execute("COMMIT")


def init_db(conn: sqlite3.Connection, attempts: int = 5) -> None:
    # Several gunicorn workers may boot at once; switching journal mode needs
    # a brief exclusive lock, so retry instead of failing startup.
    for attempt in range(attempts):
        try:
            if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                conn.execute("PRAGMA journal_mode = WAL")
            break
        except sqlite3.OperationalError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.1 * (attempt + 1))
    conn.executescript(SCHEMA)
    _migrate(conn)

    # A crash or a failed recompute can leave a batch whose stored rows are
    # older than its accepted revisions.  Never let that combination appear
    # as a current result, even before the next page request.
    with immediate(conn):
        conn.execute(
            """
            UPDATE batches
               SET result_state = 'stale',
                   result_version = NULL,
                   result_error = COALESCE(
                       result_error,
                       '服务重启后发现结果版本与数据版本不一致，已作废'
                   )
             WHERE result_state = 'current'
               AND (result_version IS NULL OR result_version != data_version)
            """
        )
        conn.execute(
            """
            DELETE FROM results
             WHERE batch_id IN (
                 SELECT id FROM batches
                  WHERE result_state != 'current'
                     OR result_version IS NULL
                     OR result_version != data_version
             )
            """
        )
        # Precision verdicts depend on both the accepted revisions and the
        # current grouping.  A restart must not resurrect a verdict whose
        # inputs have moved since it was computed.
        conn.execute(
            """
            UPDATE batches
               SET precision_state = 'pending',
                   precision_error = COALESCE(
                       precision_error,
                       '服务重启后发现精密度结论与数据或分组版本不一致，已作废'
                   )
             WHERE precision_state = 'current'
               AND (precision_data_version IS NULL
                    OR precision_data_version != data_version
                    OR precision_group_version IS NULL
                    OR precision_group_version != group_version
                    OR result_state != 'current'
                    OR result_version IS NULL
                    OR result_version != data_version)
            """
        )
        conn.execute(
            """
            DELETE FROM precision_results
             WHERE batch_id IN (
                 SELECT id FROM batches
                  WHERE precision_state != 'current'
                     OR precision_data_version IS NULL
                     OR precision_data_version != data_version
                     OR precision_group_version IS NULL
                     OR precision_group_version != group_version
                     OR result_state != 'current'
                     OR result_version IS NULL
                     OR result_version != data_version
             )
            """
        )


# Columns added after the first release; named volumes survive image upgrades.
_NEW_BATCH_COLUMNS = {
    "rsd_limit": "REAL CHECK (rsd_limit IS NULL OR rsd_limit >= 0)",
    "group_version": "INTEGER NOT NULL DEFAULT 0",
    "precision_state": (
        "TEXT NOT NULL DEFAULT 'none' "
        "CHECK (precision_state IN ('none', 'pending', 'current'))"
    ),
    "precision_data_version": "INTEGER",
    "precision_group_version": "INTEGER",
    "precision_error": "TEXT",
    "precision_computed_at": "TEXT",
}


def _migrate(conn: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(batches)").fetchall()
    }
    missing = [
        (name, ddl) for name, ddl in _NEW_BATCH_COLUMNS.items() if name not in existing
    ]
    if not missing:
        return
    with immediate(conn):
        for name, ddl in missing:
            conn.execute(f"ALTER TABLE batches ADD COLUMN {name} {ddl}")
        # A batch created before precision existed has no grouping; anything
        # that somehow claims a precision verdict must be re-verified.
        conn.execute(
            """
            UPDATE batches
               SET precision_state = 'none',
                   precision_error = NULL,
                   precision_data_version = NULL,
                   precision_group_version = NULL
             WHERE rsd_limit IS NULL
            """
        )
