"""Flask routes for uploading batches and reviewing current carryover results."""

from __future__ import annotations

import csv
import io
import math
import os
import sqlite3
from pathlib import Path

from flask import (
    Flask,
    Response,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge

from . import db, store
from .engine import KIND_LABELS, PRECISION_STATUS_LABELS, STATUS_LABELS
from .ingest import CsvError, MAX_FILE_BYTES, parse_csv, parse_grouping_csv


STATE_LABELS = {
    "pending": "待重算",
    "current": "现行结果",
    "stale": "结果已作废",
}

PRECISION_STATE_LABELS = {
    "none": "未设定",
    "pending": "待核查",
    "current": "现行结论",
}


def _parse_limit(raw: str, label: str) -> float:
    text = (raw or "").strip()
    if not text:
        raise ValueError(f"请填写{label}")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{label}必须是数字") from None
    if not math.isfinite(value):
        raise ValueError(f"{label}必须是有限数值")
    if value < 0:
        raise ValueError(f"{label}不能为负数")
    return value


def _read_upload() -> tuple[str, bytes]:
    uploaded = request.files.get("csv_file")
    if uploaded is None or not uploaded.filename:
        raise CsvError("请选择要上传的 CSV 文件")
    data = uploaded.stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise CsvError(f"CSV 文件超过 {MAX_FILE_BYTES // (1024 * 1024)} MB 限制")
    return uploaded.filename, data


def _recompute_safely(batch_id: int) -> bool:
    """Recompute outside the accept transaction; never keep an old verdict."""
    try:
        rows = store.recompute(g.db, batch_id)
    except Exception as exc:  # pragma: no cover - surfaced through tests
        current_app.logger.exception("batch %s recompute failed", batch_id)
        try:
            store.mark_failed(g.db, batch_id, f"重算失败：{exc}")
        except Exception:
            current_app.logger.exception("batch %s could not be marked stale", batch_id)
        flash("更正已受理，但重算失败；旧判定已作废。", "error")
        return False
    flash(f"重算完成，共 {len(rows)} 条现行结果。", "success")
    # Carryover is the input of the precision check; recompute it whenever it
    # was configured and the carryover verdict is now publishable.
    _recompute_precision_safely(batch_id)
    return True


def _recompute_precision_safely(batch_id: int, flash_result: bool = True) -> bool:
    """Recompute RSD verdicts; a failure leaves them pending, never stale."""
    conn = g.db
    with db.snapshot(conn):
        batch = store.get_batch(conn, batch_id)
    if batch is None or batch["precision_state"] == "none":
        return False
    try:
        rows = store.recompute_precision(conn, batch_id)
    except Exception as exc:  # pragma: no cover - surfaced through tests
        current_app.logger.exception("batch %s precision failed", batch_id)
        try:
            store.mark_precision_failed(
                conn, batch_id, f"精密度核查失败：{exc}"
            )
        except Exception:
            current_app.logger.exception(
                "batch %s could not be marked precision-pending", batch_id
            )
        flash("精密度核查失败，结论保持待核；可稍后重试。", "error")
        return False
    if not rows:
        flash("残留判定尚未完成，精密度结论保持待核；请先完成重算。", "warning")
        return False
    if flash_result:
        failed = sum(1 for row in rows if row.status != "pass")
        if failed:
            flash(f"精密度核查完成：{len(rows)} 组中 {failed} 组无法判定或超限。", "warning")
        else:
            flash(f"精密度核查完成：{len(rows)} 组均合格。", "success")
    return True


def create_app(database_path: str | None = None, testing: bool = False) -> Flask:
    app = Flask(__name__)
    app.config.update(
        # An explicit argument (tests) wins over the environment variable.
        DATABASE=database_path
        or os.environ.get("DATABASE_PATH")
        or str(Path(app.root_path).parent / "data" / "carryover.sqlite3"),
        MAX_CONTENT_LENGTH=MAX_FILE_BYTES + 512 * 1024,
        SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-change-me"),
        TESTING=testing,
        JSON_SORT_KEYS=False,
    )

    initial = db.connect(app.config["DATABASE"])
    try:
        db.init_db(initial)
    finally:
        initial.close()

    def get_db() -> sqlite3.Connection:
        if "db" not in g:
            g.db = db.connect(app.config["DATABASE"])
        return g.db

    app.get_db = get_db  # type: ignore[attr-defined]

    @app.teardown_appcontext
    def close_db(_exc: BaseException | None = None) -> None:
        conn = g.pop("db", None)
        if conn is not None:
            conn.close()

    @app.get("/")
    def index():
        conn = get_db()
        batches = store.list_batches(conn)
        return render_template("index.html", batches=batches)

    @app.post("/batches")
    def create_batch_route():
        name = (request.form.get("name") or "").strip()
        try:
            if not name:
                raise ValueError("请填写批次名称")
            if len(name) > 100:
                raise ValueError("批次名称不能超过 100 个字符")
            high_threshold = _parse_limit(request.form.get("high_threshold"), "高值阈值")
            blank_limit = _parse_limit(request.form.get("blank_limit"), "空白上限")
            filename, data = _read_upload()
            rows = parse_csv(data)
        except (ValueError, CsvError) as exc:
            messages = exc.messages if isinstance(exc, CsvError) else [str(exc)]
            for message in messages:
                flash(message, "error")
            return render_template(
                "index.html",
                batches=store.list_batches(get_db()),
                form=request.form,
            ), 400

        batch_id = store.create_batch(
            get_db(), name, high_threshold, blank_limit, rows, filename
        )
        flash(
            f"批次已创建，接收 {len(rows)} 个进样（同一文件内同号取最高修订）。",
            "success",
        )
        _recompute_safely(batch_id)
        return redirect(url_for("batch_detail", batch_id=batch_id))

    @app.get("/batches/<int:batch_id>")
    def batch_detail(batch_id: int):
        conn = get_db()
        # inspect_results refuses to hand back "current" rows that do not match
        # the accepted revisions; it invalidates them atomically if needed.
        batch, results = store.inspect_results(conn, batch_id)
        if batch is None:
            return render_template("not_found.html"), 404
        batch, precision = store.inspect_precision(conn, batch_id)
        if batch is None:  # pragma: no cover - defensive
            return render_template("not_found.html"), 404
        with db.snapshot(conn):
            revisions = store.get_revisions(conn, batch_id)
            uploads = store.get_upload_log(conn, batch_id)
            groups = store.get_grouping(conn, batch_id)
            grouping_uploads = store.get_grouping_uploads(conn, batch_id)
            grouped_counts: dict[str, int] = {}
            for group in groups:
                grouped_counts[group.group] = len(group.injection_nos)
        return render_template(
            "batch.html",
            batch=batch,
            results=results,
            precision=precision,
            revisions=revisions,
            uploads=uploads,
            groups=groups,
            grouped_counts=grouped_counts,
            grouping_uploads=grouping_uploads,
            state_labels=STATE_LABELS,
            precision_state_labels=PRECISION_STATE_LABELS,
            status_labels=STATUS_LABELS,
            precision_status_labels=PRECISION_STATUS_LABELS,
            kind_labels=KIND_LABELS,
            unavailable_reason=_unavailable_reason(batch, results),
            precision_unavailable_reason=_precision_unavailable_reason(batch, precision),
        )

    @app.post("/batches/<int:batch_id>/upload")
    def upload_correction(batch_id: int):
        conn = get_db()
        if store.get_batch(conn, batch_id) is None:
            return render_template("not_found.html"), 404

        try:
            filename, data = _read_upload()
            rows = parse_csv(data)
        except (ValueError, CsvError) as exc:
            messages = exc.messages if isinstance(exc, CsvError) else [str(exc)]
            for message in messages:
                flash(message, "error")
            return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

        try:
            summary = store.accept_upload(conn, batch_id, rows, filename)
        except store.StoreError as exc:
            flash(str(exc), "error")
            return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

        if summary.skipped:
            flash(
                f"已跳过 {summary.skipped} 行：现有修订号不低于上传值，只接受更高修订。",
                "warning",
            )
        if not summary.changed:
            flash("没有可受理的新修订，现行结果保持不变。", "warning")
            return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

        flash(f"已受理 {summary.accepted} 行更正，数据版本更新为 {summary.data_version}。", "success")
        # Any previously current rows were removed in the same transaction as
        # the correction, so a crash before this recompute cannot resurrect them.
        _recompute_safely(batch_id)
        return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

    @app.post("/batches/<int:batch_id>/recompute")
    def recompute_route(batch_id: int):
        conn = get_db()
        with db.snapshot(conn):
            if store.get_batch(conn, batch_id) is None:
                return render_template("not_found.html"), 404
        _recompute_safely(batch_id)
        return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

    @app.post("/batches/<int:batch_id>/grouping")
    def upload_grouping(batch_id: int):
        conn = get_db()
        if store.get_batch(conn, batch_id) is None:
            return render_template("not_found.html"), 404

        try:
            rsd_limit = _parse_limit(request.form.get("rsd_limit"), "RSD 上限")
            filename, data = _read_upload()
            rows = parse_grouping_csv(data)
        except (ValueError, CsvError) as exc:
            messages = exc.messages if isinstance(exc, CsvError) else [str(exc)]
            for message in messages:
                flash(message, "error")
            return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

        try:
            summary = store.accept_grouping(conn, batch_id, rows, rsd_limit, filename)
        except store.StoreError as exc:
            flash(str(exc), "error")
            return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

        flash(
            f"分组已保存：{summary.group_count} 组、{summary.injection_count} 支进样；"
            f"RSD 上限 {rsd_limit:g}%。旧精密度结论已失效。",
            "success",
        )
        if summary.unknown:
            flash(
                "以下进样号不在批次中，已记录但暂时无法核查："
                + "、".join(summary.unknown[:20]),
                "warning",
            )
        # A concurrently accepted correction may still be recomputing; the
        # precision pass below only publishes if carryover is current.
        _recompute_precision_safely(batch_id)
        return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

    @app.post("/batches/<int:batch_id>/precision/recompute")
    def recompute_precision_route(batch_id: int):
        conn = get_db()
        with db.snapshot(conn):
            if store.get_batch(conn, batch_id) is None:
                return render_template("not_found.html"), 404
        _recompute_precision_safely(batch_id)
        return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

    @app.get("/batches/<int:batch_id>/results.csv")
    def download_results(batch_id: int):
        conn = get_db()
        batch, results = store.inspect_results(conn, batch_id)
        if batch is None:
            return render_template("not_found.html"), 404
        if results is None:
            flash("当前没有可信的现行结果，不能下载；请先完成重算。", "error")
            return redirect(url_for("batch_detail", batch_id=batch_id), code=303)

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\r\n")
        writer.writerow(["进样号", "类型", "峰面积", "修订号", "判定", "触发进样", "说明"])
        for row in results:
            writer.writerow(
                [
                    _csv_safe(row["injection_no"]),
                    KIND_LABELS.get(row["kind"], row["kind"]),
                    "" if row["peak_area"] is None else f"{row['peak_area']:.6g}",
                    row["revision"],
                    STATUS_LABELS.get(row["status"], row["status"]),
                    _csv_safe(row["trigger_injection"] or ""),
                    _csv_safe(row["note"]),
                ]
            )
        payload = "﻿" + buffer.getvalue()
        filename = f"batch_{batch_id}_current_results.csv"
        return Response(
            payload.encode("utf-8"),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
            },
        )

    @app.get("/healthz")
    def healthz():
        get_db().execute("SELECT 1")
        return {"status": "ok"}

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        flash("上传内容过大。", "error")
        return redirect(url_for("index"), code=303)

    return app


def _csv_safe(value: str) -> str:
    text = str(value)
    # Prevent spreadsheet formula execution in exported notes and IDs.
    if text.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + text
    return text


def _unavailable_reason(batch: sqlite3.Row, results) -> str | None:
    if results is not None:
        return None
    state = batch["result_state"]
    if state == "pending":
        return "更正已受理或批次尚未计算，目前没有可信的现行结果。"
    if batch["result_error"]:
        return f"结果已作废：{batch['result_error']}"
    return "结果已作废，需重新计算后才能查看和下载现行结果。"


def _precision_unavailable_reason(batch: sqlite3.Row, precision) -> str | None:
    if precision is not None:
        return None
    if batch["precision_state"] == "none":
        return "尚未上传分组并设定 RSD 上限。"
    if batch["precision_error"]:
        return f"精密度结论已失效：{batch['precision_error']}"
    return "精密度核查尚未完成，请完成残留重算后重试。"
