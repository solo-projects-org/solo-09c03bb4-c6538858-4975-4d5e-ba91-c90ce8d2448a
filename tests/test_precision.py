import io
import sqlite3
import threading
import time

import pytest

from carryover import create_app, engine as engine_module, store
from carryover.engine import (
    PRECISION_FAIL,
    PRECISION_PASS,
    PRECISION_UNVERIFIED,
    PrecisionGroup,
    compute_precision,
)
from carryover.ingest import CsvError, parse_grouping_csv
from tests.conftest import create, csv_bytes, upload


GROUP_HEADER = "样品号,进样号\n"


def grouping(*rows):
    return (GROUP_HEADER + "".join(row + "\n" for row in rows)).encode()


def upload_grouping(client, payload, rsd="5", batch_id=1):
    return client.post(
        f"/batches/{batch_id}/grouping",
        data={
            "rsd_limit": rsd,
            "csv_file": (io.BytesIO(payload), "groups.csv"),
        },
        content_type="multipart/form-data",
        follow_redirects=False,
    )


def page(client, batch_id=1):
    return client.get(f"/batches/{batch_id}").get_data(as_text=True)


def precision_rows(text):
    import re

    section = text.split("<h2>精密度核查")[1].split("</section>")[0]
    rows = {}
    for status, body in re.findall(
        r'<tr class="precision-(\w+)">(.*?)</tr>', section, re.S
    ):
        cells = [
            re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", cell)).strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        ]
        rows[cells[0]] = (status, cells)
    return rows


def test_precision_needs_two_current_samples():
    injections = [
        engine_module.Injection("S1", "sample", 100.0, 1),
        engine_module.Injection("S2", "sample", 102.0, 1),
        engine_module.Injection("S3", "sample", 50.0, 1),
    ]
    results = compute_precision(
        [PrecisionGroup("A", ("S1", "S2")), PrecisionGroup("B", ("S3",))],
        injections,
        [],
        rsd_limit=5,
    )
    by_group = {row.group: row for row in results}
    assert by_group["A"].status == PRECISION_PASS
    assert by_group["A"].mean == pytest.approx(101.0)
    assert by_group["A"].stdev == pytest.approx(2 ** 0.5)
    assert by_group["A"].rsd == pytest.approx(2 ** 0.5 / 101 * 100)
    assert by_group["B"].status == PRECISION_UNVERIFIED
    assert by_group["B"].reason == "too_few"


def test_precision_marks_risky_group_unverifiable():
    injections = [
        engine_module.Injection("S1", "sample", 100.0, 1),
        engine_module.Injection("S2", "sample", 100.0, 1),
    ]
    carryover = [
        engine_module.ResultRow("S1", "sample", 100.0, 1, "carryover_risk", "H", ""),
        engine_module.ResultRow("S2", "sample", 100.0, 1, "clear", None, ""),
    ]
    rows = compute_precision([PrecisionGroup("A", ("S1", "S2"))], injections, carryover, 5)
    assert rows[0].status == PRECISION_UNVERIFIED
    assert rows[0].reason == "carryover"


def test_precision_zero_mean_and_missing_injection():
    injections = [
        engine_module.Injection("S1", "sample", 0.0, 1),
        engine_module.Injection("S2", "sample", 0.0, 1),
    ]
    rows = compute_precision(
        [PrecisionGroup("Z", ("S1", "S2")), PrecisionGroup("M", ("S1", "NX"))],
        injections,
        [],
        10,
    )
    by_group = {row.group: row for row in rows}
    assert by_group["Z"].status == PRECISION_UNVERIFIED
    assert by_group["Z"].reason == "zero_mean"
    assert by_group["M"].status == PRECISION_UNVERIFIED
    assert "NX" in by_group["M"].note


def test_grouping_csv_rejects_same_injection_in_two_groups():
    with pytest.raises(CsvError) as exc:
        parse_grouping_csv(grouping("A,S1", "B,S1"))
    assert "一支进样只能归属一组" in str(exc.value)


def test_grouping_csv_dedupes_identical_rows():
    rows = parse_grouping_csv(grouping("A,S1", "A,S1", "A,S2"))
    assert [(row.group, row.injection_no) for row in rows] == [
        ("A", "S1"),
        ("A", "S2"),
    ]


def test_precision_page_computes_and_downloads_carryover_still_works(client):
    create(client)
    response = upload_grouping(client, grouping("A,S1", "A,S2", "A,S3"))
    assert response.status_code == 303
    text = page(client)
    rows = precision_rows(text)
    # S2 is a carryover risk in the default fixture, so group A is unverifiable.
    assert rows["A"][0] == "unverified"
    # Existing carryover results and download are untouched.
    assert "残留风险" in text
    assert client.get("/batches/1/results.csv").status_code == 200


def test_precision_pass_and_fail_from_current_revisions(client):
    create(client)
    payload = csv_bytes(
        "S1,sample,100,2",
        "S2,sample,100,2",
        "N1,sample,100,1",
        "N2,sample,110,1",
    )
    upload(client, "/batches/1/upload", "fix.csv", payload)
    response = upload_grouping(
        client, grouping("A,S1", "A,S2", "B,N1", "B,N2"), rsd="5"
    )
    assert response.status_code == 303
    rows = precision_rows(page(client))
    assert rows["A"][0] == "pass"
    assert rows["A"][1][5] == "0.00%"
    assert rows["B"][0] == "fail"


def test_correction_invalidates_precision_immediately(client, monkeypatch):
    create(client)
    payload = csv_bytes("S1,sample,100,2", "S2,sample,100,2")
    upload(client, "/batches/1/upload", "fix.csv", payload)
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")
    assert precision_rows(page(client))["A"][0] == "pass"

    # Break the carryover recompute: the correction is accepted, so both the
    # carryover verdict and the precision verdict must disappear.
    original = engine_module.compute_results

    def boom(*_args, **_kwargs):
        raise RuntimeError("模拟计算失败")

    monkeypatch.setattr(engine_module, "compute_results", boom)
    upload(client, "/batches/1/upload", "fix2.csv", csv_bytes("S2,sample,120,3"))
    monkeypatch.setattr(engine_module, "compute_results", original)

    text = page(client)
    assert precision_rows(text) == {}
    assert "待核查" in text
    assert "结果已作废" in text


def test_grouping_change_invalidates_precision(client):
    create(client)
    payload = csv_bytes("S1,sample,100,2", "S2,sample,100,2")
    upload(client, "/batches/1/upload", "fix.csv", payload)
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")
    assert precision_rows(page(client))["A"][0] == "pass"

    upload_grouping(client, grouping("A,S1", "B,S2"), rsd="5")
    rows = precision_rows(page(client))
    assert rows["A"][0] == "unverified"
    assert rows["B"][0] == "unverified"


def test_restart_does_not_resurrect_precision(tmp_path):
    path = tmp_path / "restart_precision.sqlite3"
    app = create_app(str(path), testing=True)
    client = app.test_client()
    create(client)
    upload(client, "/batches/1/upload", "fix.csv", csv_bytes("S1,sample,100,2", "S2,sample,100,2"))
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")
    assert precision_rows(page(client))["A"][0] == "pass"

    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("UPDATE batches SET group_version = group_version + 1 WHERE id = 1")
    conn.close()

    restarted = create_app(str(path), testing=True)
    text = precision_rows(restarted.test_client().get("/batches/1").get_data(as_text=True))
    assert text == {}


def test_precision_failure_keeps_pending_and_retry_works(client, monkeypatch):
    create(client)
    upload(client, "/batches/1/upload", "fix.csv", csv_bytes("S1,sample,100,2", "S2,sample,100,2"))
    original = engine_module.compute_precision

    def boom(*_args, **_kwargs):
        raise RuntimeError("模拟精密度失败")

    monkeypatch.setattr(engine_module, "compute_precision", boom)
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")
    text = page(client)
    assert precision_rows(text) == {}
    assert "待核查" in text
    assert "精密度核查失败" in text
    # The carryover verdict is still current and downloadable.
    assert "现行结果" in text
    assert client.get("/batches/1/results.csv").status_code == 200

    monkeypatch.setattr(engine_module, "compute_precision", original)
    response = client.post("/batches/1/precision/recompute")
    assert response.status_code == 303
    assert precision_rows(page(client))["A"][0] == "pass"


def test_precision_not_published_while_carryover_pending(client, monkeypatch):
    create(client)
    payload = csv_bytes("S1,sample,100,2", "S2,sample,100,2")
    upload(client, "/batches/1/upload", "fix.csv", payload)
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")

    original = engine_module.compute_results

    def boom(*_args, **_kwargs):
        raise RuntimeError("模拟计算失败")

    monkeypatch.setattr(engine_module, "compute_results", boom)
    upload(client, "/batches/1/upload", "fix2.csv", csv_bytes("S2,sample,120,3"))
    monkeypatch.setattr(engine_module, "compute_results", original)
    assert precision_rows(page(client)) == {}

    # Grouping is still current, but carryover is stale: a precision-only
    # retry must keep the verdict pending rather than publish anything.
    client.post("/batches/1/precision/recompute")
    assert precision_rows(page(client)) == {}
    assert "待核查" in page(client)


def test_precision_download_and_carryover_page_unaffected(client):
    create(client)
    upload(client, "/batches/1/upload", "fix.csv", csv_bytes("S1,sample,100,2", "S2,sample,100,2"))
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")
    download = client.get("/batches/1/results.csv")
    assert download.status_code == 200
    assert "S1,样品,100,2,未标记" in download.get_data(as_text=True)


def test_concurrent_uploads_recompute_and_grouping_stay_consistent(app):
    create(app.test_client())
    errors = []

    def worker(seed):
        import random

        rnd = random.Random(seed)
        c = app.test_client()
        for index in range(4):
            rev = 2 + index
            data = csv_bytes(f"S2,sample,{100 + rnd.randint(0, 50)},{rev}")
            try:
                c.post(
                    "/batches/1/upload",
                    data={"csv_file": (io.BytesIO(data), "f.csv")},
                    content_type="multipart/form-data",
                )
                c.post(
                    "/batches/1/grouping",
                    data={
                        "rsd_limit": "5",
                        "csv_file": (
                            io.BytesIO(grouping("A,S1", "A,S2", "B,S3")),
                            "g.csv",
                        ),
                    },
                    content_type="multipart/form-data",
                )
                if rnd.random() < 0.5:
                    c.post("/batches/1/recompute")
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not errors

    conn = sqlite3.connect(app.config["DATABASE"], isolation_level=None)
    batch = conn.execute(
        "SELECT result_state, result_version, data_version, precision_state,"
        " precision_data_version, precision_group_version, group_version"
        " FROM batches WHERE id = 1"
    ).fetchone()
    precision = conn.execute(
        "SELECT COUNT(*) FROM precision_results WHERE batch_id = 1"
    ).fetchone()[0]
    conn.close()
    if batch[0] == "current":
        assert batch[1] == batch[2]
        if batch[3] == "current":
            assert batch[4] == batch[2]
            assert batch[5] == batch[6]
            assert precision == 2
        else:
            assert precision == 0
    else:
        assert precision == 0
    assert app.test_client().get("/batches/1").status_code == 200


def test_page_hides_forged_precision_rows(client):
    create(client)
    upload(client, "/batches/1/upload", "fix.csv", csv_bytes("S1,sample,100,2", "S2,sample,100,2"))
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")
    assert precision_rows(page(client))["A"][0] == "pass"

    conn = sqlite3.connect(client.application.config["DATABASE"], isolation_level=None)
    conn.execute("UPDATE precision_results SET rsd_percent = 0.01")
    conn.close()
    text = page(client)
    assert precision_rows(text) == {}
    assert "已作废" in text


def test_precision_group_ignores_listed_blanks(client):
    create(client)
    # B1 is a blank with area 10; the only two samples are one high and one
    # clear, but the blank must not count as a repeat observation.
    response = upload_grouping(client, grouping("A,S2", "A,B1"), rsd="5")
    assert response.status_code == 303
    rows = precision_rows(page(client))
    assert rows["A"][0] == "unverified"
    assert "不足两支" in rows["A"][1][7]


def test_grouping_unknown_injection_is_reported_and_unverifiable(client):
    create(client)
    upload_grouping(client, grouping("A,S1", "A,S2", "A,GHOST"), rsd="5")
    text = page(client)
    assert "不在批次中" in text
    rows = precision_rows(text)
    assert rows["A"][0] == "unverified"
    assert "GHOST" in rows["A"][1][7]


def test_grouping_requires_rsd_limit(client):
    create(client)
    response = upload_grouping(client, grouping("A,S1", "A,S2"), rsd="")
    assert response.status_code == 303
    assert "请填写RSD 上限" in page(client)


def test_restart_invalidates_precision_when_carryover_not_current(tmp_path):
    path = tmp_path / "restart_carryover.sqlite3"
    app = create_app(str(path), testing=True)
    client = app.test_client()
    create(client)
    upload(client, "/batches/1/upload", "fix.csv", csv_bytes("S1,sample,100,2", "S2,sample,100,2"))
    upload_grouping(client, grouping("A,S1", "A,S2"), rsd="5")
    assert precision_rows(page(client))["A"][0] == "pass"

    # Simulate an accepted correction whose recompute never ran: rows removed,
    # data version bumped, but a precision stamp still claims to be current.
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("DELETE FROM results WHERE batch_id = 1")
    conn.execute(
        "UPDATE batches SET data_version = data_version + 1,"
        " result_state = 'pending', result_version = NULL WHERE id = 1"
    )
    conn.close()

    restarted = create_app(str(path), testing=True)
    text = restarted.test_client().get("/batches/1").get_data(as_text=True)
    assert precision_rows(text) == {}
    assert "待核查" in text
