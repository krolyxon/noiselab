"""Plain-language glossary (app/glossary.py): complete, and served by /api/glossary."""

from app import glossary
from app.enhancers.base import FAMILIES


def test_every_entry_is_complete():
    for group in (glossary.METRICS, glossary.TERMS, glossary.FAMILIES):
        for key, t in group.items():
            assert t.short and t.tech and t.explain, key
            assert t.better in ("higher", "lower", ""), key


def test_families_match_registry():
    assert set(glossary.FAMILIES) == set(FAMILIES)


def test_label_format():
    assert glossary.label("d_si_sdr") == "Speech clarity gain, dB (ΔSI-SDR) ↑ better"
    assert glossary.label("lsd").endswith("↓ better")
    assert glossary.label("snr_in", arrow=True) == "Starting noise level, dB (input SNR)"


def test_api_glossary(client):
    r = client.get("/api/glossary")
    assert r.status_code == 200
    j = r.get_json()
    assert {"metrics", "terms", "families", "params"} <= set(j)
    assert j["metrics"]["pesq"]["tech"] == "PESQ"


def test_every_algorithm_has_a_plain_description(client):
    algos = client.get("/api/algorithms").get_json()["algorithms"]
    for a in algos:
        assert a["plain"], a["id"]
