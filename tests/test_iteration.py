"""Tests for cipherlab.iteration — MetricsSummary, IterationRecord, IterationHistory."""

import json

import pytest

from cipherlab.iteration import (
    IterationHistory,
    IterationRecord,
    MetricsSummary,
    extract_metrics_summary,
)


# ---------------------------------------------------------------------------
# MetricsSummary
# ---------------------------------------------------------------------------

class TestMetricsSummary:
    def test_delta_both_populated(self):
        before = MetricsSummary(
            pt_avalanche_mean=0.4800,
            key_avalanche_mean=0.5100,
            overall_score=0.8500,
            sac_deviation_pt=0.0300,
            sac_deviation_key=0.0200,
        )
        after = MetricsSummary(
            pt_avalanche_mean=0.5000,
            key_avalanche_mean=0.5050,
            overall_score=0.9200,
            sac_deviation_pt=0.0100,
            sac_deviation_key=0.0150,
        )
        d = after.delta(before)
        assert d["pt_avalanche_mean"] == pytest.approx(0.02, abs=1e-5)
        assert d["overall_score"] == pytest.approx(0.07, abs=1e-5)
        # SAC deviation went down (improvement) — delta is negative
        assert d["sac_deviation_pt"] == pytest.approx(-0.02, abs=1e-5)

    def test_delta_with_none_fields(self):
        before = MetricsSummary(overall_score=0.85)
        after = MetricsSummary(overall_score=0.90, pt_avalanche_mean=0.50)
        d = after.delta(before)
        assert d["overall_score"] == pytest.approx(0.05, abs=1e-5)
        assert d["pt_avalanche_mean"] is None  # before was None

    def test_to_display_dict_filters_nones(self):
        m = MetricsSummary(overall_score=0.92, roundtrip_pass=True, num_heuristic_issues=0)
        d = m.to_display_dict()
        assert "overall_score" in d
        assert "roundtrip_pass" in d
        assert "num_heuristic_issues" in d
        assert "pt_avalanche_mean" not in d  # was None


# ---------------------------------------------------------------------------
# extract_metrics_summary
# ---------------------------------------------------------------------------

class TestExtractMetricsSummary:
    def test_from_evaluate_and_score_output(self):
        metrics = {
            "plaintext_avalanche": {"mean": 0.5022, "std": 0.01},
            "key_avalanche": {"mean": 0.4980, "std": 0.02},
            "scores": {
                "plaintext_avalanche": 0.9956,
                "key_avalanche": 0.9960,
                "overall": 0.9958,
            },
        }
        s = extract_metrics_summary(metrics=metrics, issues=["weak diffusion"])
        assert s.pt_avalanche_mean == pytest.approx(0.5022)
        assert s.key_avalanche_mean == pytest.approx(0.4980)
        assert s.overall_score == pytest.approx(0.9958)
        assert s.num_heuristic_issues == 1

    def test_empty_input(self):
        s = extract_metrics_summary()
        assert s.overall_score is None
        assert s.num_heuristic_issues == 0


# ---------------------------------------------------------------------------
# IterationRecord
# ---------------------------------------------------------------------------

class TestIterationRecord:
    def test_compute_deltas(self):
        rec = IterationRecord(
            iteration_id=0,
            before_metrics=MetricsSummary(overall_score=0.80),
            after_metrics=MetricsSummary(overall_score=0.90),
            model_used="deepseek/deepseek-r1",
        )
        rec.compute_deltas()
        assert rec.metric_deltas is not None
        assert rec.metric_deltas["overall_score"] == pytest.approx(0.10, abs=1e-5)

    def test_summary_line_accepted(self):
        rec = IterationRecord(
            iteration_id=1,
            status="accepted",
            patch_summary="Increase rounds 4->6",
            model_used="deepseek/deepseek-r1",
            metric_deltas={"overall_score": 0.05},
        )
        line = rec.summary_line()
        assert "[+]" in line
        assert "#1" in line
        assert "Increase rounds" in line

    def test_summary_line_rejected(self):
        rec = IterationRecord(
            iteration_id=2,
            status="rejected",
            patch_summary="Swap S-box",
            model_used="gpt-5.2",
        )
        line = rec.summary_line()
        assert "[-]" in line

    def test_serialization_roundtrip(self):
        rec = IterationRecord(
            iteration_id=0,
            before_spec={"name": "test", "rounds": 4},
            after_spec={"name": "test", "rounds": 6},
            patch_summary="bump rounds",
            before_metrics=MetricsSummary(overall_score=0.80),
            after_metrics=MetricsSummary(overall_score=0.90),
            model_used="deepseek/deepseek-r1",
            status="accepted",
            decision_reason="score improved",
            seed=42,
        )
        rec.compute_deltas()

        data = rec.model_dump()
        json_str = json.dumps(data)
        restored = IterationRecord.model_validate(json.loads(json_str))

        assert restored.iteration_id == 0
        assert restored.status == "accepted"
        assert restored.after_spec["rounds"] == 6
        assert restored.metric_deltas["overall_score"] == pytest.approx(0.10, abs=1e-5)
        assert restored.seed == 42


# ---------------------------------------------------------------------------
# IterationHistory
# ---------------------------------------------------------------------------

def _make_record(id_: int, status: str = "accepted", overall_before: float = 0.8, overall_after: float = 0.9) -> IterationRecord:
    rec = IterationRecord(
        iteration_id=id_,
        before_spec={"name": "test", "rounds": 4 + id_},
        after_spec={"name": "test", "rounds": 5 + id_},
        patch_summary=f"change #{id_}",
        before_metrics=MetricsSummary(overall_score=overall_before),
        after_metrics=MetricsSummary(overall_score=overall_after),
        model_used="deepseek/deepseek-r1",
        status=status,
        decision_reason="test",
    )
    rec.compute_deltas()
    return rec


class TestIterationHistory:
    def test_add_and_count(self):
        h = IterationHistory(cipher_name="test_cipher")
        assert h.count == 0
        assert h.next_id == 0
        h.add(_make_record(0))
        assert h.count == 1
        assert h.next_id == 1

    def test_accepted_rejected_pending(self):
        h = IterationHistory(cipher_name="test")
        h.add(_make_record(0, status="accepted"))
        h.add(_make_record(1, status="rejected"))
        h.add(_make_record(2, status="pending"))
        h.add(_make_record(3, status="accepted"))
        assert len(h.accepted()) == 2
        assert len(h.rejected()) == 1
        assert len(h.pending()) == 1

    def test_current_spec_dict(self):
        h = IterationHistory(cipher_name="test")
        assert h.current_spec_dict() is None

        h.add(_make_record(0, status="accepted"))
        h.add(_make_record(1, status="rejected"))
        # Should return after_spec from iteration 0 (last accepted)
        spec = h.current_spec_dict()
        assert spec is not None
        assert spec["rounds"] == 5  # 5 + 0

    def test_current_spec_dict_uses_latest_accepted(self):
        h = IterationHistory(cipher_name="test")
        h.add(_make_record(0, status="accepted"))
        h.add(_make_record(1, status="accepted"))
        spec = h.current_spec_dict()
        assert spec["rounds"] == 6  # 5 + 1

    def test_get(self):
        h = IterationHistory(cipher_name="test")
        h.add(_make_record(0))
        h.add(_make_record(1))
        assert h.get(0).iteration_id == 0
        assert h.get(1).iteration_id == 1
        assert h.get(99) is None

    def test_rollback_spec_dict(self):
        h = IterationHistory(cipher_name="test")
        h.add(_make_record(0, status="accepted"))
        h.add(_make_record(1, status="accepted"))

        # Rollback to iteration 0
        spec = h.rollback_spec_dict(0)
        assert spec is not None
        assert spec["rounds"] == 5

        # Rollback to -1 means revert to original
        assert h.rollback_spec_dict(-1) is None

        # Rollback to nonexistent iteration
        assert h.rollback_spec_dict(99) is None

    def test_rollback_rejected_iteration_returns_none(self):
        h = IterationHistory(cipher_name="test")
        h.add(_make_record(0, status="rejected"))
        assert h.rollback_spec_dict(0) is None

    def test_to_context_summary(self):
        h = IterationHistory(cipher_name="test")
        assert "No improvement" in h.to_context_summary()

        h.add(_make_record(0, status="accepted"))
        summary = h.to_context_summary()
        assert "1 total" in summary
        assert "[+]" in summary

    def test_to_export_dict(self):
        h = IterationHistory(cipher_name="test_cipher")
        h.add(_make_record(0, status="accepted"))
        h.add(_make_record(1, status="rejected"))
        export = h.to_export_dict()
        assert export["cipher_name"] == "test_cipher"
        assert export["total_iterations"] == 2
        assert export["accepted_count"] == 1
        assert export["rejected_count"] == 1
        assert len(export["records"]) == 2

    def test_full_serialization_roundtrip(self):
        h = IterationHistory(cipher_name="test_cipher")
        h.add(_make_record(0, status="accepted"))
        h.add(_make_record(1, status="rejected"))

        json_str = h.model_dump_json()
        restored = IterationHistory.model_validate_json(json_str)

        assert restored.cipher_name == "test_cipher"
        assert restored.count == 2
        assert restored.accepted()[0].iteration_id == 0
        assert restored.rejected()[0].iteration_id == 1
