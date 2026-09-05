"""Tests for SiteCatalogSource and geochem domain components."""
from __future__ import annotations

import pytest

from core.data_source import DataSourceRegistry
from core.expert import ExpertRegistry
from core.pipeline import L1Pipeline
from domains.geochem.samples import GeochemSample, is_high_confidence_cu, STAGE_SCORE
from domains.geochem.sources.site_catalog import SiteCatalogSource

CSV_PATH = "datasets/geochemical/states/sites/gswa_cu_sites.csv"


@pytest.fixture(autouse=True)
def clean_registries():
    DataSourceRegistry.clear()
    ExpertRegistry.clear()
    yield
    DataSourceRegistry.clear()
    ExpertRegistry.clear()


@pytest.fixture
def catalog():
    return SiteCatalogSource(CSV_PATH)


class TestSiteCatalogSource:
    def test_loads_all_sites(self, catalog):
        samples = catalog.load_all_samples()
        assert len(samples) == 3112

    def test_feature_names(self, catalog):
        expected = {
            "commodity_richness", "co_occurrence_with_cu", "stage_score",
            "local_density_5km", "local_density_20km", "pathfinder_count", "is_polymetallic",
        }
        assert set(catalog.feature_names) == expected

    def test_get_features_returns_dict(self, catalog):
        samples = catalog.load_all_samples()
        fv = catalog.get_features(samples[0].id)
        assert fv is not None
        assert all(isinstance(v, float) for v in fv.values())

    def test_missing_id_returns_none(self, catalog):
        assert catalog.get_features("NONEXISTENT_ID_XYZ") is None

    def test_is_available_for(self, catalog):
        samples = catalog.load_all_samples()
        assert catalog.is_available_for(samples[0].id)
        assert not catalog.is_available_for("NONEXISTENT_ID_XYZ")

    def test_cu_instances_count(self, catalog):
        instances = catalog.load_cu_instances()
        # Cu Mine (698) + Cu Deposit (54) = 752
        assert len(instances) == 752

    def test_stage_score_range(self, catalog):
        samples = catalog.load_all_samples()
        for s in samples:
            fv = catalog.get_features(s.id)
            assert 0.0 <= fv["stage_score"] <= 5.0

    def test_density_non_negative(self, catalog):
        samples = catalog.load_all_samples()
        for s in samples[:50]:
            fv = catalog.get_features(s.id)
            assert fv["local_density_5km"] >= 0.0
            assert fv["local_density_20km"] >= fv["local_density_5km"]


class TestHighConfidenceCuRule:
    def test_mine_is_positive(self):
        assert is_high_confidence_cu("Cu Mine")

    def test_deposit_is_positive(self):
        assert is_high_confidence_cu("Cu Deposit")

    def test_prospect_is_negative(self):
        assert not is_high_confidence_cu("Cu Prospect")

    def test_occurrence_is_negative(self):
        assert not is_high_confidence_cu("Cu Occurrence")

    def test_nan_is_negative(self):
        assert not is_high_confidence_cu(float("nan"))  # type: ignore[arg-type]


class TestStageScore:
    def test_operating_is_highest(self):
        assert STAGE_SCORE["Operating"] > STAGE_SCORE["Under Development"]

    def test_undeveloped_is_zero(self):
        assert STAGE_SCORE["Undeveloped"] == 0


class TestPipelineExtensibility:
    """Verify that an Expert requiring an unavailable source is auto-skipped."""

    def test_unavailable_source_skipped(self, catalog):
        from domains.geochem.experts.density import DensityExpert

        DataSourceRegistry.register(catalog)
        ExpertRegistry.register(DensityExpert())

        # Register a stub Expert needing "geophysics" (not registered)
        class StubMagneticExpert:
            name = "stub_magnetic"
            required_sources = ["geophysics"]

            def fit_kb(self, bg): pass
            def score(self, s): return 0.5

        ExpertRegistry.register(StubMagneticExpert())

        pipeline = L1Pipeline()
        samples = catalog.load_all_samples()[:10]
        instances = catalog.load_cu_instances()
        pipeline.fit(samples, instances)

        # Only density should be active; stub_magnetic auto-skipped
        assert "density" in pipeline.active_expert_names
        assert "stub_magnetic" not in pipeline.active_expert_names


class TestL1PipelineSmoke:
    """End-to-end smoke test: pipeline runs without errors on real data."""

    def test_scores_all_samples(self, catalog):
        from domains.geochem.experts.density import DensityExpert
        from domains.geochem.experts.commodity_affinity import CommodityAffinityExpert
        from domains.geochem.experts.stage_anomaly import StageAnomalyExpert

        DataSourceRegistry.register(catalog)
        ExpertRegistry.register(DensityExpert())
        ExpertRegistry.register(CommodityAffinityExpert())
        ExpertRegistry.register(StageAnomalyExpert())

        samples = catalog.load_all_samples()
        instances = catalog.load_cu_instances()

        pipeline = L1Pipeline()
        pipeline.fit(samples, instances)
        results = pipeline.score_all(samples)

        assert len(results) == len(samples)
        for r in results:
            assert 0.0 <= r.g_score <= 1.0
