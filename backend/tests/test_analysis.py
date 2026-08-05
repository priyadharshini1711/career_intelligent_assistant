"""Skill extraction and fit scoring.

The fit score is the number a user will screenshot and act on, so these tests
care less about exact values than about the properties that make the number
trustworthy: it is reproducible, it separates a good match from a bad one, and
it never claims evidence that is not there.
"""

import pytest

from app.analysis.fit import build_fit_report
from app.analysis.skills import (
    estimate_resume_years,
    extract_job_requirements,
    find_skills,
    match_skills,
    required_years,
)


class TestSkillDictionary:
    def test_matches_canonical_names_and_aliases(self):
        found = find_skills("Built services in Python with Postgres and k8s")
        assert {"Python", "PostgreSQL", "Kubernetes"} <= set(found)

    def test_handles_symbol_bearing_names(self):
        """`\\b` treats + and # as boundaries, so C++ matches as bare C."""
        found = find_skills("Languages: C++, C#, Node.js", section="Skills")
        assert "C++" in found
        assert "C#" in found
        assert "Node.js" in found

    def test_prefers_the_longer_alias(self):
        found = find_skills("Built mobile apps with React Native", section="Skills")
        assert "React Native" in found

    def test_ambiguous_names_need_a_skills_context(self):
        # "go to market" must not register as Golang.
        assert "Go" not in find_skills("Owned the go to market plan for the launch")
        assert "Go" in find_skills("Languages: Go, Python, Rust", section="Skills")

    def test_unambiguous_alias_matches_anywhere(self):
        assert "Go" in find_skills("Wrote the ingestion service in Golang")

    def test_r_does_not_match_ordinary_prose(self):
        assert "R" not in find_skills("Responsible for reporting and analysis")


class TestRequirementExtraction:
    def test_pulls_requirements_with_importance(self, backend_job_doc):
        requirements = extract_job_requirements(backend_job_doc.chunks)
        assert requirements
        assert {"required", "preferred"} <= {r.importance for r in requirements}

    def test_ignores_non_requirement_sections(self, backend_job_doc):
        requirements = extract_job_requirements(backend_job_doc.chunks)
        assert all(r.section != "Benefits" for r in requirements)

    def test_deduplicates_repeated_lines(self, backend_job_doc):
        requirements = extract_job_requirements(backend_job_doc.chunks)
        texts = [r.text.lower() for r in requirements]
        assert len(texts) == len(set(texts))


class TestYearsExtraction:
    def test_takes_the_lowest_stated_requirement(self):
        # "5+ years overall, 2+ with Kubernetes" -- 2 is the bar that gates you.
        assert required_years("5+ years of experience. 2+ years with Kubernetes.") == 2

    def test_returns_none_when_unstated(self):
        assert required_years("We want a strong engineer.") is None

    def test_prefers_an_explicit_resume_claim(self, resume_doc):
        # The resume says "5 years"; the date span implies 8.
        assert estimate_resume_years(resume_doc.text) == 5.0

    def test_falls_back_to_a_date_span(self):
        assert estimate_resume_years("Engineer at Acme 2016 - 2021. Various projects.") == 5.0

    def test_returns_none_without_signal(self):
        assert estimate_resume_years("Enthusiastic developer seeking work.") is None


class TestSkillMatching:
    def test_separates_matched_partial_and_missing(self, backend_job_doc, resume_doc, embedder):
        result = match_skills(backend_job_doc.chunks, resume_doc.chunks, embedder)
        matched = {evidence.skill for evidence in result.matched}
        missing = {gap.skill for gap in result.missing}

        assert "Python" in matched
        assert "PostgreSQL" in matched
        assert "Apache Kafka" in missing  # in the JD, absent from the resume
        assert not matched & missing

    def test_matched_skills_carry_resume_evidence(self, backend_job_doc, resume_doc, embedder):
        """Every claimed match must point at a line the user can check."""
        result = match_skills(backend_job_doc.chunks, resume_doc.chunks, embedder)
        assert all(evidence.resume_snippet for evidence in result.matched)

    def test_counts_are_consistent_with_the_lists(self, backend_job_doc, resume_doc, embedder):
        result = match_skills(backend_job_doc.chunks, resume_doc.chunks, embedder)
        # Regression: reporting len(matched) against required_total mixed
        # preferred skills in and produced "11 of 10 matched".
        assert result.required_matched <= result.required_total
        assert result.preferred_matched <= result.preferred_total
        assert 0.0 <= result.required_covered <= 1.0

    def test_unrelated_resume_matches_almost_nothing(self, backend_job_doc, embedder):
        from app.schemas import DocumentKind
        from tests.conftest import make_document

        chef = make_document(
            "chef",
            DocumentKind.RESUME,
            "Resume",
            "SKILLS\nViennoiserie, chocolate tempering, plated desserts, menu costing\n"
            "EXPERIENCE\n- Head pastry chef running a brigade of six in a 90-cover restaurant.\n",
        )
        result = match_skills(backend_job_doc.chunks, chef.chunks, embedder)
        assert len(result.missing) > len(result.matched)


class TestFitReport:
    def test_is_reproducible(self, backend_job_doc, resume_doc, embedder):
        """A score that moves between identical runs is worthless."""
        first = build_fit_report(backend_job_doc, resume_doc, embedder)
        second = build_fit_report(backend_job_doc, resume_doc, embedder)
        assert first.overall_score == second.overall_score

    def test_ranks_a_close_role_above_a_distant_one(
        self, backend_job_doc, ml_job_doc, resume_doc, embedder
    ):
        backend = build_fit_report(backend_job_doc, resume_doc, embedder)
        ml = build_fit_report(ml_job_doc, resume_doc, embedder)
        assert backend.overall_score > ml.overall_score

    def test_score_stays_in_range(self, platform_job_doc, resume_doc, embedder):
        report = build_fit_report(platform_job_doc, resume_doc, embedder)
        assert 0.0 <= report.overall_score <= 100.0
        assert report.verdict

    def test_every_component_is_explained(self, backend_job_doc, resume_doc, embedder):
        report = build_fit_report(backend_job_doc, resume_doc, embedder)
        assert report.components
        assert all(component.explanation for component in report.components)

    def test_applied_weights_sum_to_one(self, backend_job_doc, resume_doc, embedder):
        report = build_fit_report(backend_job_doc, resume_doc, embedder)
        applied = sum(c.weight for c in report.components if c.weight > 0)
        assert applied == pytest.approx(1.0, abs=0.01)

    def test_missing_seniority_is_surfaced_at_zero_weight(self, resume_doc, embedder):
        """A dropped component should be explained, not silently vanish."""
        from app.schemas import DocumentKind
        from tests.conftest import make_document

        job = make_document(
            "j",
            DocumentKind.JOB,
            "Engineer",
            "REQUIREMENTS\n- Strong Python and PostgreSQL skills for backend services.\n"
            "- Comfortable with Docker and continuous delivery pipelines.\n",
        )
        report = build_fit_report(job, resume_doc, embedder)
        seniority = next(c for c in report.components if c.name == "Seniority")
        assert seniority.weight == 0.0
        assert "Excluded" in seniority.explanation
