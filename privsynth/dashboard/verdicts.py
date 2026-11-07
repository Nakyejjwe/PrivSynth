"""Turn metrics into plain-language verdicts for decision makers.

A data protection officer needs "is this safe to share, and is it useful?"
rather than a table of numbers. Each function here reads results from the
privacy and evaluation modules and returns a :class:`Verdict`: a level,
a short label and one or two sentences of explanation.

The thresholds are rules of thumb chosen from the Week 3 experiments on
UCI Adult (see ``results/week3_results.txt``): releasing the real data
scored inference risk 0.136 and closer-to-train share 0.62, while every
differentially private run stayed at or below 0.04 and 0.54. They are a
guide for a first decision, not a legal standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from privsynth.evaluation.suite import EvaluationReport
from privsynth.privacy.risk import RiskReport

#: Level -> (icon, colour). The icon always accompanies the colour so the
#: meaning never relies on colour alone.
LEVELS = {
    "good": ("✅", "#0ca30c"),
    "warning": ("⚠️", "#fab219"),
    "critical": ("⛔", "#d03b3b"),
    "unknown": ("ℹ️", "#52514e"),
}


@dataclass(frozen=True)
class Verdict:
    """A plain-language judgement.

    Attributes:
        level: ``"good"``, ``"warning"``, ``"critical"`` or ``"unknown"``.
        label: A few words, e.g. ``"High re-identification risk"``.
        message: One or two sentences explaining the label.
    """

    level: str
    label: str
    message: str

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"Unknown verdict level {self.level!r}.")

    @property
    def icon(self) -> str:
        return LEVELS[self.level][0]

    @property
    def color(self) -> str:
        return LEVELS[self.level][1]

    def __str__(self) -> str:
        return f"{self.icon} {self.label}: {self.message}"


def assess_original_risk(report: RiskReport) -> Verdict:
    """How exposed are the people in the original data?"""
    unique, at_risk = report.pct_unique, report.pct_at_risk
    qis = ", ".join(report.quasi_identifiers)
    if unique >= 1 or at_risk >= 10:
        return Verdict(
            "critical", "High re-identification risk",
            f"{report.n_unique:,} people ({unique:.1f}%) are the only person with their combination of "
            f"{qis}, and {at_risk:.1f}% share it with fewer than {report.k_target - 1} others. "
            "Sharing this table, even without names, could expose them.",
        )
    if at_risk > 0:
        return Verdict(
            "warning", "Some re-identification risk",
            f"{report.n_at_risk:,} records ({at_risk:.1f}%) are in groups smaller than k = {report.k_target} "
            f"on {qis}.",
        )
    return Verdict(
        "good", f"k-anonymous (k = {report.k_achieved})",
        f"Every combination of {qis} is shared by at least {report.k_achieved} people. Note that "
        "k-anonymity alone does not protect against all attacks.",
    )


#: Metric -> (plain meaning, which value is better). Used for the dashboard table.
METRIC_GUIDE: dict[str, tuple[str, str]] = {
    "tvd_1way_mean": ("Average difference in single-column distributions", "lower"),
    "tvd_1way_max": ("Largest difference for any single column", "lower"),
    "tvd_2way_mean": ("Average difference for pairs of columns", "lower"),
    "tvd_2way_max": ("Largest difference for any pair of columns", "lower"),
    "assoc_diff_mean": ("Average change in how strongly columns are related", "lower"),
    "assoc_diff_max": ("Largest change in how strongly two columns are related", "lower"),
    "accuracy_real": ("Accuracy of a model trained on real data (reference)", "reference"),
    "f1_real": ("F1 score of a model trained on real data (reference)", "reference"),
    "auc_real": ("AUC of a model trained on real data (reference)", "reference"),
    "accuracy_synthetic": ("Accuracy on real people of a model trained on synthetic data", "higher"),
    "f1_synthetic": ("F1 score on real people of a model trained on synthetic data", "higher"),
    "auc_synthetic": ("AUC on real people of a model trained on synthetic data", "higher"),
    "accuracy_majority": ("Accuracy of always guessing the most common value (reference)", "reference"),
    "utility_ratio": ("Synthetic model's score as a share of the real model's", "higher"),
    "dcr_synthetic_median": ("Typical distance from a synthetic record to the nearest real one", "higher"),
    "dcr_holdout_median": ("Typical distance from a new real person to the nearest real one (reference)", "reference"),
    "dcr_ratio": ("The two distances above, divided (below 1 = suspiciously close)", "1 or more"),
    "exact_match_synthetic": ("Share of synthetic records that copy a real person", "lower"),
    "exact_match_holdout": ("Share of new real people who match one by chance (reference)", "reference"),
    "closer_to_train_share": ("Share of synthetic records closer to training than to new people", "about 0.5"),
    "accuracy_members": ("Attacker's success on people in the training data", "close to non-members"),
    "accuracy_non_members": ("Attacker's success on people not in the data (reference)", "reference"),
    "baseline_accuracy": ("Success of guessing the most common value (reference)", "reference"),
    "match_rate_members": ("Share of targets the attacker found a synthetic match for", "reference"),
    "inference_risk": ("Attacker's extra success on people in the data (0 = none)", "lower"),
    "inference_risk_lower": ("95% lower bound of that extra success (0 = could be chance)", "lower"),
}


#: If a model trained on the real data reaches no more than this AUC, the
#: target is barely predictable, so the prediction test says little.
MIN_LEARNABLE_AUC = 0.55


def _improve(generator: Optional[str]) -> str:
    """What to try next for more usefulness, given the generator used."""
    if generator == "independent":
        return "Try the Bayesian network generator, which keeps links between columns."
    return "A larger ε keeps more detail, if your sharing policy allows it."


def assess_utility(report: EvaluationReport, generator: Optional[str] = None) -> Verdict:
    """How useful is the synthetic data for analysis?

    Args:
        report: Evaluation results.
        generator: Generator key used (``"bayesian"`` or ``"independent"``),
            so the advice suggests the right next step.
    """
    ml = report["ml"] if "ml" in report else None
    if ml is not None and "auc_real" in ml.metrics and ml["auc_real"] <= MIN_LEARNABLE_AUC:
        ml = None          # even real data barely predicts the target: judge on distributions
    if ml is not None:
        if "auc_real" in ml.metrics:
            # AUC 0.5 is a coin toss, so compare the gain *above chance*:
            # how much of the real data's predictive power survives.
            name, real, syn = "AUC", ml["auc_real"], ml["auc_synthetic"]
            ratio = max(0.0, (syn - 0.5) / (real - 0.5))
            share = f"it keeps {ratio:.0%} of the real data's predictive power above chance"
        else:
            name, real, syn = "F1 score", ml["f1_real"], ml["f1_synthetic"]
            ratio = ml["utility_ratio"]
            share = f"{ratio:.0%} of the real model's score"
        detail = (f"A model trained on the synthetic data scores {name} {syn:.2f} on real people, "
                  f"against {real:.2f} for one trained on the real data: {share}.")
        if ratio >= 0.9:
            return Verdict("good", "Highly useful", detail + " Suitable for most analysis and modelling.")
        if ratio >= 0.75:
            return Verdict("warning", "Moderately useful",
                           detail + " Good for exploring patterns; check key results on the real data.")
        return Verdict("critical", "Limited usefulness",
                       detail + " Fine for testing code or teaching, not for conclusions. " + _improve(generator))
    if "marginal" in report:
        tvd = report["marginal"]["tvd_2way_mean"]
        detail = f"Pairs of columns differ from the real data by {tvd:.0%} on average (2-way TVD)."
        if "ml" in report:
            detail += (" The prediction test was not used: even a model trained on the real data "
                       "predicts the sensitive attribute little better than chance.")
        if tvd <= 0.05:
            return Verdict("good", "Highly useful", detail)
        if tvd <= 0.15:
            return Verdict("warning", "Moderately useful", detail)
        return Verdict("critical", "Limited usefulness", detail + " " + _improve(generator))
    return Verdict("unknown", "Usefulness not measured", "No utility evaluator ran.")


def assess_disclosure(report: EvaluationReport) -> Verdict:
    """How much does the synthetic data still reveal about real people?"""
    signals: list[str] = []
    level = "good"
    measured = False
    if "dcr" in report:
        measured = True
        dcr = report["dcr"]
        excess_copies = dcr["exact_match_synthetic"] - dcr["exact_match_holdout"]
        closer = dcr["closer_to_train_share"]
        if excess_copies > 0.01 or closer >= 0.6:
            level = "critical"
        elif closer >= 0.55 and level != "critical":
            level = "warning"
        signals.append(f"{dcr['exact_match_synthetic']:.1%} of synthetic records copy a real person "
                       f"({dcr['exact_match_holdout']:.1%} expected by chance); "
                       f"closer-to-real share {closer:.2f} (0.50 = no leak)")
    if "inference" in report:
        measured = True
        inference = report["inference"]
        risk = inference["inference_risk"]
        # Judge on the 95% lower bound, so sampling noise on small data is not
        # mistaken for a leak (results saved before it existed fall back to the estimate).
        lower = inference.metrics.get("inference_risk_lower", risk)
        if lower >= 0.1:
            level = "critical"
        elif lower >= 0.03 and level == "good":
            level = "warning"
        text = f"attribute-inference advantage {risk:.3f} (0 = none)"
        if risk > 0 and lower == 0:
            text = f"attribute-inference advantage {risk:.3f}, within chance (95% lower bound 0)"
        elif risk > 0:
            text = f"attribute-inference advantage {risk:.3f} (95% lower bound {lower:.3f})"
        signals.append(text)
        targets = int(inference.details["targets"].iloc[0]) if inference.details is not None else 0
        if 0 < targets < 1000:
            small_sample = f" Measured on only {targets:,} people per group, so treat it as rough."
        else:
            small_sample = ""
    else:
        small_sample = ""
    if not measured:
        return Verdict("unknown", "Disclosure risk not measured",
                       "Risk evaluators need a holdout set and a sensitive attribute.")
    detail = "; ".join(signals)
    detail = detail[0].upper() + detail[1:] + "."
    labels = {
        "good": ("Low disclosure risk", " The synthetic data reveals general patterns, not individuals."),
        "warning": ("Some disclosure risk", " Consider a smaller ε before sharing widely."),
        "critical": ("High disclosure risk", " Do not share; real records are leaking through."),
    }
    label, advice = labels[level]
    return Verdict(level, label, detail + advice + small_sample)


def overall_recommendation(
    utility: Verdict, disclosure: Verdict, epsilon: float, generator: Optional[str] = None
) -> Optional[str]:
    """One sentence combining both verdicts, or None if either was not measured."""
    if "unknown" in (utility.level, disclosure.level):
        return None
    if disclosure.level == "critical":
        return "Not safe to share. Lower ε and generate again."
    if disclosure.level == "good" and utility.level == "good":
        return f"Safe to share and useful at ε = {epsilon:g}."
    if disclosure.level == "good" and utility.level == "critical":
        return f"Safe to share, but usefulness at ε = {epsilon:g} is limited. " + _improve(generator)
    if disclosure.level == "good":
        return f"Safe to share at ε = {epsilon:g}; validate important findings on the real data."
    return f"Usable at ε = {epsilon:g} for controlled sharing; choose a smaller ε for public release."
