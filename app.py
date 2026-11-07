"""PrivSynth dashboard.

Run from the project folder with::

    streamlit run app.py

The page follows the proposal's workflow: choose or upload data, see who
could be re-identified, pick a privacy level, generate synthetic data,
compare it with the real data, and download it. All logic lives in the
``privsynth`` package; this file only arranges it on screen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from privsynth import Dataset, ExperimentResults, KAnonymityChecker, PrivSynthError
from privsynth.dashboard import (
    GENERATOR_INFO,
    NOISE_INFO,
    PRIVACY_LEVELS,
    PrivacySettings,
    SampleDataSource,
    UploadedCSVSource,
    assess_disclosure,
    assess_original_risk,
    assess_utility,
    overall_recommendation,
    sensitive_candidates,
    suggest_schema,
)
from privsynth.dashboard import charts
from privsynth.data.loaders import AdultDataLoader
from privsynth.dashboard.sources import MAX_UPLOAD_MB
from privsynth.dashboard.verdicts import METRIC_GUIDE
from privsynth.experiments import REAL_DATA

RESULTS_CSV = Path(__file__).resolve().parent / "results" / "week3_runs.csv"
NONE = "(none)"

st.set_page_config(page_title="PrivSynth", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Cached loading (re-running the script must not re-read the file each time)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Loading data ...")
def load_sample(path: str) -> Dataset:
    return SampleDataSource(Path(path).parent).load()


@st.cache_data(show_spinner="Reading your file ...")
def load_upload(content: bytes, filename: str) -> tuple[Dataset, int]:
    """The dataset and the number of incomplete rows that were removed."""
    source = UploadedCSVSource(content, filename)
    dataset = source.load()
    return dataset, source.rows_dropped


@st.cache_data(show_spinner=False)
def load_experiments(path: str, modified: float) -> ExperimentResults:
    return ExperimentResults.from_csv(path)


def show_verdict(verdict) -> None:
    """Coloured box with icon and label, so meaning never relies on colour alone."""
    box = {"good": st.success, "warning": st.warning, "critical": st.error}.get(verdict.level, st.info)
    box(f"**{verdict.label}.** {verdict.message}", icon=verdict.icon)


def show_figure(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Sidebar: 1. data, 2. privacy settings, 3. generate
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ PrivSynth")
st.sidebar.caption("Privacy-protected synthetic data")

st.sidebar.header("1. Data")
sample = SampleDataSource()
choice = st.sidebar.radio("Data source", [sample.label, UploadedCSVSource.label], label_visibility="collapsed")

dataset: Dataset | None = None
bin_edges = None
source_key = "none"
default_qis: list[str] = []
default_sensitive = None
direct_ids: list[str] = []
reasons: dict[str, str] = {}

try:
    if choice == sample.label:
        dataset = load_sample(str(sample.path))
        bin_edges = sample.bin_edges
        source_key = f"sample-{sample.path.name}"
        st.sidebar.caption(sample.description)
        default_qis = list(AdultDataLoader.QUASI_IDENTIFIERS)
        default_sensitive = AdultDataLoader.SENSITIVE_ATTRIBUTE
    else:
        upload = st.sidebar.file_uploader(f"CSV file with a header row (max {MAX_UPLOAD_MB} MB)", type=["csv"])
        if upload is not None:
            content = upload.getvalue()
            dataset, dropped = load_upload(content, upload.name)
            source_key = "upload-" + hashlib.sha1(content).hexdigest()[:12]
            st.sidebar.caption(UploadedCSVSource(content, upload.name).description)
            if dropped:
                st.sidebar.warning(f"{dropped:,} rows with missing values (blank, ?, NA, None ...) "
                                   "were removed.", icon="⚠️")
            suggestion = suggest_schema(dataset.data)
            default_qis = suggestion.quasi_identifiers
            default_sensitive = suggestion.sensitive_attribute
            direct_ids = suggestion.direct_identifiers
            reasons = suggestion.reasons
except PrivSynthError as err:
    st.sidebar.error(f"Could not load the data: {err}")
    dataset = None

st.sidebar.header("2. Privacy level")
level_names = [lvl.name for lvl in PRIVACY_LEVELS]
level_name = st.sidebar.select_slider(
    "How much protection?", options=level_names, value="Balanced",
    help="Moves the privacy budget ε. Left = more noise and more privacy; right = more detail.",
)
level = PRIVACY_LEVELS[level_names.index(level_name)]
st.sidebar.caption(f"**ε = {level.epsilon:g}.** {level.explanation}")

generator = st.sidebar.radio(
    "Generator", list(GENERATOR_INFO), format_func=lambda k: GENERATOR_INFO[k][0],
    captions=[GENERATOR_INFO[k][1] for k in GENERATOR_INFO],
)

# Filled in below, once all settings are known, but shown above "Advanced settings".
generate_box = st.sidebar.container()

with st.sidebar.expander("Advanced settings"):
    noise = st.radio("Noise", list(NOISE_INFO), format_func=lambda k: NOISE_INFO[k][0],
                     captions=[NOISE_INFO[k][1] for k in NOISE_INFO])
    custom = st.checkbox("Type my own ε")
    epsilon = (st.number_input("ε", min_value=0.01, max_value=20.0, value=level.epsilon, step=0.1)
               if custom else level.epsilon)
    delta = None
    if noise == "gaussian":
        delta = st.number_input("δ", min_value=1e-10, max_value=0.01, value=1e-5, format="%.0e")
    k = st.number_input("k for the risk check", min_value=2, max_value=100, value=5, step=1,
                        help="A record is 'at risk' if fewer than k people share its details.")
    holdout_pct = st.slider("Real records held back for testing (%)", 10, 40, 20, step=5,
                            help="These people are never shown to the generator; they make the "
                                 "risk measurements fair.")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1,
                           help="Same seed + same settings = same synthetic data.")

settings = PrivacySettings(
    generator=generator, noise=noise, epsilon=float(epsilon), delta=delta,
    k=int(k), holdout_fraction=holdout_pct / 100, seed=int(seed),
)

with generate_box:
    st.header("3. Generate")
    generate = st.button("Generate synthetic data", type="primary", disabled=dataset is None)
    st.caption(f"Settings: {settings.describe()}")


# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.title("PrivSynth")
st.markdown("Share the patterns in sensitive data **without exposing the people in it**. "
            "PrivSynth measures who could be re-identified, generates synthetic records under "
            "differential privacy, and tells you how useful and how safe the result is.")

tab_data, tab_synth, tab_compare, tab_exp, tab_about = st.tabs(
    ["① Data and risk", "② Synthetic data", "③ Compare", "④ Experiments", "How it works"]
)

prepared: Dataset | None = None

with tab_data:
    if dataset is None:
        st.info("Choose the sample data or upload a CSV file in the sidebar to begin.", icon="👈")
    else:
        frame = dataset.data
        c1, c2, c3 = st.columns(3)
        c1.metric("Records", f"{dataset.n_rows:,}")
        c2.metric("Columns", len(dataset.columns))
        c3.metric("Source", dataset.name)

        st.subheader("Which columns are which?")
        st.caption("PrivSynth guessed these from the column names and values. Please check them.")
        remove = st.multiselect(
            "Direct identifiers to remove (names, ID or phone numbers)", dataset.columns,
            default=direct_ids, key=f"remove-{source_key}",
            help="These identify a person on their own and are dropped before anything else.",
        )
        if remove and reasons:
            st.caption("Flagged: " + "; ".join(f"`{c}` ({reasons[c]})" for c in remove if c in reasons))
        kept = [c for c in dataset.columns if c not in remove]
        sens_options = [NONE] + [c for c in sensitive_candidates(frame[kept]) if c in kept]
        sens_default = default_sensitive if default_sensitive in sens_options else NONE
        sensitive = st.selectbox(
            "Sensitive attribute (the detail to protect)", sens_options,
            index=sens_options.index(sens_default), key=f"sens-{source_key}",
            help="Columns with 2 to 20 different values, such as income band or diagnosis.",
        )
        sensitive = None if sensitive == NONE else sensitive
        qi_options = [c for c in kept if c != sensitive]
        qis = st.multiselect(
            "Quasi-identifiers (details an attacker might know, such as age, sex, district)",
            qi_options, default=[c for c in default_qis if c in qi_options], key=f"qis-{source_key}",
        )
        with st.expander("Preview the data"):
            st.dataframe(frame[kept].head(20), hide_index=True)

        if not qis:
            st.warning("Choose at least one quasi-identifier to measure re-identification risk.", icon="⚠️")
        else:
            prepared = Dataset(frame[kept], name=dataset.name, quasi_identifiers=qis,
                               sensitive_attribute=sensitive)
            checker = KAnonymityChecker(settings.k)
            report = checker.evaluate(prepared)
            st.subheader("Re-identification risk of the original data")
            show_verdict(assess_original_risk(report))
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Unique people", f"{report.pct_unique:.1f}%", help="Alone in their group: certain to be identified.")
            m2.metric(f"At risk (group < {settings.k})", f"{report.pct_at_risk:.1f}%")
            m3.metric("Smallest group (k achieved)", report.k_achieved)
            m4.metric("l-diversity", report.l_diversity if report.l_diversity is not None else "n/a",
                      help="Fewest different sensitive values in any group. 1 means some groups reveal it outright.")
            show_figure(charts.group_size_chart(report))
            with st.expander(f"Records most at risk ({report.n_at_risk:,})"):
                st.caption("Shown only to you, to help decide what to protect.")
                st.dataframe(checker.risky_records(prepared, report).head(50),
                             hide_index=True)

# Run the pipeline after the column roles are known --------------------------
if generate:
    if prepared is None:
        st.sidebar.error("Choose at least one quasi-identifier first (tab ①).")
    elif prepared.n_rows < 100:
        st.sidebar.error("At least 100 complete records are needed to generate and evaluate.")
    else:
        try:
            with st.spinner("Generating and evaluating synthetic data ..."):
                result = settings.build_pipeline(bin_edges).run(prepared)
            st.session_state["result"] = result
            st.session_state["settings"] = settings
            st.session_state["bin_edges"] = bin_edges
            st.session_state["signature"] = (source_key, tuple(prepared.columns), tuple(qis), sensitive, settings)
            st.sidebar.success("Done. See tabs ② and ③.", icon="✅")
        except (PrivSynthError, ValueError) as err:
            st.sidebar.error(f"Could not generate: {err}")

result = st.session_state.get("result")
run_settings: PrivacySettings | None = st.session_state.get("settings")
stale = result is not None and prepared is not None and st.session_state.get("signature") != (
    source_key, tuple(prepared.columns), tuple(prepared.quasi_identifiers), prepared.sensitive_attribute, settings)

with tab_synth:
    if result is None:
        st.info("Pick a privacy level and click **Generate synthetic data** in the sidebar.", icon="👈")
    else:
        if stale:
            st.warning("You have changed the data or settings since this run. Click Generate to update.", icon="⚠️")
        st.caption(f"Run: {run_settings.describe()}. Trained on {result.train.n_rows:,} records; "
                   f"{result.holdout.n_rows:,} held back for testing.")
        utility = assess_utility(result.evaluation, generator=run_settings.generator)
        disclosure = assess_disclosure(result.evaluation)
        advice = overall_recommendation(utility, disclosure, run_settings.epsilon, run_settings.generator)
        if advice:
            st.subheader(advice)
        col_u, col_d = st.columns(2)
        with col_u:
            st.markdown("**Usefulness**")
            show_verdict(utility)
        with col_d:
            st.markdown("**Remaining disclosure risk**")
            show_verdict(disclosure)

        m1, m2, m3 = st.columns(3)
        m1.metric("Synthetic records", f"{result.synthetic.n_rows:,}")
        m2.metric("Privacy budget spent", f"ε = {result.budget.spent_epsilon:.3g}"
                  + (f", δ = {result.budget.spent_delta:.0e}" if result.budget.spent_delta else ""))
        m3.metric("Time", f"{sum(result.timings.values()):.1f} s")

        st.dataframe(result.synthetic.data.head(100), hide_index=True)

        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ Synthetic data (CSV)", result.synthetic.data.to_csv(index=False).encode(),
                           file_name=f"{result.train.name.split(' (')[0]}_synthetic_eps{run_settings.epsilon:g}.csv",
                           mime="text/csv")
        summary = (f"PrivSynth report\nSettings: {run_settings.describe()}\n\n"
                   f"{utility}\n{disclosure}\n\n{result}\n")
        d2.download_button("⬇️ Privacy report (TXT)", summary.encode(), file_name="privsynth_report.txt",
                           mime="text/plain")
        d3.download_button("⬇️ All metrics (CSV)", result.evaluation.to_frame().to_csv(index=False).encode(),
                           file_name="privsynth_metrics.csv", mime="text/csv")

        with st.expander("Where the privacy budget was spent"):
            ledger = pd.DataFrame([{"step": e.label, "ε": e.epsilon, "δ": e.delta} for e in result.budget.ledger])
            st.dataframe(ledger, hide_index=True)
            st.caption(f"Total: ε = {result.budget.spent_epsilon:.4g} of {result.budget.total_epsilon:g}. "
                       "The PrivacyBudget object refuses any step that would overspend.")
        network = getattr(result.synthesizer, "network", None)
        if network:
            with st.expander("What the Bayesian network learned"):
                st.dataframe(pd.DataFrame(network, columns=["column", "depends on"]).fillna("—"), hide_index=True)
                st.caption("Each column is generated from the column it depends on. The structure "
                           "was chosen with the exponential mechanism, so it is private too.")

with tab_compare:
    if result is None:
        st.info("Generate synthetic data first.", icon="👈")
    else:
        st.subheader("Column by column")
        column = st.selectbox("Column", result.train.columns,
                              index=result.train.columns.index(result.train.sensitive_attribute)
                              if result.train.sensitive_attribute else 0)
        show_figure(charts.distribution_chart(result.train, result.synthetic, column,
                                              bin_edges=st.session_state.get("bin_edges")))
        if "association" in result.evaluation:
            st.subheader("Relationships between columns")
            st.caption("Darker = more strongly related. A good generator keeps the right-hand map "
                       "looking like the left-hand one.")
            show_figure(charts.association_chart(result.evaluation["association"].details,
                                                 columns=result.train.columns))
        st.subheader("All measurements")
        table = result.evaluation.to_frame()
        table["meaning"] = table["metric"].map(lambda m: METRIC_GUIDE.get(m, ("", ""))[0])
        table["better"] = table["metric"].map(lambda m: METRIC_GUIDE.get(m, ("", ""))[1])
        table = table[["evaluator", "kind", "meaning", "value", "better", "metric"]]
        st.dataframe(table, hide_index=True, height=35 * (len(table) + 1) + 3,
                     column_config={"value": st.column_config.NumberColumn(format="%.3f")})
        for name, reason in result.evaluation.skipped.items():
            st.caption(f"{name} was skipped: {reason}")

with tab_exp:
    st.subheader("How ε trades privacy for usefulness (UCI Adult)")
    if not RESULTS_CSV.exists():
        st.info("Run `python scripts/week3_experiments.py` to produce the experiment results.")
    else:
        experiments = load_experiments(str(RESULTS_CSV), RESULTS_CSV.stat().st_mtime)
        options = {
            "Prediction quality (AUC: train on synthetic, test on real)": ("ml.auc_synthetic", 0.5, "coin toss"),
            "Column pairs kept (2-way TVD, lower is better)": ("marginal.tvd_2way_mean", None, None),
            "Attribute inference risk (lower is better)": ("inference.inference_risk", None, None),
            "Closer to training than to new people (0.5 = no leak)": ("dcr.closer_to_train_share", 0.5, "no leak"),
        }
        pick = st.selectbox("Measure", list(options))
        metric, ref, ref_label = options[pick]
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        experiments.plot(metric, ax=ax, ylabel=pick.split(" (")[0], reference=ref, reference_label=ref_label)
        fig.patch.set_facecolor(charts.SURFACE)
        fig.tight_layout()
        show_figure(fig)
        st.caption(f"{len(experiments) - 1} runs: 2 generators × 2 noise types × 5 values of ε × 3 seeds, "
                   "on one fixed 80/20 split. Lines are means; bands show ±1 standard deviation. "
                   "The dotted line is what releasing the real data would give.")
        means = experiments.summary([metric])[metric]
        means = means[means.index.get_level_values("generator") != REAL_DATA]
        grid = means.unstack(["generator", "noise"])
        grid.columns = [f"{g} ({n})" for g, n in grid.columns]
        grid.index = [f"ε = {e:g}" for e in grid.index]
        st.markdown("**Mean over 3 seeds**")
        st.dataframe(grid.round(3))
        if experiments.baseline is not None:
            st.caption(f"Releasing the real data: {experiments.baseline[metric]:.3f}")

with tab_about:
    st.markdown(
        """
**The problem.** Removing names is not enough: a few details such as age, sex and district can
single a person out. Organisations therefore either share data unsafely or not at all.

**What PrivSynth does.**
1. **Measures risk** with *k-anonymity*: how many people share each combination of
   quasi-identifiers. A group of one means that person can be re-identified.
2. **Generates synthetic records** that follow the real patterns but describe nobody, using
   *differential privacy*: carefully calibrated noise guarantees that the output would look almost
   the same whether or not any one person was in the data. The **privacy budget ε** sets how much
   noise; a `PrivacyBudget` object tracks every step and refuses to overspend.
3. **Evaluates the result** for usefulness (distributions, relationships, and a model trained on
   synthetic data tested on real people) and for remaining risk (copies of real records, and an
   attack that tries to guess a person's sensitive value).

**The two generators.** *Independent columns* keeps each column's distribution but loses the
links between columns. The *Bayesian network* learns which columns depend on each other, so
relationships survive; its structure is chosen privately with the exponential mechanism.

**Built with** Python, NumPy, pandas, SciPy, scikit-learn, Matplotlib and Streamlit, designed
with object-oriented principles: abstract base classes for data loaders, privacy mechanisms,
synthesizers and evaluators, connected by a `PrivSynthPipeline`.
        """
    )
    st.caption("Data: UCI Adult (public). Verdict thresholds are rules of thumb from our experiments, "
               "not legal advice.")
