import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.stats import f_oneway
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

st.set_page_config(page_title="Face Restoration Analytics", page_icon="🔬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
h1 {
    font-family: 'Syne', sans-serif !important; font-weight: 800 !important;
    font-size: 2.4rem !important; letter-spacing: -0.02em !important;
    background: linear-gradient(90deg, #00C9A7 0%, #4FACFE 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
h2, h3 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #E6EDF3 !important; }
section[data-testid="stSidebar"] { background: #0D1117 !important; border-right: 1px solid #21262D !important; }
[data-testid="stMetric"] { background: #161B27; border: 1px solid #21262D; border-radius: 12px; padding: 1rem 1.2rem; }
[data-testid="stMetricLabel"] { font-family: 'Space Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.08em !important; color: #8B949E !important; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #00C9A7 !important; }
.stSelectbox label, .stSlider label { font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important; color: #8B949E !important; text-transform: uppercase; }
.badge { display:inline-block; background:#1A2A22; color:#00C9A7; font-family:'Space Mono',monospace; font-size:0.72rem; letter-spacing:0.12em; padding:4px 12px; border-radius:20px; border:1px solid #00C9A755; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#161B27", "axes.facecolor": "#161B27",
    "axes.edgecolor": "#21262D", "axes.labelcolor": "#8B949E",
    "axes.titlecolor": "#E6EDF3", "xtick.color": "#8B949E",
    "ytick.color": "#8B949E", "text.color": "#E6EDF3",
    "grid.color": "#21262D", "grid.linestyle": "--", "grid.linewidth": 0.5,
})
ACCENT, ACCENT2, ACCENT3 = "#00C9A7", "#4FACFE", "#FFD166"

BASE_DIR = Path(__file__).parent
METRICS_CSV = BASE_DIR / "paired_metrics.csv"
IMGPATHS_CSV = BASE_DIR / "image_paths.csv"

@st.cache_data
def load_data():
    return pd.read_csv(METRICS_CSV), pd.read_csv(IMGPATHS_CSV)

metrics_df, imgpaths_df = load_data()
degradations = sorted(metrics_df["degradation"].unique())

# Extract a short image name for display
metrics_df["image_name"] = metrics_df["original_path"].apply(lambda p: Path(p).name)

with st.sidebar:
    st.markdown('<div class="badge">GFPGAN · ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown("## Face Restoration")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Overview",
        "EDA & Distribution",
        "Statistical Tests (ANOVA)",
        "ML — Linear Regression",
        "Restoration Insights",
        "Degradation Analysis",
        "Image Comparison",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p style="font-family:Space Mono,monospace;font-size:0.68rem;color:#484F58;letter-spacing:0.06em;">FACE RESTORATION DASHBOARD</p>', unsafe_allow_html=True)

st.title("Face Restoration Analytics")
st.markdown("Explore **PSNR, SSIM, and NIQE** quality metrics across degradation types using EDA, statistical testing, and restoration analysis.")
st.markdown("---")

# =====================================================================
# PAGE 1: OVERVIEW
# =====================================================================
if page == "Overview":
    st.subheader("Overall Quality Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Avg PSNR", f"{metrics_df['psnr'].mean():.2f} dB")
    with col2: st.metric("Avg SSIM", f"{metrics_df['ssim'].mean():.3f}")
    with col3: st.metric("Avg NIQE", f"{metrics_df['niqe'].mean():.2f}")
    with col4: st.metric("Total Images", len(metrics_df))
    st.markdown("### Dataset Snapshot")
    st.dataframe(metrics_df.head(), use_container_width=True)
    st.markdown("### Descriptive Statistics")
    st.dataframe(metrics_df[["psnr","ssim","niqe"]].describe().T, use_container_width=True)

# =====================================================================
# PAGE 2: EDA & DISTRIBUTION
# =====================================================================
elif page == "EDA & Distribution":
    st.subheader("Exploratory Data Analysis")
    metric = st.selectbox("Select metric", ["psnr","ssim","niqe"], index=0)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {metric.upper()} Distribution")
        fig, ax = plt.subplots()
        ax.hist(metrics_df[metric].dropna(), bins=20, color=ACCENT, edgecolor="#0D1117", linewidth=0.5)
        ax.set_xlabel(metric.upper()); ax.set_ylabel("Frequency")
        st.pyplot(fig)
    with col2:
        st.markdown(f"#### {metric.upper()} by Degradation")
        fig, ax = plt.subplots(figsize=(6,4))
        metrics_df.boxplot(column=metric, by="degradation", ax=ax, grid=False, rot=45,
            boxprops=dict(color=ACCENT), whiskerprops=dict(color=ACCENT2),
            capprops=dict(color=ACCENT2), medianprops=dict(color=ACCENT3),
            flierprops=dict(markerfacecolor=ACCENT, marker="o", markersize=4))
        ax.set_title(f"{metric.upper()} by Degradation"); ax.set_xlabel("Degradation Type"); ax.set_ylabel(metric.upper()); fig.suptitle("")
        st.pyplot(fig)
    st.markdown("---")
    st.markdown("### Correlation Matrix")
    corr = metrics_df[["psnr","ssim","niqe"]].corr()
    st.dataframe(corr.style.background_gradient(cmap="BuGn"), use_container_width=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### PSNR vs SSIM")
        fig, ax = plt.subplots()
        ax.scatter(metrics_df["ssim"], metrics_df["psnr"], alpha=0.6, color=ACCENT, s=18)
        ax.set_xlabel("SSIM"); ax.set_ylabel("PSNR (dB)"); st.pyplot(fig)
    with col4:
        st.markdown("#### PSNR vs NIQE")
        fig, ax = plt.subplots()
        ax.scatter(metrics_df["niqe"], metrics_df["psnr"], alpha=0.6, color=ACCENT2, s=18)
        ax.set_xlabel("NIQE"); ax.set_ylabel("PSNR (dB)"); st.pyplot(fig)

# =====================================================================
# PAGE 3: STATISTICAL TESTS (ANOVA)
# =====================================================================
elif page == "Statistical Tests (ANOVA)":
    st.subheader("One-Way ANOVA")
    st.write("Testing whether **mean PSNR differs significantly** across degradation types.")
    if not SCIPY_AVAILABLE:
        st.error("SciPy not installed. Run `pip install scipy`.")
    else:
        groups = []
        for d in degradations:
            g = metrics_df.loc[metrics_df["degradation"]==d, "psnr"].dropna()
            if len(g) > 1: groups.append(g)
        if len(groups) < 2:
            st.warning("Not enough groups to run ANOVA.")
        else:
            F, p = f_oneway(*groups)
            c1, c2 = st.columns(2); c1.metric("F-Statistic", f"{F:.4f}"); c2.metric("p-value", f"{p:.6f}")
            if p < 0.05: st.success("p < 0.05 → Reject H₀ — degradation type **significantly affects** PSNR.")
            else: st.warning("p ≥ 0.05 → Fail to reject H₀ — no significant difference detected.")
            st.markdown("### Mean PSNR per Degradation")
            summary = metrics_df.groupby("degradation")["psnr"].agg(["mean","std","count"])
            st.dataframe(summary.style.format({"mean":"{:.2f}","std":"{:.2f}"}), use_container_width=True)
            fig, ax = plt.subplots()
            ax.bar(summary.index, summary["mean"], color=ACCENT, edgecolor="#0D1117", linewidth=0.5)
            ax.set_ylabel("Mean PSNR (dB)"); ax.set_xlabel("Degradation"); ax.set_title("Mean PSNR per Degradation"); plt.xticks(rotation=45)
            st.pyplot(fig)

# =====================================================================
# PAGE 4: ML — LINEAR REGRESSION
# =====================================================================
elif page == "ML — Linear Regression":
    st.subheader("ML — Linear Regression: Predicting PSNR from SSIM")
    st.write(
        "A **Linear Regression** model is trained to predict **PSNR** using **SSIM** as the sole feature. "
        "Both are continuous variables, making regression a natural fit. "
        "The model is evaluated with **MAE** and **R² score**, and the fit is visualised with a regression plot."
    )

    if not SKLEARN_AVAILABLE:
        st.error("scikit-learn is not installed. Run `pip install scikit-learn` and restart the app.")
    else:
        # ── Data prep ──────────────────────────────────────────────────────────
        ml_df = metrics_df[["ssim", "psnr"]].dropna().reset_index(drop=True)

        if len(ml_df) < 10:
            st.warning("Not enough data points to train a regression model (need at least 10).")
        else:
            test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20, step=5)
            random_state = st.number_input("Random seed", min_value=0, max_value=999, value=42, step=1)

            X = ml_df[["ssim"]].values
            y = ml_df["psnr"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size / 100, random_state=int(random_state)
            )

            # ── Train ──────────────────────────────────────────────────────────
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae  = mean_absolute_error(y_test, y_pred)
            r2   = r2_score(y_test, y_pred)
            coef = model.coef_[0]
            intercept = model.intercept_

            # ── Metrics row ───────────────────────────────────────────────────
            st.markdown("### Model Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("MAE", f"{mae:.4f} dB")
            c2.metric("R² Score", f"{r2:.4f}")
            c3.metric("Coefficient (SSIM)", f"{coef:.4f}")
            c4.metric("Intercept", f"{intercept:.4f}")

            if r2 >= 0.7:
                st.success(f"R² = {r2:.4f} — Strong linear relationship between SSIM and PSNR.")
            elif r2 >= 0.4:
                st.info(f"R² = {r2:.4f} — Moderate linear relationship between SSIM and PSNR.")
            else:
                st.warning(f"R² = {r2:.4f} — Weak linear relationship; SSIM alone explains little variance in PSNR.")

            st.markdown("---")

            # ── Regression plot ───────────────────────────────────────────────
            st.markdown("### Regression Plot")
            st.caption("Blue dots = test set samples · Red line = fitted regression line")

            x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
            y_line = model.predict(x_line)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(X_test, y_test, color=ACCENT2, alpha=0.65, s=22, label="Test samples", zorder=3)
            ax.plot(x_line, y_line, color=ACCENT, linewidth=2.5, label=f"Fit: PSNR = {coef:.2f}·SSIM + {intercept:.2f}", zorder=4)
            ax.set_xlabel("SSIM")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title("Linear Regression — PSNR Predicted from SSIM")
            ax.legend(facecolor="#1C2333", edgecolor="#21262D", labelcolor="#E6EDF3", fontsize=9)
            st.pyplot(fig)

            st.markdown("---")

            # ── Residual plot ─────────────────────────────────────────────────
            st.markdown("### Residual Plot")
            st.caption("Residuals = Actual PSNR − Predicted PSNR. Ideally scattered randomly around zero.")

            residuals = y_test - y_pred
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.scatter(y_pred, residuals, color=ACCENT3, alpha=0.65, s=22, zorder=3)
            ax2.axhline(0, color=ACCENT, linewidth=1.5, linestyle="--", zorder=4)
            ax2.set_xlabel("Predicted PSNR (dB)")
            ax2.set_ylabel("Residual (dB)")
            ax2.set_title("Residuals vs Predicted PSNR")
            st.pyplot(fig2)

            st.markdown("---")

            # ── Actual vs Predicted ───────────────────────────────────────────
            st.markdown("### Actual vs Predicted PSNR")
            st.caption("Points close to the diagonal line indicate accurate predictions.")

            fig3, ax3 = plt.subplots(figsize=(6, 6))
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax3.scatter(y_test, y_pred, color=ACCENT2, alpha=0.65, s=22, zorder=3)
            ax3.plot([min_val, max_val], [min_val, max_val], color=ACCENT, linewidth=1.8,
                     linestyle="--", label="Perfect prediction", zorder=4)
            ax3.set_xlabel("Actual PSNR (dB)")
            ax3.set_ylabel("Predicted PSNR (dB)")
            ax3.set_title("Actual vs Predicted PSNR")
            ax3.legend(facecolor="#1C2333", edgecolor="#21262D", labelcolor="#E6EDF3", fontsize=9)
            st.pyplot(fig3)

            st.markdown("---")

            # ── Prediction on custom SSIM ─────────────────────────────────────
            st.markdown("### Try It — Predict PSNR for a Custom SSIM Value")
            ssim_min = float(ml_df["ssim"].min())
            ssim_max = float(ml_df["ssim"].max())
            custom_ssim = st.slider(
                "SSIM value",
                min_value=round(ssim_min, 3),
                max_value=round(ssim_max, 3),
                value=round(ml_df["ssim"].median(), 3),
                step=0.001,
                format="%.3f",
            )
            predicted_psnr = model.predict([[custom_ssim]])[0]
            st.metric("Predicted PSNR", f"{predicted_psnr:.4f} dB", help=f"For SSIM = {custom_ssim:.3f}")

# =====================================================================
# PAGE 5: RESTORATION INSIGHTS (replaces Regression)
# =====================================================================
elif page == "Restoration Insights":
    st.subheader("Restoration Insights")
    st.write("Understand which images were restored best and worst, and which degradation types are hardest to recover from.")

    metric_choice = st.selectbox("Rank by metric", ["psnr", "ssim", "niqe"], index=0)
    higher_is_better = metric_choice in ["psnr", "ssim"]
    note = "Higher = better" if higher_is_better else "Lower = better (perceptual quality)"
    st.caption(f"_{note}_")

    st.markdown("---")

    # --- Best & Worst ---
    col1, col2 = st.columns(2)
    n = st.slider("How many top/bottom samples to show", 3, 20, 10)

    sorted_df = metrics_df.sort_values(metric_choice, ascending=not higher_is_better).reset_index(drop=True)
    best = sorted_df.head(n)[["image_name", "degradation", "psnr", "ssim", "niqe"]]
    worst = sorted_df.tail(n).sort_values(metric_choice, ascending=higher_is_better)[["image_name", "degradation", "psnr", "ssim", "niqe"]]

    with col1:
        st.markdown(f"#### ✅ Top {n} Best Restored")
        st.dataframe(
            best.style
                .format({"psnr":"{:.2f}","ssim":"{:.3f}","niqe":"{:.2f}"})
                .highlight_max(subset=[metric_choice], color="#1A3A2A"),
            use_container_width=True,
        )

    with col2:
        st.markdown(f"#### ❌ Top {n} Worst Restored")
        st.dataframe(
            worst.style
                .format({"psnr":"{:.2f}","ssim":"{:.3f}","niqe":"{:.2f}"})
                .highlight_min(subset=[metric_choice], color="#3A1A1A"),
            use_container_width=True,
        )

    st.markdown("---")

    # --- Degradation Difficulty Ranking ---
    st.markdown("### Degradation Difficulty Ranking")
    st.write("Which degradation type does GFPGAN struggle with the most?")

    diff = metrics_df.groupby("degradation").agg(
        avg_psnr=("psnr","mean"),
        avg_ssim=("ssim","mean"),
        avg_niqe=("niqe","mean"),
        count=("psnr","count"),
    ).reset_index()

    diff = diff.sort_values("avg_psnr", ascending=True).reset_index(drop=True)
    diff.index = diff.index + 1
    diff.index.name = "Rank (hardest → easiest)"

    st.dataframe(
        diff.style.format({"avg_psnr":"{:.2f}","avg_ssim":"{:.3f}","avg_niqe":"{:.2f}"}),
        use_container_width=True,
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [ACCENT if i == 0 else ACCENT2 for i in range(len(diff))]
    bars = ax.barh(diff["degradation"], diff["avg_psnr"], color=colors[::-1], edgecolor="#0D1117", linewidth=0.5)
    ax.set_xlabel("Avg PSNR (dB) — lower = harder to restore")
    ax.set_title("Degradation Difficulty (by Avg PSNR)")
    ax.invert_yaxis()
    st.pyplot(fig)

    st.markdown("---")

    st.markdown("### Full Per-Image Metric Breakdown")
    st.write("Sort by any column to explore individual restoration results.")

    sort_col = st.selectbox("Sort by", ["psnr","ssim","niqe","degradation","image_name"], index=0)
    sort_asc = st.checkbox("Ascending", value=False)

    display_df = metrics_df[["image_name","degradation","psnr","ssim","niqe"]].sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)
    st.dataframe(
        display_df.style.format({"psnr":"{:.2f}","ssim":"{:.3f}","niqe":"{:.2f}"}),
        use_container_width=True,
        height=400,
    )

# =====================================================================
# PAGE 6: DEGRADATION ANALYSIS
# =====================================================================
elif page == "Degradation Analysis":
    st.subheader("Performance by Degradation Type")
    grouped = metrics_df.groupby("degradation").agg(
        psnr_mean=("psnr","mean"), ssim_mean=("ssim","mean"),
        niqe_mean=("niqe","mean"), count=("psnr","count")
    ).reset_index()
    st.markdown("### Summary Table")
    st.dataframe(grouped.style.format({"psnr_mean":"{:.2f}","ssim_mean":"{:.3f}","niqe_mean":"{:.2f}"}), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1: st.markdown("#### PSNR by Degradation"); st.bar_chart(grouped.set_index("degradation")["psnr_mean"])
    with c2: st.markdown("#### SSIM by Degradation"); st.bar_chart(grouped.set_index("degradation")["ssim_mean"])
    st.markdown("---")
    st.markdown("#### NIQE by Degradation _(lower = better)_")
    st.bar_chart(grouped.set_index("degradation")["niqe_mean"])

# =====================================================================
# PAGE 7: IMAGE COMPARISON
# =====================================================================
elif page == "Image Comparison":
    st.subheader("Original vs Restored vs Corrected")
    degr_sel = st.selectbox("Degradation type:", degradations)
    subset = imgpaths_df[imgpaths_df["degradation"]==degr_sel].copy()
    def row_has_all_files(row):
        return all((BASE_DIR/row[k]).resolve().exists() for k in ["original","restored","corrected"])
    subset["ok"] = subset.apply(row_has_all_files, axis=1)
    subset = subset[subset["ok"]].reset_index(drop=True)
    if subset.empty:
        st.error("No image files found for this degradation type.")
    else:
        idx = st.slider("Sample index", 0, len(subset)-1, 0)
        row = subset.iloc[idx]
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown("**Original**"); st.image(str((BASE_DIR/row["original"]).resolve()), use_container_width=True)
        with col2: st.markdown("**Restored (GFPGAN)**"); st.image(str((BASE_DIR/row["restored"]).resolve()), use_container_width=True)
        with col3: st.markdown("**Corrected (Color-Tuned)**"); st.image(str((BASE_DIR/row["corrected"]).resolve()), use_container_width=True)
        st.caption(f"Degradation: **{degr_sel}** · Sample: **{idx}**")
