import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st

st.set_page_config(page_title="CVA Posture Analysis", layout="wide")

@st.cache_data
def load_data():
    with open("posture_data.json", "r") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["samples"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    return df.sort_values("ts").reset_index(drop=True)

def filter_noise(df):
    s1 = df[df["hasPose"]].copy()

    q1, q3 = s1["angleDeg"].quantile(0.25), s1["angleDeg"].quantile(0.75)
    iqr = q3 - q1
    s2 = s1[(s1["angleDeg"] >= q1 - 1.5 * iqr) & (s1["angleDeg"] <= q3 + 1.5 * iqr)].copy()

    diff = s2["angleDeg"].diff().abs()
    cutoff = diff.mean() + 3 * diff.std()
    s3 = s2[diff.isna() | (diff <= cutoff)].copy()

    removed = {
        "hasPose filter": len(df) - len(s1),
        "IQR outlier": len(s1) - len(s2),
        "Continuity": len(s2) - len(s3),
    }
    return s3, removed, q1 - 1.5 * iqr, q3 + 1.5 * iqr

df = load_data()
cleaned, removed, iqr_lo, iqr_hi = filter_noise(df)
noise_pct = round((len(df) - len(cleaned)) / len(df) * 100, 1)

st.title("CVA Posture Analysis Dashboard")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Raw samples", f"{len(df):,}")
c2.metric("Clean samples", f"{len(cleaned):,}")
c3.metric("Noise removed", f"{len(df)-len(cleaned):,}", delta=f"-{noise_pct}%", delta_color="inverse")
c4.metric("Avg CVA (clean)", f"{cleaned['angleDeg'].mean():.1f}°",
          delta=f"{cleaned['angleDeg'].mean() - df['angleDeg'].mean():.1f}° vs raw")

st.markdown("---")
st.markdown("#### Filter pipeline")

col_left, col_right = st.columns([1, 2])

with col_left:
    for name, count in removed.items():
        st.markdown(f"**{name}**")
        st.progress(count / len(df), text=f"{count:,} samples ({count/len(df)*100:.1f}%)")

with col_right:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.barh(list(removed.keys()), list(removed.values()),
            color=["#e74c3c", "#e67e22", "#f1c40f"], height=0.45)
    for i, (name, val) in enumerate(removed.items()):
        ax.text(val + 5, i, f"{val:,}", va="center", fontsize=10)
    ax.set_xlabel("Samples removed")
    ax.set_xlim(0, max(removed.values()) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.markdown("#### Angle distribution — raw vs clean")

col_hist, col_stats = st.columns(2)

with col_hist:
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.hist(df["angleDeg"], bins=60, alpha=0.4, color="#aaaaaa", label="Raw")
    ax2.hist(cleaned["angleDeg"], bins=60, alpha=0.8, color="#3498db", label="Clean")
    ax2.axvline(45, color="red", linestyle="--", linewidth=1.2, label="Turtle threshold (45°)")
    ax2.axvline(iqr_lo, color="orange", linestyle=":", linewidth=1,
                label=f"IQR bounds ({iqr_lo:.1f}° / {iqr_hi:.1f}°)")
    ax2.axvline(iqr_hi, color="orange", linestyle=":", linewidth=1)
    ax2.set_xlabel("CVA Angle (°)")
    ax2.set_ylabel("Count")
    ax2.legend(fontsize=9)
    ax2.spines[["top", "right"]].set_visible(False)
    fig2.tight_layout()
    st.pyplot(fig2)

with col_stats:
    st.dataframe(pd.DataFrame({
        "Metric": ["Mean (°)", "Std dev (°)", "Min (°)", "Max (°)", "Turtle ratio (%)"],
        "Raw": [f"{df['angleDeg'].mean():.2f}", f"{df['angleDeg'].std():.2f}",
                f"{df['angleDeg'].min():.2f}", f"{df['angleDeg'].max():.2f}",
                f"{df['isTurtle'].mean()*100:.1f}%"],
        "Clean": [f"{cleaned['angleDeg'].mean():.2f}", f"{cleaned['angleDeg'].std():.2f}",
                  f"{cleaned['angleDeg'].min():.2f}", f"{cleaned['angleDeg'].max():.2f}",
                  f"{cleaned['isTurtle'].mean()*100:.1f}%"]
    }), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("#### CVA angle over time")

dates = sorted(cleaned["ts"].dt.date.unique())
selected = st.select_slider("Select date", options=dates,
                             format_func=lambda d: d.strftime("%b %d"))

day_raw = df[df["ts"].dt.date == selected]
day_clean = cleaned[cleaned["ts"].dt.date == selected]

fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(day_raw["ts"], day_raw["angleDeg"], alpha=0.2, color="#aaaaaa", linewidth=0.7, label="Raw")
ax3.plot(day_clean["ts"], day_clean["angleDeg"], alpha=0.85, color="#3498db", linewidth=0.9, label="Clean")
ax3.axhline(45, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Turtle threshold (45°)")
ax3.fill_between(day_clean["ts"], day_clean["angleDeg"], 45,
                 where=(day_clean["angleDeg"] < 45), alpha=0.15, color="red", label="Turtle zone")
ax3.set_ylabel("CVA Angle (°)")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax3.legend(fontsize=9)
ax3.spines[["top", "right"]].set_visible(False)
fig3.tight_layout()
st.pyplot(fig3)

st.markdown("---")
st.markdown("#### Session summary")

sessions = cleaned.groupby("sessionId").agg(
    avg_angle=("angleDeg", "mean"),
    turtle_count=("isTurtle", "sum"),
    total=("angleDeg", "count"),
    start=("ts", "min")
).reset_index()
sessions["turtle_ratio"] = (sessions["turtle_count"] / sessions["total"] * 100).round(1)
sessions["avg_angle"] = sessions["avg_angle"].round(2)
sessions["date"] = sessions["start"].dt.strftime("%b %d")
st.dataframe(
    sessions[["sessionId", "date", "avg_angle", "turtle_count", "turtle_ratio", "total"]]
    .rename(columns={
        "sessionId": "Session", "date": "Date", "avg_angle": "Avg CVA (°)",
        "turtle_count": "Turtle alerts", "turtle_ratio": "Turtle ratio (%)", "total": "Samples"
    }),
    use_container_width=True, hide_index=True
)
