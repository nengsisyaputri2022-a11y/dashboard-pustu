# =========================================================
# DASHBOARD KLASIFIKASI PENYAKIT PUSTU (FINAL)
# - WARNA/THEME TETAP PUNYA KAMU (LIGHT + GRADIENT)
# - RINGKASAN: BIARKAN SEPERTI KODE KAMU (SUDAH KEREN)
# - KLASIFIKASI PASIEN: DIPERKAYA (KPI MINI + DONUT TOP-1 + GAUGE PRIORITAS + TOP PROBA CARD)
# - SIDEBAR FILTER (Gender + Range Umur) tetap untuk Ringkasan/Analisis
# =========================================================

import os
import re
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    hamming_loss, accuracy_score, precision_score, recall_score, f1_score,
    multilabel_confusion_matrix, roc_curve, auc
)

# =========================================================
# 0) KONFIGURASI UTAMA (cukup ubah di sini saja)
# =========================================================
BASE_DIR = os.path.dirname(__file__)
ARTIFACT_DIR = os.path.join(BASE_DIR, ".")
DATA_XLSX_PATH = os.path.join(BASE_DIR, "")

# =========================================================
# A) UI THEME / STYLE (punya kamu + KPI mini)
# =========================================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(1000px 520px at 12% 0%, rgba(99,102,241,0.22), transparent 60%),
                radial-gradient(800px 420px at 90% 10%, rgba(16,185,129,0.18), transparent 55%),
                linear-gradient(180deg, rgba(15,23,42,0.03), rgba(15,23,42,0.00));
        }
        div.block-container { padding-top: 1.05rem; }
        html, body, [class*="css"] { font-size: 15px; }

        section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, rgba(2,6,23,0.06), rgba(2,6,23,0.00));
            border-right: 1px solid rgba(2,6,23,0.08);
        }

        .hero {
            border-radius: 20px;
            padding: 18px 18px 14px 18px;
            border: 1px solid rgba(2,6,23,0.08);
            background:
              radial-gradient(820px 280px at 15% 0%, rgba(99,102,241,0.30), transparent 60%),
              radial-gradient(720px 240px at 85% 10%, rgba(16,185,129,0.22), transparent 60%),
              linear-gradient(180deg, rgba(255,255,255,0.90), rgba(255,255,255,0.72));
            box-shadow: 0 12px 34px rgba(2,6,23,0.10);
        }
        .hero h1 { margin:0; font-size: 30px; line-height: 1.1; }
        .hero p { margin: 8px 0 0 0; opacity: 0.92; }
        .hero .chips { margin-top: 10px; }
        .chip {
            display:inline-block;
            padding: 6px 10px;
            margin-right: 8px;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(2,6,23,0.10);
            background: rgba(255,255,255,0.62);
        }

        .card {
            border-radius: 16px;
            padding: 14px 14px 12px 14px;
            border: 1px solid rgba(2,6,23,0.08);
            background: rgba(255,255,255,0.78);
            box-shadow: 0 10px 22px rgba(2,6,23,0.07);
        }
        .card-title {
            font-weight: 900;
            font-size: 14px;
            letter-spacing: 0.2px;
            margin-bottom: 6px;
            opacity: 0.92;
        }
        .kpi {
            display:flex;
            align-items:flex-end;
            justify-content:space-between;
            gap: 10px;
        }
        .kpi .value{
            font-size: 28px;
            font-weight: 950;
            line-height: 1.0;
        }
        .kpi .sub{
            font-size: 12px;
            opacity: 0.80;
            margin-top: 4px;
        }

        /* ===== KPI mini ===== */
        .kpi-mini{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap: 12px;
        }
        .kpi-mini .left{
            display:flex;
            align-items:center;
            gap: 10px;
        }
        .kpi-icon{
            width: 38px;
            height: 38px;
            border-radius: 12px;
            display:flex;
            align-items:center;
            justify-content:center;
            border: 1px solid rgba(2,6,23,0.10);
            background: rgba(255,255,255,0.62);
            font-size: 18px;
        }
        .kpi-mini .label{
            font-weight: 900;
            font-size: 13px;
            opacity: 0.92;
            margin: 0;
        }
        .kpi-mini .num{
            font-weight: 950;
            font-size: 22px;
            line-height: 1.0;
            margin-top: 2px;
        }
        .kpi-mini .hint{
            font-size: 12px;
            opacity: 0.78;
            margin-top: 2px;
        }

        .badge {
            display:inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            font-weight: 900;
            font-size: 12px;
            border: 1px solid rgba(2,6,23,0.10);
        }
        .low { background: rgba(34,197,94,0.14); }
        .mid { background: rgba(245,158,11,0.18); }
        .high{ background: rgba(239,68,68,0.18); }

        .chips-wrap { margin-top: 6px; }
        .label-chip{
            display:inline-block;
            padding: 6px 10px;
            margin: 6px 6px 0 0;
            border-radius: 999px;
            font-weight: 800;
            font-size: 12px;
            border: 1px solid rgba(2,6,23,0.10);
            background: rgba(255,255,255,0.66);
        }

        div[data-testid="stDataFrame"]{
            border-radius: 14px;
            overflow:hidden;
            border: 1px solid rgba(2,6,23,0.08);
        }
        details{
            border-radius: 14px !important;
            border: 1px solid rgba(2,6,23,0.08) !important;
            background: rgba(255,255,255,0.70) !important;
        }
        div.stDownloadButton > button, div.stButton > button {
            border-radius: 12px !important;
            padding: 10px 14px !important;
            border: 1px solid rgba(2,6,23,0.12) !important;
            box-shadow: 0 10px 18px rgba(2,6,23,0.07);
        }

        .footer {
            margin-top: 18px;
            padding: 12px 14px;
            border-radius: 14px;
            border: 1px dashed rgba(2,6,23,0.14);
            background: rgba(255,255,255,0.58);
            font-size: 12px;
            opacity: 0.9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_hero():
    st.markdown(
        """
        <div class="hero">
          <h1>🩺 Dashboard Klasifikasi Penyakit Pustu</h1>
          <p>Sistem pendukung keputusan: <b>multi-label</b> + <b>prioritas penanganan</b> (threshold otomatis dari artefak training).</p>
          <div class="chips">
            <span class="chip">XGBoost One-vs-Rest</span>
            <span class="chip">TF-IDF + Umur & Gender</span>
            <span class="chip">UI Dashboard Modern</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{title}</div>
          <div class="kpi">
            <div>
              <div class="value">{value}</div>
              <div class="sub">{subtitle}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def kpi_mini(icon, label, value, hint=""):
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-mini">
            <div class="left">
              <div class="kpi-icon">{icon}</div>
              <div>
                <div class="label">{label}</div>
                <div class="num">{value}</div>
                <div class="hint">{hint}</div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def priority_badge(cat: str):
    if cat == "Tinggi":
        cls = "badge high"
    elif cat == "Sedang":
        cls = "badge mid"
    else:
        cls = "badge low"
    st.markdown(f'<span class="{cls}">Prioritas: {cat}</span>', unsafe_allow_html=True)


def score_progress(score: int, max_score: int = 8):
    score = int(max(0, min(score, max_score)))
    frac = score / max_score if max_score else 0
    st.progress(frac, text=f"Skor {score}/{max_score}")


def label_chips(labels):
    if not labels:
        st.write("-")
        return
    chips = "".join([f'<span class="label-chip">{l}</span>' for l in labels])
    st.markdown(f'<div class="chips-wrap">{chips}</div>', unsafe_allow_html=True)


# =========================================================
# B) GLOBAL PLOT STYLE
# =========================================================
def set_plot_style():
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.titlepad": 10,
    })

set_plot_style()


def plot_bar_counts(vc, title="", ylabel="Jumlah", rotate=0, figsize=(5.0, 3.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if rotate:
        ax.tick_params(axis="x", labelrotation=rotate)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_hist(series, title="", xlabel="", ylabel="Jumlah", bins=15, figsize=(5.0, 3.0)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_horizontal_bar(df_label_proba: pd.DataFrame, title="Top Probabilitas", figsize=(5.4, 3.6)):
    d = df_label_proba.sort_values("Probabilitas", ascending=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(d["Label"].astype(str), d["Probabilitas"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("Probabilitas")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_donut(values, labels, title="", center_text=None, figsize=(4.2, 3.2)):
    values = np.array(values, dtype=float)
    total = float(values.sum()) if values.sum() else 1.0
    fig, ax = plt.subplots(figsize=figsize)
    wedges, _ = ax.pie(
        values,
        labels=None,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.38, edgecolor="white", linewidth=1),
    )
    ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    ax.set_title(title)

    if center_text is None:
        center_text = f"{int(total):,}".replace(",", ".")

    ax.text(0, 0.05, str(center_text), ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, -0.12, "total", ha="center", va="center", fontsize=10, alpha=0.8)

    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_line_counts(x_labels, y_values, title="", xlabel="", ylabel="Jumlah", figsize=(5.0, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(y_values)), y_values, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(x) for x in x_labels], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


# ===== TAMBAHAN KHUSUS KLASIFIKASI PASIEN =====
def plot_gauge(score: int, max_score: int = 8, title="Skor Prioritas", figsize=(4.6, 2.4)):
    score = int(max(0, min(score, max_score)))
    frac = score / max_score if max_score else 0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # semi-circle base
    theta = np.linspace(np.pi, 2*np.pi, 300)
    ax.plot(np.cos(theta), np.sin(theta), linewidth=14, alpha=0.18)

    # progress arc
    theta2 = np.linspace(np.pi, np.pi + np.pi * frac, 250)
    ax.plot(np.cos(theta2), np.sin(theta2), linewidth=14, alpha=0.95)

    # needle
    ang = np.pi + np.pi * frac
    ax.plot([0, 0.88*np.cos(ang)], [0, 0.88*np.sin(ang)], linewidth=3, alpha=0.95)

    ax.text(0, 0.15, title, ha="center", va="center", fontsize=11, fontweight="bold", alpha=0.9)
    ax.text(0, -0.12, f"{score}/{max_score}", ha="center", va="center", fontsize=16, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_donut_single(value: float, title="Top-1 Confidence", figsize=(3.8, 3.0)):
    v = float(max(0.0, min(1.0, value)))
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(
        [v, 1 - v],
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.35, edgecolor="white", linewidth=1),
    )
    ax.text(0, 0.05, f"{v*100:.1f}%", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, -0.13, "top-1", ha="center", va="center", fontsize=10, alpha=0.8)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


# =========================================================
# 1) UTIL: LOAD ARTEFAK OTOMATIS
# =========================================================
def pick_latest(pattern: str) -> str:
    files = glob.glob(os.path.join(ARTIFACT_DIR, pattern))
    if not files:
        raise FileNotFoundError(f"Tidak menemukan '{pattern}' di folder: {ARTIFACT_DIR}")
    return max(files, key=os.path.getmtime)


@st.cache_resource
def load_artifacts():
    model_path  = pick_latest("model_xgb_ovr_*.joblib")
    tfidf_path  = pick_latest("tfidf_*.joblib")
    mlb_path    = pick_latest("mlb_*.joblib")
    gender_path = pick_latest("le_gender_*.joblib")
    prio_path   = pick_latest("priority_weight_*.joblib")
    valid_path  = pick_latest("valid_labels_*.joblib")
    thr_path    = pick_latest("threshold_*.joblib")

    model = joblib.load(model_path)
    tfidf = joblib.load(tfidf_path)
    mlb = joblib.load(mlb_path)
    le_gender = joblib.load(gender_path)
    priority_weight = joblib.load(prio_path)
    valid_labels = joblib.load(valid_path)
    threshold_default = float(joblib.load(thr_path))

    loaded = {
        "model": os.path.basename(model_path),
        "tfidf": os.path.basename(tfidf_path),
        "mlb": os.path.basename(mlb_path),
        "le_gender": os.path.basename(gender_path),
        "priority_weight": os.path.basename(prio_path),
        "valid_labels": os.path.basename(valid_path),
        "threshold(otomatis)": os.path.basename(thr_path),
    }
    return model, tfidf, mlb, le_gender, threshold_default, priority_weight, valid_labels, loaded


# =========================================================
# 2) UTIL: PRIORITAS
# =========================================================
def priority_score(pred_labels, priority_weight: dict):
    base = sum(priority_weight.get(lbl, 1) for lbl in pred_labels)
    if len(pred_labels) >= 2:
        base += 2
    return int(base)

def priority_category(score: int):
    if score >= 6:
        return "Tinggi"
    elif score >= 4:
        return "Sedang"
    else:
        return "Rendah"


# =========================================================
# 3) UTIL: BUILD FEATURE + PREDICT PROBA
# =========================================================
def _sanitize_gender_list(le_gender, series_like):
    fixed = []
    for jk in series_like:
        jk = "Tidak Diketahui" if jk is None else str(jk)
        if jk not in le_gender.classes_:
            fixed.append("Tidak Diketahui")
        else:
            fixed.append(jk)
    return fixed

def build_feature_matrix(tfidf, le_gender, texts, ages, genders):
    texts = pd.Series(texts).fillna("").astype(str).str.lower().str.strip().tolist()
    ages = pd.Series(ages).fillna(0).astype(float).values
    genders_fixed = _sanitize_gender_list(le_gender, genders)
    gender_enc = le_gender.transform(genders_fixed)

    X_text = tfidf.transform(texts)
    X_num = csr_matrix(np.c_[ages, gender_enc])
    X_all = hstack([X_text, X_num]).tocsr()
    return X_all

@st.cache_data(show_spinner=False)
def predict_proba_df(df_eval_local: pd.DataFrame,
                     col_text: str, col_age: str, col_gender: str,
                     threshold_default: float):
    X_all = build_feature_matrix(
        tfidf, le_gender,
        df_eval_local[col_text].values,
        df_eval_local[col_age].values,
        df_eval_local[col_gender].values,
    )
    Y_proba = model.predict_proba(X_all)
    _ = float(threshold_default)
    return Y_proba


# =========================================================
# 4) UTIL: KLASIFIKASI 1 PASIEN
# =========================================================
def classify_patient(model, tfidf, mlb, le_gender, diagnosa_text, umur, jenis_kelamin, threshold_default: float):
    text = "" if diagnosa_text is None else str(diagnosa_text).lower().strip()

    jk = "Tidak Diketahui" if jenis_kelamin is None else str(jenis_kelamin)
    if jk not in le_gender.classes_:
        jk = "Tidak Diketahui"
    jk_enc = le_gender.transform([jk])[0]

    umur = 0.0 if umur is None else float(umur)

    xt = tfidf.transform([text])
    xn = csr_matrix([[umur, jk_enc]])
    x_input = hstack([xt, xn]).tocsr()

    proba = model.predict_proba(x_input)[0]
    y_hat = (proba >= float(threshold_default)).astype(int)

    if y_hat.sum() == 0:
        y_hat[np.argmax(proba)] = 1

    labels = mlb.inverse_transform(y_hat.reshape(1, -1))[0]
    return list(labels), proba


# =========================================================
# 5) UTIL: CLEAN LABEL DARI DIAGNOSA UNTUK EVALUASI/ANALISIS
# =========================================================
penyakit_map = {
    "ispa": "ispa", "batuk": "ispa", "flu": "ispa",
    "influenza": "influenza", "influensa": "influenza", "fluenza": "influenza",
    "gastristis": "gastritis", "gastristus": "gastritis",
    "s.dypepsia": "gastritis", "dyspepsia": "gastritis", "maag": "gastritis",
    "demam": "demam", "febris": "demam", "confebris": "demam", "deman": "demam",
    "a.urat": "asam urat", "asam urat": "asam urat",
    "goat": "gout", "goat atritis": "gout",
    "ht": "hipertensi", "darah tinggi": "hipertensi",
    "asma": "asma", "alergi": "alergi", "rematik": "rematik",
    "dermatitis": "dermatitis", "gatal-gatal": "gatal-gatal", "diare": "diare",
    "luka": "luka", "scabies": "scabies", "tinea cruris": "tinea cruris",
    "gingivitis": "gingivitis", "herpes": "herpes",
    "sakit kepala": "sakit kepala", "sakit gigi": "sakit gigi", "sakit mata": "sakit mata",
    "nyeri sendi": "nyeri sendi", "vertigo": "vertigo",
    "anc": "anc", "kb": "kb", "suntik kb": "kb", "suntik 3bl": "kb", "kb pasca salin": "kb",
    "cc": "kanker serviks",
    "colostrol": "kolesterol",
    "redresing": "dressing luka",
    "dm": "diabetes", "oa": "osteoartritis", "ra": "rematik", "vl": "vertigo", "of": "osteoartritis",
    "abses payudara": "abses", "bisul": "abses",
}
non_penyakit = ["cek", "kontrol", "ulangan", "tensi", "obat", "rujukan", "td"]

def clean_diagnosis_to_labels(text: str, valid_labels_set: set):
    t = str(text).lower().strip()
    if t in ["nan", "none", ""] or t in non_penyakit:
        return []
    if any(x in t for x in ["rujukan", "kontrol", "cek", "ulangan"]):
        return []
    t = re.sub(r"[,+/;]", " dan ", t)
    t = re.sub(r"\s+", " ", t).strip()

    parts = [p.strip() for p in t.split("dan") if p.strip()]
    labels = []
    for p in parts:
        if p in non_penyakit:
            continue
        p = penyakit_map.get(p, p)
        if p in valid_labels_set:
            labels.append(p)
    return list(set(labels))


# =========================================================
# 6) LOAD DATASET (untuk dashboard ringkasan + filter)
# =========================================================
@st.cache_data(show_spinner=False)
def load_dataset_if_exists(path):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_excel(path)
    except Exception:
        return None


# =========================================================
# 7) UI SETTINGS
# =========================================================
st.set_page_config(page_title="Dashboard Klasifikasi Penyakit Pustu", page_icon="🩺", layout="wide")
inject_custom_css()
render_hero()

model, tfidf, mlb, le_gender, threshold_default, priority_weight, valid_labels, loaded_files = load_artifacts()
valid_labels_set = set(valid_labels)

df_default = load_dataset_if_exists(DATA_XLSX_PATH)

# ===================== SIDEBAR NAV + FILTER =====================
st.sidebar.markdown("## 🧭 Navigasi")
menu = st.sidebar.radio(
    "Menu",
    ["🏠 Ringkasan", "🧾 Klasifikasi Pasien", "📦 Batch Klasifikasi (Excel)", "📊 Analisis Data", "✅ Evaluasi Model", "ℹ️ Info Sistem"],
    label_visibility="collapsed"
)

with st.sidebar.expander("📦 Artefak yang digunakan"):
    st.json(loaded_files)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Filter (Ringkasan/Analisis)")

gender_filter = None
age_min, age_max = 0, 120
if df_default is not None:
    col_gender = "Jenis Kelamin" if "Jenis Kelamin" in df_default.columns else None
    col_age = "Umur" if "Umur" in df_default.columns else None

    if col_gender:
        options_gender = sorted(df_default[col_gender].dropna().astype(str).unique().tolist())
        if len(options_gender) > 0:
            gender_filter = st.sidebar.multiselect("Jenis Kelamin", options_gender, default=options_gender)

    if col_age:
        try:
            a_min = int(np.nanmin(pd.to_numeric(df_default[col_age], errors="coerce")))
            a_max = int(np.nanmax(pd.to_numeric(df_default[col_age], errors="coerce")))
            a_min = max(0, a_min)
            a_max = min(120, a_max) if a_max > 0 else 120
        except Exception:
            a_min, a_max = 0, 120
        age_min, age_max = st.sidebar.slider("Range Umur", 0, 120, (a_min, a_max))

st.sidebar.caption("Tip: input teks seperti data latih (mis. `demam dan diare`).")


def apply_filters(df):
    if df is None:
        return None
    out = df.copy()
    if "Jenis Kelamin" in out.columns and gender_filter is not None and len(gender_filter) > 0:
        out = out[out["Jenis Kelamin"].astype(str).isin([str(x) for x in gender_filter])]
    if "Umur" in out.columns:
        umur_num = pd.to_numeric(out["Umur"], errors="coerce").fillna(0)
        out = out[(umur_num >= age_min) & (umur_num <= age_max)]
    return out


# =========================================================
# PAGE: RINGKASAN (BIARKAN SEPERTI KODE KAMU)
# =========================================================
if menu == "🏠 Ringkasan":
    df_f = apply_filters(df_default)

    c1, c2, c3, c4, c5 = st.columns(5)

    total_rows = int(len(df_f)) if df_f is not None else 0
    total_cols = int(df_f.shape[1]) if df_f is not None else 0
    missing_total = int(df_f.isnull().sum().sum()) if df_f is not None else 0

    mean_age = "-"
    if df_f is not None and "Umur" in df_f.columns:
        umur_num = pd.to_numeric(df_f["Umur"], errors="coerce")
        if np.isfinite(umur_num.mean()):
            mean_age = f"{umur_num.mean():.1f}"

    unique_diag = "-"
    if df_f is not None and "Diagnosa" in df_f.columns:
        unique_diag = str(int(df_f["Diagnosa"].fillna("").astype(str).nunique()))

    with c1: kpi_mini("🧾", "Total Data", f"{total_rows:,}".replace(",", "."), "baris setelah filter")
    with c2: kpi_mini("🧱", "Kolom", f"{total_cols}", "jumlah kolom dataset")
    with c3: kpi_mini("🧯", "Missing", f"{missing_total:,}".replace(",", "."), "total sel kosong")
    with c4: kpi_mini("🎂", "Rata-rata Umur", f"{mean_age}", "tahun")
    with c5: kpi_mini("🏷️", "Unique Diagnosa", f"{unique_diag}", "variasi diagnosa")

    st.markdown("")
    st.markdown('<div class="card"><div class="card-title">Apa yang dilakukan sistem ini?</div>', unsafe_allow_html=True)
    st.write(
        "- **Klasifikasi multi-label**: 1 pasien bisa keluar lebih dari 1 label penyakit.\n"
        "- **Prioritas penanganan**: dari label yang keluar → dihitung **skor** dan **kategori**.\n"
        "- **Threshold otomatis**: tidak ditampilkan di UI (diambil dari artefak training)."
    )
    st.info("Catatan: visual ringkasan di bawah mengambil data dari dataset default (jika file Excel ada).")
    st.markdown("</div>", unsafe_allow_html=True)

    if df_f is None:
        st.warning("Dataset default belum ditemukan, jadi visual ringkasan belum bisa ditampilkan. Cek DATA_XLSX_PATH.")
    else:
        gender_counts = None
        if "Jenis Kelamin" in df_f.columns:
            gender_counts = df_f["Jenis Kelamin"].astype(str).value_counts()

        top_label_ser = None
        if "Diagnosa" in df_f.columns:
            labels_all = []
            for x in df_f["Diagnosa"].fillna(""):
                labels_all.extend(clean_diagnosis_to_labels(x, valid_labels_set))
            if len(labels_all) > 0:
                top_label_ser = pd.Series(labels_all).value_counts().head(10)

        age_bins = None
        if "Umur" in df_f.columns:
            umur_num = pd.to_numeric(df_f["Umur"], errors="coerce").fillna(0)
            bins = [0, 5, 12, 18, 30, 45, 60, 120]
            labels_bin = ["0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "61+"]
            age_bins = pd.cut(umur_num, bins=bins, labels=labels_bin, include_lowest=True).value_counts().reindex(labels_bin).fillna(0)

        row1_left, row1_mid, row1_right = st.columns([0.9, 1.2, 1.1], vertical_alignment="top")

        with row1_left:
            st.markdown('<div class="card"><div class="card-title">Gender (Donut)</div>', unsafe_allow_html=True)
            if gender_counts is None or len(gender_counts) == 0:
                st.write("Tidak ada kolom **Jenis Kelamin** / data kosong.")
            else:
                fig = plot_donut(gender_counts.values, gender_counts.index.tolist(), title="Distribusi Jenis Kelamin", figsize=(4.2, 3.0))
                st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row1_mid:
            st.markdown('<div class="card"><div class="card-title">Top Label Penyakit (Bar)</div>', unsafe_allow_html=True)
            if top_label_ser is None:
                st.write("Tidak ada label yang bisa diambil dari kolom Diagnosa (cek format diagnosa).")
            else:
                fig = plot_bar_counts(top_label_ser, title="Top 10 Label", ylabel="Jumlah", rotate=25, figsize=(5.6, 3.0))
                st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with row1_right:
            st.markdown('<div class="card"><div class="card-title">Trend Umur (Line)</div>', unsafe_allow_html=True)
            if age_bins is None:
                st.write("Tidak ada kolom **Umur** / data kosong.")
            else:
                fig = plot_line_counts(age_bins.index.tolist(), age_bins.values.tolist(),
                                       title="Jumlah Pasien per Kelompok Umur",
                                       xlabel="Kelompok Umur",
                                       ylabel="Jumlah",
                                       figsize=(5.2, 3.0))
                st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Lihat preview data (setelah filter)"):
            st.dataframe(df_f.head(30), use_container_width=True)


# =========================================================
# PAGE: KLASIFIKASI PASIEN (DIPERKAYA)
# =========================================================
elif menu == "🧾 Klasifikasi Pasien":
    st.markdown('<div class="card"><div class="card-title">🧾 Klasifikasi 1 Pasien</div>', unsafe_allow_html=True)
    st.caption("Tampilan dibuat lebih dashboard: KPI mini + donut confidence + gauge prioritas + top probabilities.")
    st.markdown("</div>", unsafe_allow_html=True)

    left, right = st.columns([1.05, 0.95], vertical_alignment="top")

    with left:
        st.markdown('<div class="card"><div class="card-title">Input Pasien</div>', unsafe_allow_html=True)
        with st.form("form_klasifikasi", clear_on_submit=False):
            diagnosa = st.text_area("Keluhan/Diagnosa (teks)", value="vertigo dan gatal-gatal", height=110)
            c1, c2 = st.columns([0.5, 0.5])
            with c1:
                umur = st.number_input("Umur", min_value=0, max_value=120, value=30)
            with c2:
                jk = st.selectbox("Jenis Kelamin", options=list(le_gender.classes_))
            submit = st.form_submit_button("🔍 Klasifikasikan")
        st.caption("Tip: format input seperti data latih (contoh: `demam dan diare`).")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><div class="card-title">Ringkas Output</div>', unsafe_allow_html=True)
        if "last_pred" not in st.session_state:
            kpi_mini("⏳", "Status", "Menunggu", "klik tombol klasifikasikan")
            kpi_mini("⚙️", "Threshold", f"{float(threshold_default):.3f}", "otomatis dari artefak")
            kpi_mini("🏷️", "Output", "-", "label belum dihitung")
        else:
            lp = st.session_state["last_pred"]
            kpi_mini("🚦", "Prioritas", lp["cat"], f"Skor {lp['score']}/8")
            kpi_mini("🏷️", "_toggle Labels", str(lp["n_labels"]), "jumlah label keluar")
            kpi_mini("📈", "Top-1 Prob", f"{lp['top1']*100:.1f}%", lp["top1_label"])
        st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        labels, proba = classify_patient(model, tfidf, mlb, le_gender, diagnosa, umur, jk, threshold_default)
        score = priority_score(labels, priority_weight)
        cat = priority_category(score)

        top1_idx = int(np.argmax(proba))
        top1 = float(proba[top1_idx])
        top1_label = str(mlb.classes_[top1_idx])

        st.session_state["last_pred"] = {
            "labels": labels,
            "score": score,
            "cat": cat,
            "n_labels": len(labels),
            "top1": top1,
            "top1_label": top1_label,
            "proba": proba,
        }

    if "last_pred" in st.session_state:
        lp = st.session_state["last_pred"]
        labels = lp["labels"]
        score = lp["score"]
        cat = lp["cat"]
        proba = lp["proba"]

        st.markdown("---")

        # Row: Labels + Gauge + Donut
        colA, colB, colC = st.columns([0.52, 0.28, 0.20], vertical_alignment="top")

        with colA:
            st.markdown('<div class="card"><div class="card-title">📌 Label Hasil</div>', unsafe_allow_html=True)
            label_chips(labels)
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown('<div class="card"><div class="card-title">🚦 Prioritas Penanganan</div>', unsafe_allow_html=True)
            priority_badge(cat)
            st.markdown("")
            fig_g = plot_gauge(score, max_score=8, title="Skor Prioritas", figsize=(4.6, 2.4))
            st.pyplot(fig_g, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with colC:
            st.markdown('<div class="card"><div class="card-title">✅ Confidence</div>', unsafe_allow_html=True)
            fig_d = plot_donut_single(lp["top1"], title="Top-1 Confidence", figsize=(3.6, 2.9))
            st.pyplot(fig_d, clear_figure=True)
            st.caption(f"Top-1: **{lp['top1_label']}**")
            st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE: BATCH KLASIFIKASI EXCEL (FINAL: LEBIH KEREN)
# =========================================================
elif menu == "📦 Batch Klasifikasi (Excel)":
    st.markdown('<div class="card"><div class="card-title">📦 Batch Klasifikasi dari Excel</div>', unsafe_allow_html=True)
    st.write("Klasifikasi multi-label + prioritas untuk banyak data. (Vectorized + cache)")
    st.caption("Hasil dilengkapi ringkasan KPI, filter, chart, dan download Excel/CSV.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- helper: plot anti dempet ----------
    def plot_counts_auto(vc, title="", axis_label="Jumlah", max_labels_vertical=10):
        vc = vc.copy()
        vc.index = vc.index.astype(str)
        n = len(vc)
        max_len = int(vc.index.map(len).max()) if n > 0 else 0

        if (n > max_labels_vertical) or (max_len >= 12):
            vc2 = vc.sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.28 * n + 1.6)))
            ax.barh(vc2.index, vc2.values)
            ax.set_title(title)
            ax.set_xlabel(axis_label)
            ax.grid(axis="x", alpha=0.25)
            fig.tight_layout()
            return fig
        else:
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            ax.bar(vc.index, vc.values)
            ax.set_title(title)
            ax.set_ylabel(axis_label)
            ax.tick_params(axis="x", labelrotation=45)
            for t in ax.get_xticklabels():
                t.set_ha("right")
            ax.grid(axis="y", alpha=0.25)
            fig.subplots_adjust(bottom=0.32)
            fig.tight_layout()
            return fig

    # ---------- helper: deteksi kolom fleksibel ----------
    def find_col(df_cols, candidates):
        cols_lower = {c.lower(): c for c in df_cols}
        for cand in candidates:
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        for c in df_cols:
            cl = c.lower()
            for cand in candidates:
                if cand.lower() in cl:
                    return c
        return None

    # ---------- sumber data ----------
    source = st.radio("Sumber data", ["Gunakan dataset default (langsung)", "Upload file Excel lain"], horizontal=True)

    df_in = None
    data_key = None

    if source == "Gunakan dataset default (langsung)":
        if not os.path.exists(DATA_XLSX_PATH):
            st.error(f"Dataset default tidak ditemukan di:\n{DATA_XLSX_PATH}")
        else:
            df_in = pd.read_excel(DATA_XLSX_PATH)
            data_key = f"default::{os.path.getmtime(DATA_XLSX_PATH)}::{len(df_in)}"
            st.success("✅ Dataset default berhasil dimuat.")
    else:
        up = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if up is None:
            st.info("Upload file untuk memulai batch.")
        else:
            df_in = pd.read_excel(up)
            data_key = f"upload::{up.name}::{len(df_in)}"
            st.success("✅ File upload berhasil dimuat.")

    if df_in is None:
        st.stop()

    # ---------- preview + deteksi kolom ----------
    st.markdown('<div class="card"><div class="card-title">Preview Data</div>', unsafe_allow_html=True)
    st.dataframe(df_in.head(15), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col_diag = find_col(df_in.columns, ["Diagnosa", "Keluhan", "Diagnosis", "Anamnesis"])
    col_umur = find_col(df_in.columns, ["Umur", "Usia", "Age"])
    col_jk   = find_col(df_in.columns, ["Jenis Kelamin", "JK", "Gender", "Kelamin"])
    col_nama = find_col(df_in.columns, ["Nama", "Name", "Pasien"])

    st.markdown('<div class="card"><div class="card-title">Deteksi Kolom</div>', unsafe_allow_html=True)
    st.write({
        "Kolom Diagnosa/Keluhan": col_diag,
        "Kolom Umur/Usia": col_umur,
        "Kolom Jenis Kelamin/JK": col_jk,
        "Kolom Nama (opsional)": col_nama
    })
    st.caption("Minimal harus ada: Diagnosa/Keluhan, Umur/Usia, Jenis Kelamin/JK.")
    st.markdown("</div>", unsafe_allow_html=True)

    if col_diag is None or col_umur is None or col_jk is None:
        st.error("❌ Kolom wajib tidak ditemukan. Minimal harus ada: Diagnosa/Keluhan, Umur/Usia, Jenis Kelamin/JK.")
        st.stop()

    # ---------- opsi output ----------
    st.markdown('<div class="card"><div class="card-title">Pengaturan Batch</div>', unsafe_allow_html=True)
    copt1, copt2, copt3 = st.columns([1, 1, 1])
    with copt1:
        topk_prob = st.slider("Top-K probabilitas yang disimpan", 3, 15, 5)
    with copt2:
        simpan_proba = st.checkbox("Simpan Top-K Probabilitas ke kolom", value=True)
    with copt3:
        urut_default = st.selectbox("Urutan default hasil", ["Skor_Prioritas (desc)", "Kategori_Prioritas", "Nama/Index"])

    st.caption("Tip: menyimpan Top-K probabilitas membuat hasil lebih jelas saat audit.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- batch runner (vectorized) ----------
    @st.cache_data(show_spinner=False)
    def run_batch_vectorized(df_local: pd.DataFrame,
                             col_diag: str, col_umur: str, col_jk: str,
                             threshold_default: float, data_key: str,
                             topk_prob: int, simpan_proba: bool):
        _ = str(data_key) + f"::{topk_prob}::{simpan_proba}"

        Y_proba = predict_proba_df(df_local, col_diag, col_umur, col_jk, threshold_default)
        Y_pred = (Y_proba >= float(threshold_default)).astype(int)

        empty = np.where(Y_pred.sum(axis=1) == 0)[0]
        if len(empty) > 0:
            best_idx = np.argmax(Y_proba[empty], axis=1)
            Y_pred[empty, best_idx] = 1

        out = df_local.copy()
        label_texts, scores, cats = [], [], []

        # top-k proba (optional)
        topk_text = []

        for i in range(Y_pred.shape[0]):
            idxs = np.where(Y_pred[i] == 1)[0]
            labels = [mlb.classes_[j] for j in idxs]
            label_texts.append(", ".join(labels))

            sc = priority_score(labels, priority_weight)
            scores.append(sc)
            cats.append(priority_category(sc))

            if simpan_proba:
                probs = Y_proba[i]
                top_idx = np.argsort(probs)[::-1][:topk_prob]
                pairs = [f"{mlb.classes_[j]} ({probs[j]*100:.1f}%)" for j in top_idx]
                topk_text.append(" | ".join(pairs))

        out["Label_Hasil"] = label_texts
        out["Skor_Prioritas"] = scores
        out["Kategori_Prioritas"] = cats
        if simpan_proba:
            out[f"Top{topk_prob}_Prob"] = topk_text

        return out

    # ---------- tombol run + progress ----------
    run_now = st.button("🚀 Jalankan Batch", use_container_width=True)

    if "batch_done" not in st.session_state:
        st.session_state["batch_done"] = False

    if run_now:
        with st.spinner("Memproses batch klasifikasi..."):
            prog = st.progress(0, text="Menyiapkan...")
            prog.progress(20, text="Menghitung probabilitas (predict_proba)...")
            df_out = run_batch_vectorized(df_in, col_diag, col_umur, col_jk, threshold_default, data_key, topk_prob, simpan_proba)
            prog.progress(80, text="Menyusun output & ringkasan...")
            st.session_state["df_out"] = df_out
            st.session_state["batch_done"] = True
            prog.progress(100, text="Selesai ✅")

    if not st.session_state["batch_done"]:
        st.info("Klik **Jalankan Batch** untuk menghasilkan output.")
        st.stop()

    df_out = st.session_state["df_out"]

    # ---------- KPI ringkasan ----------
    total = len(df_out)
    vc_cat = df_out["Kategori_Prioritas"].value_counts()
    n_rendah = int(vc_cat.get("Rendah", 0))
    n_sedang = int(vc_cat.get("Sedang", 0))
    n_tinggi = int(vc_cat.get("Tinggi", 0))

    st.markdown("### Ringkasan Batch")
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_mini("🧾", "Total Baris", f"{total:,}".replace(",", "."), "diproses")
    with k2: kpi_mini("🟢", "Rendah", f"{n_rendah:,}".replace(",", "."), "kategori prioritas")
    with k3: kpi_mini("🟠", "Sedang", f"{n_sedang:,}".replace(",", "."), "kategori prioritas")
    with k4: kpi_mini("🔴", "Tinggi", f"{n_tinggi:,}".replace(",", "."), "kategori prioritas")
    with k5: kpi_mini("⚙️", "Threshold", f"{float(threshold_default):.3f}", "otomatis")

    st.markdown("---")

    # ---------- filter hasil ----------
    st.markdown('<div class="card"><div class="card-title">🔎 Filter & Tampilan Hasil</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns([0.32, 0.38, 0.30])
    with f1:
        pilih_cat = st.multiselect("Kategori Prioritas", ["Tinggi", "Sedang", "Rendah"], default=["Tinggi", "Sedang", "Rendah"])
    with f2:
        keyword = st.text_input("Cari (Nama/Diagnosa/Label)", value="")
    with f3:
        max_show = st.slider("Maks baris ditampilkan", 50, 500, 200)

    dview = df_out.copy()
    if pilih_cat:
        dview = dview[dview["Kategori_Prioritas"].isin(pilih_cat)]

    if keyword.strip():
        key = keyword.strip().lower()
        cols_search = []
        if col_nama and col_nama in dview.columns:
            cols_search.append(col_nama)
        cols_search += [col_diag, "Label_Hasil"]

        mask = False
        for c in cols_search:
            mask = mask | dview[c].fillna("").astype(str).str.lower().str.contains(key)
        dview = dview[mask]

    # sort
    if urut_default == "Skor_Prioritas (desc)":
        dview = dview.sort_values(["Skor_Prioritas", "Kategori_Prioritas"], ascending=[False, True])
    elif urut_default == "Kategori_Prioritas":
        # Tinggi dulu
        order_map = {"Tinggi": 0, "Sedang": 1, "Rendah": 2}
        dview["_ord"] = dview["Kategori_Prioritas"].map(order_map).fillna(99).astype(int)
        dview = dview.sort_values(["_ord", "Skor_Prioritas"], ascending=[True, False]).drop(columns=["_ord"])
    else:
        if col_nama and col_nama in dview.columns:
            dview = dview.sort_values(col_nama, ascending=True)

    st.caption(f"Menampilkan **{min(len(dview), max_show):,}** dari **{len(dview):,}** baris hasil filter.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- tabs hasil ----------
    tabA, tabB, tabC, tabD = st.tabs(["📋 Tabel Hasil", "📌 Ringkasan", "🏷️ Top Label", "⬇️ Download"])

    # ---- TAB A: tabel hasil (styled) ----
    with tabA:
        st.markdown('<div class="card"><div class="card-title">📋 Hasil Batch (Tabel)</div>', unsafe_allow_html=True)

        show_df = dview.head(max_show).copy()

        def _style_cat(row):
            cat = str(row.get("Kategori_Prioritas", ""))
            if cat == "Tinggi":
                return ["background-color: rgba(239,68,68,0.12)"] * len(row)
            if cat == "Sedang":
                return ["background-color: rgba(245,158,11,0.14)"] * len(row)
            if cat == "Rendah":
                return ["background-color: rgba(34,197,94,0.12)"] * len(row)
            return [""] * len(row)

        st.dataframe(show_df, use_container_width=True, height=420)
        st.caption("Tip: gunakan filter untuk fokus ke pasien prioritas tinggi.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- TAB B: ringkasan & chart prioritas ----
    with tabB:
        colL, colR = st.columns([0.55, 0.45], vertical_alignment="top")

        with colL:
            st.markdown('<div class="card"><div class="card-title">📌 Distribusi Prioritas</div>', unsafe_allow_html=True)
            st.write(vc_cat)
            fig = plot_counts_auto(vc_cat, title="Distribusi Kategori Prioritas", axis_label="Jumlah", max_labels_vertical=10)
            st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with colR:
            st.markdown('<div class="card"><div class="card-title">📌 Statistik Skor</div>', unsafe_allow_html=True)
            s = pd.to_numeric(df_out["Skor_Prioritas"], errors="coerce").fillna(0)
            st.write({
                "Min": int(s.min()),
                "Median": float(s.median()),
                "Mean": round(float(s.mean()), 2),
                "Max": int(s.max()),
            })
            fig, ax = plt.subplots(figsize=(5.2, 3.1))
            ax.hist(s, bins=10)
            ax.set_title("Distribusi Skor Prioritas")
            ax.set_xlabel("Skor")
            ax.set_ylabel("Jumlah")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ---- TAB C: top label hasil prediksi ----
    with tabC:
        st.markdown('<div class="card"><div class="card-title">🏷️ Top Label Hasil Prediksi</div>', unsafe_allow_html=True)

        # pecah Label_Hasil jadi list label
        labels_all = []
        for x in df_out["Label_Hasil"].fillna("").astype(str):
            parts = [p.strip() for p in x.split(",") if p.strip()]
            labels_all.extend(parts)

        if len(labels_all) == 0:
            st.info("Tidak ada label hasil yang bisa dihitung.")
        else:
            topk_lbl = st.slider("Top-K label", 5, 30, 15)
            vc_lbl = pd.Series(labels_all).value_counts().head(topk_lbl)
            fig = plot_counts_auto(vc_lbl, title=f"Top {topk_lbl} Label Prediksi", axis_label="Jumlah", max_labels_vertical=10)
            st.pyplot(fig, clear_figure=True)

            with st.expander("Lihat tabel top label"):
                st.dataframe(vc_lbl.rename("Jumlah").reset_index().rename(columns={"index": "Label"}), use_container_width=True)

        st.caption("Ini berguna untuk melihat penyakit apa yang paling sering muncul dari hasil prediksi batch.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- TAB D: download CSV + Excel ----
    with tabD:
        st.markdown('<div class="card"><div class="card-title">⬇️ Download Output</div>', unsafe_allow_html=True)
        st.caption("Unduh hasil batch untuk laporan atau arsip.")

        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download hasil (CSV)", csv,
                           file_name="hasil_batch_klasifikasi_pustu.csv",
                           mime="text/csv")

        # Excel (xlsx)
        import io
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df_out.to_excel(writer, index=False, sheet_name="Hasil_Batch")
            vc_cat.reset_index().rename(columns={"index": "Kategori", "Kategori_Prioritas": "Jumlah"}).to_excel(
                writer, index=False, sheet_name="Ringkasan"
            )
        st.download_button(
            "⬇️ Download hasil (Excel .xlsx)",
            data=buffer.getvalue(),
            file_name="hasil_batch_klasifikasi_pustu.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE: ANALISIS DATA (FINAL: LEBIH KEREN & MUDAH DIPAHAMI)
# =========================================================
elif menu == "📊 Analisis Data":
    st.markdown('<div class="card"><div class="card-title">📊 Analisis Data Pasien</div>', unsafe_allow_html=True)
    st.caption("Ringkasan dataset + distribusi + top penyakit + kualitas data (missing).")
    st.markdown("</div>", unsafe_allow_html=True)

    if not os.path.exists(DATA_XLSX_PATH):
        st.warning("File dataset Excel belum ditemukan. Cek DATA_XLSX_PATH di konfigurasi.")
    else:
        df = pd.read_excel(DATA_XLSX_PATH)
        df = apply_filters(df)

        if df is None or len(df) == 0:
            st.warning("Data kosong setelah filter sidebar. Coba longgarkan filter Jenis Kelamin / Range Umur.")
        else:
            # =====================================================
            # HELPER: PLOT ANTI-DEMPET (AUTO)
            # - kalau kategori banyak/label panjang -> otomatis bar horizontal
            # =====================================================
            def plot_counts_auto(vc, title="", axis_label="Jumlah", max_labels_vertical=10):
                vc = vc.copy()
                vc.index = vc.index.astype(str)

                n = len(vc)
                max_len = int(vc.index.map(len).max()) if n > 0 else 0

                # Kriteria auto horizontal:
                # - kategori banyak, atau
                # - label panjang
                if (n > max_labels_vertical) or (max_len >= 12):
                    # horizontal lebih enak dibaca
                    vc2 = vc.sort_values(ascending=True)
                    fig, ax = plt.subplots(figsize=(7.0, max(3.2, 0.28 * n + 1.6)))
                    ax.barh(vc2.index, vc2.values)
                    ax.set_title(title)
                    ax.set_xlabel(axis_label)
                    ax.grid(axis="x", alpha=0.25)
                    fig.tight_layout()
                    return fig
                else:
                    # vertical, tapi dibikin anti-dempet
                    fig, ax = plt.subplots(figsize=(7.2, 3.6))
                    ax.bar(vc.index, vc.values)
                    ax.set_title(title)
                    ax.set_ylabel(axis_label)
                    ax.tick_params(axis="x", labelrotation=45)
                    for t in ax.get_xticklabels():
                        t.set_ha("right")
                    ax.grid(axis="y", alpha=0.25)
                    fig.subplots_adjust(bottom=0.32)
                    fig.tight_layout()
                    return fig

            # =====================================================
            # PREP
            # =====================================================
            total_rows = int(len(df))
            total_cols = int(df.shape[1])
            missing_total = int(df.isnull().sum().sum())

            col_gender = "Jenis Kelamin" if "Jenis Kelamin" in df.columns else None
            col_age    = "Umur" if "Umur" in df.columns else None
            col_diag   = "Diagnosa" if "Diagnosa" in df.columns else None

            mean_age = "-"
            if col_age:
                umur_num = pd.to_numeric(df[col_age], errors="coerce")
                if np.isfinite(umur_num.mean()):
                    mean_age = f"{umur_num.mean():.1f}"

            unique_diag = "-"
            if col_diag:
                unique_diag = str(int(df[col_diag].fillna("").astype(str).nunique()))

            # =====================================================
            # KPI ROW
            # =====================================================
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: kpi_mini("🧾", "Jumlah Data", f"{total_rows:,}".replace(",", "."), "setelah filter")
            with k2: kpi_mini("🧱", "Kolom", f"{total_cols}", "jumlah fitur/kolom")
            with k3: kpi_mini("🧯", "Missing", f"{missing_total:,}".replace(",", "."), "total sel kosong")
            with k4: kpi_mini("🎂", "Rata-rata Umur", f"{mean_age}", "tahun")
            with k5: kpi_mini("🏷️", "Unique Diagnosa", f"{unique_diag}", "variasi diagnosa")

            st.markdown("---")

            # =====================================================
            # TABS
            # =====================================================
            tab1, tab2, tab3, tab4 = st.tabs(["📌 Overview", "📈 Distribusi", "🏷️ Top Penyakit", "🧹 Kualitas Data"])

            # =====================================================
            # TAB 1: OVERVIEW
            # =====================================================
            with tab1:
                left, right = st.columns([0.65, 0.35], vertical_alignment="top")

                with left:
                    st.markdown('<div class="card"><div class="card-title">Preview Data</div>', unsafe_allow_html=True)
                    st.dataframe(df.head(20), use_container_width=True)
                    st.caption("Preview 20 baris pertama (setelah filter sidebar).")
                    st.markdown("</div>", unsafe_allow_html=True)

                with right:
                    st.markdown('<div class="card"><div class="card-title">Info Kolom (Tipe + Missing)</div>', unsafe_allow_html=True)
                    info_df = pd.DataFrame({
                        "Kolom": df.columns,
                        "Tipe": [str(df[c].dtype) for c in df.columns],
                        "Missing": [int(df[c].isna().sum()) for c in df.columns],
                        "Unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
                    }).sort_values(["Missing", "Unique"], ascending=[False, False])

                    st.dataframe(info_df, use_container_width=True, height=420)
                    st.caption("Urut dari missing terbesar → fokus perbaikan data.")
                    st.markdown("</div>", unsafe_allow_html=True)

            # =====================================================
            # TAB 2: DISTRIBUSI
            # =====================================================
            with tab2:
                cA, cB, cC = st.columns([1.0, 1.1, 1.0], vertical_alignment="top")

                # 1) Gender
                with cA:
                    st.markdown('<div class="card"><div class="card-title">Distribusi Jenis Kelamin</div>', unsafe_allow_html=True)
                    if col_gender is None:
                        st.info("Kolom **Jenis Kelamin** tidak ditemukan.")
                    else:
                        vc = df[col_gender].astype(str).value_counts()
                        fig = plot_counts_auto(vc, title="Distribusi Jenis Kelamin", axis_label="Jumlah", max_labels_vertical=8)
                        st.pyplot(fig, clear_figure=True)
                        st.caption("Melihat dominasi gender pada data pasien.")
                    st.markdown("</div>", unsafe_allow_html=True)

                # 2) Kelompok Umur (lebih mudah dipahami)
                with cB:
                    st.markdown('<div class="card"><div class="card-title">Distribusi Umur (Kelompok)</div>', unsafe_allow_html=True)
                    if col_age is None:
                        st.info("Kolom **Umur** tidak ditemukan.")
                    else:
                        umur_num = pd.to_numeric(df[col_age], errors="coerce").fillna(0)
                        bins = [0, 5, 12, 18, 30, 45, 60, 120]
                        labels_bin = ["0-5", "6-12", "13-18", "19-30", "31-45", "46-60", "61+"]
                        age_bins = pd.cut(umur_num, bins=bins, labels=labels_bin, include_lowest=True) \
                            .value_counts().reindex(labels_bin).fillna(0)

                        fig = plot_line_counts(
                            age_bins.index.tolist(),
                            age_bins.values.tolist(),
                            title="Jumlah Pasien per Kelompok Umur",
                            xlabel="Kelompok Umur",
                            ylabel="Jumlah",
                            figsize=(5.6, 3.0)
                        )
                        st.pyplot(fig, clear_figure=True)
                        st.caption("Lebih mudah dibaca daripada histogram mentah.")
                    st.markdown("</div>", unsafe_allow_html=True)

                # 3) Histogram Umur (detail)
                with cC:
                    st.markdown('<div class="card"><div class="card-title">Histogram Umur (Detail)</div>', unsafe_allow_html=True)
                    if col_age is None:
                        st.info("Kolom **Umur** tidak ditemukan.")
                    else:
                        bins_hist = st.slider("Jumlah bins histogram", 8, 35, 15)
                        fig = plot_hist(
                            pd.to_numeric(df[col_age], errors="coerce"),
                            title="Distribusi Umur (Histogram)",
                            xlabel="Umur",
                            bins=bins_hist,
                            figsize=(5.0, 3.0)
                        )
                        st.pyplot(fig, clear_figure=True)
                        st.caption("Untuk melihat bentuk sebaran umur lebih detail.")
                    st.markdown("</div>", unsafe_allow_html=True)

            # =====================================================
            # TAB 3: TOP PENYAKIT
            # =====================================================
            with tab3:
                if col_diag is None:
                    st.info("Kolom **Diagnosa** tidak ditemukan.")
                else:
                    topk = st.slider("Top-K yang ditampilkan", 5, 25, 12)

                    c1, c2 = st.columns([1, 1], vertical_alignment="top")

                    # A) Top diagnosa mentah
                    with c1:
                        st.markdown('<div class="card"><div class="card-title">Top Diagnosa (Mentah dari Excel)</div>', unsafe_allow_html=True)
                        raw = df[col_diag].fillna("").astype(str).str.lower().str.strip()
                        raw = raw[raw != ""]
                        if len(raw) == 0:
                            st.info("Tidak ada data diagnosa yang valid.")
                        else:
                            vc_raw = raw.value_counts().head(topk)
                            fig = plot_counts_auto(vc_raw, title=f"Top {topk} Diagnosa Mentah", axis_label="Jumlah", max_labels_vertical=10)
                            st.pyplot(fig, clear_figure=True)

                            with st.expander("Lihat tabel top diagnosa mentah"):
                                st.dataframe(
                                    vc_raw.rename("Jumlah").reset_index().rename(columns={"index": "Diagnosa"}),
                                    use_container_width=True
                                )
                        st.markdown("</div>", unsafe_allow_html=True)

                    # B) Top label normalisasi (lebih stabil)
                    with c2:
                        st.markdown('<div class="card"><div class="card-title">Top Label Penyakit (Setelah Normalisasi)</div>', unsafe_allow_html=True)
                        labels_all = []
                        for x in df[col_diag].fillna(""):
                            labels_all.extend(clean_diagnosis_to_labels(x, valid_labels_set))

                        if len(labels_all) == 0:
                            st.info("Tidak ada label yang bisa diambil dari diagnosa. Cek mapping/format teks.")
                        else:
                            vc_lbl = pd.Series(labels_all).value_counts().head(topk)
                            fig = plot_counts_auto(vc_lbl, title=f"Top {topk} Label Penyakit", axis_label="Jumlah", max_labels_vertical=10)
                            st.pyplot(fig, clear_figure=True)

                            with st.expander("Lihat tabel top label (normalisasi)"):
                                st.dataframe(
                                    vc_lbl.rename("Jumlah").reset_index().rename(columns={"index": "Label"}),
                                    use_container_width=True
                                )

                        st.caption("Normalisasi membuat analisis lebih stabil (mis. batuk/flu → ispa).")
                        st.markdown("</div>", unsafe_allow_html=True)

            # =====================================================
            # TAB 4: KUALITAS DATA
            # =====================================================
            with tab4:
                st.markdown('<div class="card"><div class="card-title">Missing Value per Kolom</div>', unsafe_allow_html=True)
                miss = df.isnull().sum().sort_values(ascending=False)
                miss = miss[miss > 0]

                if len(miss) == 0:
                    st.success("✅ Tidak ada missing value pada dataset setelah filter.")
                else:
                    fig = plot_counts_auto(miss, title="Missing per Kolom", axis_label="Jumlah Missing", max_labels_vertical=10)
                    st.pyplot(fig, clear_figure=True)

                    with st.expander("Lihat tabel missing detail"):
                        miss_df = miss.rename("Missing").reset_index().rename(columns={"index": "Kolom"})
                        miss_df["Persen"] = (miss_df["Missing"] / max(1, total_rows) * 100).round(2)
                        st.dataframe(miss_df, use_container_width=True)

                    st.caption("Saran: prioritaskan pembersihan di kolom dengan missing tertinggi.")
                st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PAGE: EVALUASI MODEL (FINAL: LEBIH BAGUS & MUDAH DIPAHAMI)
# =========================================================
elif menu == "✅ Evaluasi Model":
    st.markdown('<div class="card"><div class="card-title">✅ Evaluasi Model</div>', unsafe_allow_html=True)
    st.write("Halaman ini mengevaluasi performa model multi-label menggunakan dataset Excel.")
    st.caption("Optimasi: predict_proba dihitung sekali. Threshold otomatis dari artefak.")
    st.markdown("</div>", unsafe_allow_html=True)

    if not os.path.exists(DATA_XLSX_PATH):
        st.warning("Dataset Excel belum ditemukan. Set DATA_XLSX_PATH di konfigurasi.")
    else:
        df = pd.read_excel(DATA_XLSX_PATH)
        required_cols = {"Diagnosa", "Umur", "Jenis Kelamin"}

        if not required_cols.issubset(df.columns):
            st.error(f"Dataset harus punya kolom: {required_cols}")
        else:
            # =========================
            # 1) BUILD GROUND TRUTH
            # =========================
            y_true_labels = df["Diagnosa"].fillna("").apply(lambda x: clean_diagnosis_to_labels(x, valid_labels_set))
            mask = y_true_labels.apply(len) > 0
            df_eval = df.loc[mask].copy()
            y_true_labels = y_true_labels.loc[mask]

            st.markdown('<div class="card"><div class="card-title">📌 Data yang dievaluasi</div>', unsafe_allow_html=True)
            st.write(
                f"- Total baris dataset: **{len(df):,}**\n"
                f"- Baris valid (punya label setelah normalisasi): **{len(df_eval):,}**\n"
                f"- Baris dibuang (tidak jadi label): **{len(df) - len(df_eval):,}**"
            )
            st.caption("Catatan: baris yang diagnosanya kosong/rujukan/kontrol/cek biasanya tidak dihitung sebagai ground truth.")
            st.markdown("</div>", unsafe_allow_html=True)

            if len(df_eval) == 0:
                st.warning("Tidak ada baris yang menghasilkan label ground truth setelah cleaning.")
            else:
                # =========================
                # 2) PREDICT (CACHED)
                # =========================
                Y_true = mlb.transform(y_true_labels)

                Y_proba = predict_proba_df(df_eval, "Diagnosa", "Umur", "Jenis Kelamin", threshold_default)
                Y_pred = (Y_proba >= float(threshold_default)).astype(int)

                # fallback: kalau kosong semua, ambil top-1
                empty = np.where(Y_pred.sum(axis=1) == 0)[0]
                if len(empty) > 0:
                    best_idx = np.argmax(Y_proba[empty], axis=1)
                    Y_pred[empty, best_idx] = 1

                # =========================
                # 3) METRICS (RINGKAS)
                # =========================
                ham = hamming_loss(Y_true, Y_pred)
                subset_acc = accuracy_score(Y_true, Y_pred)
                micro_f1 = f1_score(Y_true, Y_pred, average="micro", zero_division=0)
                macro_f1 = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

                # micro precision/recall biar makin jelas
                micro_p = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
                micro_r = recall_score(Y_true, Y_pred, average="micro", zero_division=0)

                st.markdown("### Ringkasan Performa")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1: kpi_card("Hamming Loss", f"{ham:.4f}", "lebih kecil lebih baik")
                with col2: kpi_card("Subset Acc", f"{subset_acc:.4f}", "label harus sama persis")
                with col3: kpi_card("Micro Precision", f"{micro_p:.4f}", "ketepatan keseluruhan")
                with col4: kpi_card("Micro Recall", f"{micro_r:.4f}", "cakupan label benar")
                with col5: kpi_card("Micro F1", f"{micro_f1:.4f}", "umum untuk multi-label")
                with col6: kpi_card("Macro F1", f"{macro_f1:.4f}", "rata-rata per label")

                st.markdown('<div class="card"><div class="card-title">🧠 Cara membaca metrik (singkat)</div>', unsafe_allow_html=True)
                st.write(
                    "- **Hamming Loss**: seberapa sering model salah label (lebih kecil lebih baik).\n"
                    "- **Subset Accuracy**: paling ketat, benar hanya jika semua label cocok.\n"
                    "- **Micro (P/R/F1)**: menilai performa keseluruhan (bagus kalau dataset tidak seimbang).\n"
                    "- **Macro F1**: rata-rata per label (bagus untuk melihat label kecil/rare)."
                )
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

                # =========================
                # 4) TAB DETAIL
                # =========================
                tabA, tabB, tabC = st.tabs(["🏷️ Per Label (Top-K)", "🧩 Confusion Matrix (Top-K)", "📈 ROC Micro-Average"])

                # -------------------------------------------------
                # TAB A: Per-label metrics (Top-K by support)
                # -------------------------------------------------
                with tabA:
                    st.markdown('<div class="card"><div class="card-title">🏷️ Performa per Label (Top-K)</div>', unsafe_allow_html=True)

                    support = Y_true.sum(axis=0)  # jumlah true per label
                    order = np.argsort(support)[::-1]

                    top_k = st.slider("Top-K label untuk tabel per-label", 5, 25, 12)

                    rows = []
                    for idx in order[:top_k]:
                        yt = Y_true[:, idx]
                        yp = Y_pred[:, idx]

                        # precision/recall/f1 untuk label ini
                        p = precision_score(yt, yp, zero_division=0)
                        r = recall_score(yt, yp, zero_division=0)
                        f1 = f1_score(yt, yp, zero_division=0)

                        rows.append({
                            "Label": mlb.classes_[idx],
                            "Support (True=1)": int(support[idx]),
                            "Precision": round(float(p), 4),
                            "Recall": round(float(r), 4),
                            "F1": round(float(f1), 4),
                        })

                    per_label_df = pd.DataFrame(rows)
                    st.dataframe(per_label_df, use_container_width=True)

                    st.caption("Support tinggi berarti label sering muncul. Macro F1 akan sensitif untuk label support kecil.")
                    st.markdown("</div>", unsafe_allow_html=True)

                # -------------------------------------------------
                # TAB B: Multi-label confusion matrix (Top-K)
                # -------------------------------------------------
                with tabB:
                    st.markdown('<div class="card"><div class="card-title">🧩 Multilabel Confusion Matrix (Top-K)</div>', unsafe_allow_html=True)
                    st.caption("Menampilkan confusion matrix per label untuk label yang paling sering muncul.")

                    support = Y_true.sum(axis=0)
                    order = np.argsort(support)[::-1]
                    top_k_cm = st.slider("Top-K label untuk confusion matrix", 3, 15, 10)

                    top_idx = order[:top_k_cm]
                    mcm = multilabel_confusion_matrix(Y_true, Y_pred)

                    cols = 5
                    rows = int(np.ceil(top_k_cm / cols))
                    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.4))
                    axes = np.array(axes).reshape(rows, cols)

                    for k, idx in enumerate(top_idx):
                        r = k // cols
                        c = k % cols
                        ax = axes[r, c]
                        tn, fp, fn, tp = mcm[idx].ravel()
                        mat = np.array([[tn, fp], [fn, tp]])
                        ax.imshow(mat)
                        ax.set_title(f"{mlb.classes_[idx]}", fontsize=10)
                        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                        ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=9)
                        ax.set_yticklabels(["True 0", "True 1"], fontsize=9)

                        # angka di kotak
                        for rr in range(2):
                            for cc in range(2):
                                ax.text(cc, rr, str(mat[rr, cc]), ha="center", va="center", fontsize=10)

                    # kosongkan subplot yang sisa
                    for k in range(len(top_idx), rows * cols):
                        r = k // cols
                        c = k % cols
                        axes[r, c].axis("off")

                    fig.suptitle("Multilabel Confusion Matrix (Top-K)", fontsize=13, y=1.02)
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)

                    st.markdown(
                        "- **TP** (True1, Pred1): label benar terdeteksi\n"
                        "- **FN** (True1, Pred0): label ada tapi tidak terdeteksi (miss)\n"
                        "- **FP** (True0, Pred1): label tidak ada tapi diprediksi (false alarm)\n"
                        "- **TN** (True0, Pred0): benar tidak memprediksi label"
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

                # -------------------------------------------------
                # TAB C: ROC curve micro-average
                # -------------------------------------------------
                with tabC:
                    st.markdown('<div class="card"><div class="card-title">📈 ROC Curve (Micro-Average)</div>', unsafe_allow_html=True)
                    st.caption("ROC micro-average memberi gambaran performa keseluruhan untuk klasifikasi label (1 vs 0).")

                    fpr, tpr, _ = roc_curve(Y_true.ravel(), Y_proba.ravel())
                    roc_auc = auc(fpr, tpr)

                    fig, ax = plt.subplots(figsize=(6.0, 3.8))
                    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
                    ax.plot([0, 1], [0, 1], "--")
                    ax.set_title("ROC Curve (Micro-Average)")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend()
                    ax.grid(alpha=0.25)
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)

                    st.caption("AUC makin mendekati 1 → performa makin baik.")
                    st.markdown("</div>", unsafe_allow_html=True)

                # =========================
                # OPTIONAL: tampilkan threshold
                # =========================
                with st.expander("⚙️ Lihat threshold otomatis (opsional)"):
                    st.write(f"Threshold default dari artefak training: **{float(threshold_default):.4f}**")
                    st.caption("Threshold ini dipakai untuk menentukan label yang dianggap 'aktif' (proba >= threshold).")

# =========================================================
# PAGE: INFO SISTEM
# =========================================================
elif menu == "ℹ️ Info Sistem":
    st.markdown('<div class="card"><div class="card-title">ℹ️ Info Sistem</div>', unsafe_allow_html=True)
    st.write(
        "Dashboard ini dibuat untuk:\n"
        "- **Klasifikasi multi-label** jenis penyakit.\n"
        "- Menghasilkan **prioritas penanganan** (skor & kategori).\n"
        "- Menyediakan halaman analisis data & evaluasi model.\n"
        "- Threshold **tidak ditampilkan** (otomatis dari artefak)."
    )
    st.markdown("### Komponen utama")
    st.write("- Model: XGBoost One-vs-Rest\n- Vectorizer: TF-IDF\n- Label: MultiLabelBinarizer\n- Prioritas: rule-based weight + bonus komorbid")
    st.markdown("### Tips penggunaan")
    st.write("- Input teks gunakan format seperti data latih (contoh: `demam dan diare`).\n- Untuk evaluasi & analisis: pastikan `DATA_XLSX_PATH` benar.")
    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
      <b>Catatan:</b> Dashboard ini untuk <i>decision support</i>. Hasil label & prioritas tergantung artefak training dan kualitas input teks.
    </div>
    """,
    unsafe_allow_html=True
)


