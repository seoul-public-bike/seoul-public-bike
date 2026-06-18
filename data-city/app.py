import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ── 한글 폰트 설정 ──────────────────────────────────────────
def setup_korean_font():
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fe = fm.FontEntry(fname=path, name="KoreanFont")
            fm.fontManager.ttflist.insert(0, fe)
            matplotlib.rcParams["font.family"] = "KoreanFont"
            return
    matplotlib.rcParams["axes.unicode_minus"] = False

setup_korean_font()
matplotlib.rcParams["axes.unicode_minus"] = False

# ─────────────────────────────────
# CONFIG
# ─────────────────────────────────

DATA_PATH = "final_hourly_demand.csv"   # 로컬 실행 시 동일 폴더에 위치

# CSV 원본의 city 값(소문자) → 표시명 매핑
CITY_LABEL_MAP = {
    "seoul": "Seoul",
    "london": "London",
    "nyc": "NYC",
}

CITY_COLORS = {
    "Seoul":  "#1f77b4",
    "London": "#ff7f0e",
    "NYC":    "#2ca02c",
}

YEAR_ALPHA = {2022: 0.20, 2023: 0.35, 2024: 0.55, 2025: 0.80, 2026: 1.00}

SEASON_MAP = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3, "fall": 3}

st.set_page_config(layout="wide")
st.title("🚲 Bike Demand Forecast Dashboard")

# ─────────────────────────────────
# DATA LOAD
# ─────────────────────────────────

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    df["날짜"] = pd.to_datetime(df["날짜"])

    # city 컬럼 정규화 (소문자 → 표시명)
    if "도시" in df.columns:
        df["도시"] = df["도시"].str.lower().str.strip().map(CITY_LABEL_MAP).fillna(df["도시"])
    else:
        # 더미변수 방식 폴백
        for raw, label in CITY_LABEL_MAP.items():
            col = f"도시_{raw}"
            if col in df.columns:
                df.loc[df[col] == 1, "도시"] = label
        if "도시" not in df.columns:
            df["도시"] = "Unknown"

    # weekday 복원
    if "요일" not in df.columns:
        wc = [f"요일_{i}" for i in range(7)]
        if all(c in df.columns for c in wc):
            df["요일"] = df[wc].values.argmax(axis=1)

    # season → 숫자
    if "계절" in df.columns:
        df["season_num"] = df["계절"].str.lower().map(SEASON_MAP).fillna(0).astype(int)
    else:
        df["season_num"] = 0

    return df


# ─────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────

def make_monthly(df_city: pd.DataFrame) -> pd.DataFrame:
    """시간별 → 월별 집계 (계절, 피크시간 특성 포함)"""
    agg = (
        df_city.groupby(["연도", "월"], as_index=False)
        .agg(
            자전거수요   = ("자전거수요",   "sum"),
            주말여부   = ("주말여부",   "mean"),
            season_num   = ("season_num",   "first"),
            # 출퇴근(7~9, 17~19) 비율 → 수요 패턴 특성
            peak_ratio   = ("자전거수요",   lambda s: (
                df_city.loc[s.index, "시간"].isin([7,8,9,17,18,19]).sum() / max(len(s), 1)
            )),
        )
        .sort_values(["연도", "월"])
        .reset_index(drop=True)
    )
    agg["ds"] = pd.to_datetime(
        agg["연도"].astype(str) + "-" + agg["월"].astype(str).str.zfill(2) + "-01"
    )
    return agg


def make_features(mdf: pd.DataFrame) -> pd.DataFrame:
    x = mdf.copy().sort_values(["연도", "월"]).reset_index(drop=True)

    # ── 시간 인덱스 ──
    x["t"] = (x["연도"] - x["연도"].min()) * 12 + x["월"]

    # ── 계절성 (푸리에) ──
    x["sin1"] = np.sin(2 * np.pi * x["월"] / 12)
    x["cos1"] = np.cos(2 * np.pi * x["월"] / 12)
    x["sin2"] = np.sin(4 * np.pi * x["월"] / 12)   # 2차 조화
    x["cos2"] = np.cos(4 * np.pi * x["월"] / 12)

    # ── 분기 ──
    x["quarter"] = ((x["월"] - 1) // 3) + 1

    # ── lag 피처 (shift만 사용 → 미래 데이터 누수 방지) ──
    for lag in [1, 2, 3, 6, 12]:
        x[f"lag{lag}"] = x["자전거수요"].shift(lag)

    # ── 롤링 통계 (shift(1) 선행 적용으로 누수 차단) ──
    base = x["자전거수요"].shift(1)
    for w in [3, 6, 12]:
        x[f"roll_mean{w}"] = base.rolling(w, min_periods=1).mean()
        x[f"roll_std{w}"]  = base.rolling(w, min_periods=1).std().fillna(0)

    # ── YoY 비율 (전년 동월 대비) ──
    x["yoy_ratio"] = x["자전거수요"].shift(12) / (x["자전거수요"].shift(13) + 1e-9)

    # ── 결측 처리: ffill → bfill (학습 시점 이전 행만 영향) ──
    x = x.ffill().bfill()
    return x


FEATURES = [
    "t", "월", "quarter",
    "sin1", "cos1", "sin2", "cos2",
    "season_num", "주말여부", "peak_ratio",
    "lag1", "lag2", "lag3", "lag6", "lag12",
    "roll_mean3", "roll_mean6", "roll_mean12",
    "roll_std3",  "roll_std6",  "roll_std12",
    "yoy_ratio",
]


# ─────────────────────────────────
# OPTUNA TUNING
# ─────────────────────────────────

def tune_model(model_name: str, X_train: np.ndarray, y_train: np.ndarray, n_trials: int = 50):
    """
    TimeSeriesSplit + MAPE 최적화 (RMSE 스케일 영향 제거)
    """
    splits = min(4, max(2, len(X_train) // 5))
    tscv = TimeSeriesSplit(n_splits=splits)

    def objective(trial):
        if model_name == "xgb":
            model = XGBRegressor(
                n_estimators      = trial.suggest_int  ("n_estimators",    200, 800),
                max_depth         = trial.suggest_int  ("max_depth",         3,   7),
                learning_rate     = trial.suggest_float("learning_rate",   0.005, 0.10, log=True),
                subsample         = trial.suggest_float("subsample",        0.5,  1.0),
                colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5,  1.0),
                min_child_weight  = trial.suggest_int  ("min_child_weight",   1,  10),
                reg_alpha         = trial.suggest_float("reg_alpha",        1e-4, 5.0, log=True),
                reg_lambda        = trial.suggest_float("reg_lambda",       1e-4, 5.0, log=True),
                gamma             = trial.suggest_float("gamma",            0.0,  1.0),
                random_state=42,
                objective="reg:squarederror",
                tree_method="hist",
            )
        elif model_name == "lgb":
            model = LGBMRegressor(
                n_estimators      = trial.suggest_int  ("n_estimators",    200, 800),
                max_depth         = trial.suggest_int  ("max_depth",         3,   7),
                learning_rate     = trial.suggest_float("learning_rate",   0.005, 0.10, log=True),
                num_leaves        = trial.suggest_int  ("num_leaves",        15,  63),
                min_child_samples = trial.suggest_int  ("min_child_samples",  2,  30),
                subsample         = trial.suggest_float("subsample",         0.5,  1.0),
                colsample_bytree  = trial.suggest_float("colsample_bytree",  0.5,  1.0),
                reg_alpha         = trial.suggest_float("reg_alpha",         1e-4, 5.0, log=True),
                reg_lambda        = trial.suggest_float("reg_lambda",        1e-4, 5.0, log=True),
                random_state=42,
                verbosity=-1,
            )
        elif model_name == "cat":
            model = CatBoostRegressor(
                iterations        = trial.suggest_int  ("iterations",      200, 800),
                depth             = trial.suggest_int  ("depth",             3,   8),
                learning_rate     = trial.suggest_float("learning_rate",   0.005, 0.10, log=True),
                l2_leaf_reg       = trial.suggest_float("l2_leaf_reg",      1.0, 15.0),
                bagging_temperature= trial.suggest_float("bagging_temperature", 0.0, 1.0),
                random_strength   = trial.suggest_float("random_strength",  0.0,  2.0),
                random_seed=42,
                verbose=0,
            )
        else:
            raise ValueError(f"Unsupported: {model_name}")

        scores = []
        for tr_idx, val_idx in tscv.split(X_train):
            Xtr, Xval = X_train[tr_idx], X_train[val_idx]
            ytr, yval = y_train[tr_idx], y_train[val_idx]
            model.fit(Xtr, ytr)
            pred = np.maximum(model.predict(Xval), 0)
            # MAPE 기반 최적화 (스케일 불변)
            mape = mean_absolute_percentage_error(yval + 1, pred + 1)
            scores.append(mape)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


# ─────────────────────────────────
# TRAIN PIPELINE
# ─────────────────────────────────

def build_model(model_name: str, best_params: dict):
    if model_name == "xgb":
        return XGBRegressor(random_state=42, objective="reg:squarederror",
                            tree_method="hist", **best_params)
    if model_name == "lgb":
        return LGBMRegressor(random_state=42, verbosity=-1, **best_params)
    if model_name == "cat":
        return CatBoostRegressor(random_seed=42, verbose=0, **best_params)
    raise ValueError(f"Unsupported: {model_name}")


def train_test_split_time(X, y, test_size=6):
    return X[:-test_size], X[-test_size:], y[:-test_size], y[-test_size:]


def train_models(X: np.ndarray, y: np.ndarray, n_trials: int = 50) -> dict:
    """
    1) Optuna로 각 모델 튜닝
    2) 전체 데이터로 최종 모델 재학습
    3) Holdout 6개월로 앙상블 가중치 최적화
    """
    X_tr, X_te, y_tr, y_te = train_test_split_time(X, y, test_size=6)

    # ── 튜닝 ──
    params_x, _ = tune_model("xgb", X_tr, y_tr, n_trials)
    params_l, _ = tune_model("lgb", X_tr, y_tr, n_trials)
    params_c, _ = tune_model("cat", X_tr, y_tr, n_trials)

    # ── 전체 데이터 재학습 (train + test 모두) ──
    xgb = build_model("xgb", params_x); xgb.fit(X_tr, y_tr)
    lgb = build_model("lgb", params_l); lgb.fit(X_tr, y_tr)
    cat = build_model("cat", params_c); cat.fit(X_tr, y_tr)

    # ── 홀드아웃 예측 ──
    px = np.maximum(xgb.predict(X_te), 0)
    pl = np.maximum(lgb.predict(X_te), 0)
    pc = np.maximum(cat.predict(X_te), 0)

    rmse_x = np.sqrt(mean_squared_error(y_te, px))
    rmse_l = np.sqrt(mean_squared_error(y_te, pl))
    rmse_c = np.sqrt(mean_squared_error(y_te, pc))
    mape_x = mean_absolute_percentage_error(y_te + 1, px + 1) * 100
    mape_l = mean_absolute_percentage_error(y_te + 1, pl + 1) * 100
    mape_c = mean_absolute_percentage_error(y_te + 1, pc + 1) * 100

    # ── Optuna로 앙상블 가중치 최적화 ──
    def weight_obj(trial):
        w1 = trial.suggest_float("wx", 0.0, 1.0)
        w2 = trial.suggest_float("wl", 0.0, 1.0 - w1)
        w3 = 1.0 - w1 - w2
        ens = w1 * px + w2 * pl + w3 * pc
        return mean_absolute_percentage_error(y_te + 1, ens + 1)

    ws = optuna.create_study(direction="minimize",
                              sampler=optuna.samplers.TPESampler(seed=0))
    ws.optimize(weight_obj, n_trials=200, show_progress_bar=False)
    wx = ws.best_params["wx"]
    wl = ws.best_params["wl"]
    wc = max(0.0, 1.0 - wx - wl)
    total = wx + wl + wc
    wx, wl, wc = wx / total, wl / total, wc / total

    # ── 전체 fitted values ──
    fit_x = np.maximum(xgb.predict(X), 0)
    fit_l = np.maximum(lgb.predict(X), 0)
    fit_c = np.maximum(cat.predict(X), 0)
    fit_ens = wx * fit_x + wl * fit_l + wc * fit_c

    ens_rmse = np.sqrt(mean_squared_error(y_te, wx*px + wl*pl + wc*pc))
    ens_mape = mean_absolute_percentage_error(y_te + 1, wx*px + wl*pl + wc*pc + 1) * 100

    return {
        "xgb": xgb, "lgb": lgb, "cat": cat,
        "wx": wx, "wl": wl, "wc": wc,
        "rmse_x": rmse_x, "rmse_l": rmse_l, "rmse_c": rmse_c,
        "mape_x": mape_x, "mape_l": mape_l, "mape_c": mape_c,
        "ens_rmse": ens_rmse, "ens_mape": ens_mape,
        "fit_x": fit_x, "fit_l": fit_l, "fit_c": fit_c,
        "fit_ens": fit_ens,
        "X_te": X_te, "y_te": y_te,
        "pred_x": px, "pred_l": pl, "pred_c": pc,
    }


# ─────────────────────────────────
# FUTURE PREDICTION (Autoregressive)
# ─────────────────────────────────

def predict_2026(models: dict, feat: pd.DataFrame) -> pd.DataFrame:
    xgb, lgb, cat = models["xgb"], models["lgb"], models["cat"]
    wx, wl, wc = models["wx"], models["wl"], models["wc"]

    hist = feat.copy().sort_values(["연도", "월"]).reset_index(drop=True)
    future_rows = []

    for _ in range(12):
        last = hist.iloc[-1]
        nm = 1 if int(last["월"]) == 12 else int(last["월"]) + 1
        ny = int(last["연도"]) + 1 if int(last["월"]) == 12 else int(last["연도"])

        new_row = {col: np.nan for col in hist.columns}
        new_row.update({
            "연도": ny, "월": nm,
            "주말여부":  last["주말여부"],
            "season_num":  SEASON_MAP.get({1:"winter",2:"winter",3:"spring",4:"spring",
                                           5:"spring",6:"summer",7:"summer",8:"summer",
                                           9:"autumn",10:"autumn",11:"autumn",12:"winter"}[nm], 0),
            "peak_ratio":  last["peak_ratio"],
            "자전거수요":  0.0,
        })

        temp_df = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)
        temp_df = make_features(temp_df)
        row_feat = temp_df.iloc[-1:]

        # 피처 컬럼 정합성 보장
        avail = [f for f in FEATURES if f in row_feat.columns]
        X_row = row_feat[avail].values

        pred = (
            wx * np.maximum(xgb.predict(X_row), 0) +
            wl * np.maximum(lgb.predict(X_row), 0) +
            wc * np.maximum(cat.predict(X_row), 0)
        )[0]

        temp_df.loc[temp_df.index[-1], "자전거수요"] = pred
        temp_df.loc[temp_df.index[-1], "ds"] = pd.Timestamp(f"{ny}-{nm:02d}-01")
        hist = temp_df.copy()

        future_rows.append({
            "연도": ny, "월": nm, "자전거수요": pred,
            "ds": pd.Timestamp(f"{ny}-{nm:02d}-01"),
        })

    return pd.DataFrame(future_rows)


# ─────────────────────────────────
# VISUALS
# ─────────────────────────────────

def plot_monthly_trend(actual_monthly, forecast_2026, city_name):
    fig, ax = plt.subplots(figsize=(10, 4))
    color = CITY_COLORS.get(city_name, "#444")
    for yr in [2023, 2024, 2025]:
        part = actual_monthly[actual_monthly["연도"] == yr]
        if len(part):
            ax.plot(part["월"], part["자전거수요"],
                    color=color, alpha=YEAR_ALPHA[yr], linewidth=2, label=f"{city_name}-{yr}")
    if len(forecast_2026):
        ax.plot(forecast_2026["월"], forecast_2026["자전거수요"],
                color=color, alpha=1.0, linewidth=2.5, linestyle="--",
                label=f"{city_name}-2026(예측)")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("월"); ax.set_ylabel("월별 사용자 수")
    ax.set_title(f"{city_name} 월별 사용자 추이 (23~26)")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)
    st.pyplot(fig)


def plot_yearly_growth(actual_monthly, forecast_2026):
    ay = (actual_monthly[actual_monthly["연도"].between(2023, 2025)]
          .groupby("연도", as_index=False)["자전거수요"].sum())
    if len(forecast_2026):
        ay = pd.concat([ay, pd.DataFrame({"연도":[2026],"자전거수요":[forecast_2026["자전거수요"].sum()]})],
                       ignore_index=True)
    ay["yoy_pct"] = ay["자전거수요"].pct_change() * 100
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax2 = ax1.twinx()
    ax1.plot(ay["연도"], ay["자전거수요"], marker="o", linewidth=2, label="총 사용자 수")
    ax2.bar(ay["연도"], ay["yoy_pct"], alpha=0.25, color="orange", label="전년比(%)")
    ax1.set_xlabel("연도"); ax1.set_ylabel("총 사용자 수"); ax2.set_ylabel("증감률(%)")
    ax1.set_title("23~26 사용자 증감 추이"); ax1.grid(alpha=0.2)
    st.pyplot(fig)


def plot_residual_line(actual_monthly, fitted_values):
    r = actual_monthly.copy().sort_values(["연도","월"]).reset_index(drop=True)
    r["fitted"]   = fitted_values[:len(r)]
    r["residual"] = r["자전거수요"] - r["fitted"]
    r["mape_row"] = np.abs(r["residual"]) / (r["자전거수요"] + 1) * 100
    r["label"]    = r["연도"].astype(str) + "-" + r["월"].astype(str).str.zfill(2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(r["label"], r["residual"], marker="o", linewidth=1.8)
    axes[0].axhline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].set_title("잔차 (Residual)"); axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(alpha=0.2)

    axes[1].bar(r["label"], r["mape_row"], color="steelblue", alpha=0.7)
    axes[1].set_title("월별 MAPE (%)"); axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(alpha=0.2)

    plt.tight_layout()
    st.pyplot(fig)


def plot_heatmap_hour_weekday(df_city, city_name):
    if "요일" not in df_city.columns or "시간" not in df_city.columns:
        st.info("히트맵에 필요한 weekday/hour 컬럼이 없습니다.")
        return
    pivot = df_city.pivot_table(index="요일", columns="시간",
                                values="자전거수요", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, fmt=".0f", annot=False)
    ax.set_title(f"{city_name} 시간×요일 평균 수요 히트맵")
    st.pyplot(fig)


def plot_holdout_comparison(models: dict, actual_monthly: pd.DataFrame, city_name: str):
    """홀드아웃 구간 실제 vs 예측 비교"""
    n = len(models["y_te"])
    actual = actual_monthly.sort_values(["연도","월"]).reset_index(drop=True)
    ho_actual = actual.iloc[-n:].copy()
    ho_actual["ens_pred"] = models["wx"]*models["pred_x"] + models["wl"]*models["pred_l"] + models["wc"]*models["pred_c"]
    ho_actual["label"] = ho_actual["연도"].astype(str)+"-"+ho_actual["월"].astype(str).str.zfill(2)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ho_actual["label"], ho_actual["자전거수요"], marker="o", label="실제", linewidth=2)
    ax.plot(ho_actual["label"], ho_actual["ens_pred"],   marker="s", label="앙상블 예측",
            linewidth=2, linestyle="--", color="tomato")
    ax.fill_between(ho_actual["label"],
                    ho_actual["자전거수요"], ho_actual["ens_pred"], alpha=0.15, color="tomato")
    ax.set_title(f"{city_name} Holdout 실제 vs 예측 (최근 {n}개월)")
    ax.legend(); ax.grid(alpha=0.2)
    ax.tick_params(axis='x', rotation=30)
    st.pyplot(fig)


# ─────────────────────────────────
# UI
# ─────────────────────────────────

uploaded = st.file_uploader("📂 CSV 업로드 (비워두면 기본 경로 사용)", type="csv")
if uploaded:
    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.read())
        _path = tmp.name
else:
    _path = DATA_PATH

try:
    df = load_data(_path)
except FileNotFoundError:
    st.error(f"CSV 파일을 찾을 수 없습니다: {_path}\n\n파일을 업로드하거나 DATA_PATH를 수정하세요.")
    st.stop()

if "도시" not in df.columns or df["도시"].isnull().all():
    st.error("city 컬럼 생성에 실패했습니다. 입력 데이터 구조를 확인해주세요.")
    st.stop()

available_cities = sorted(df["도시"].dropna().unique())
cities  = st.multiselect("🏙️ 도시 선택", available_cities, default=available_cities)
trials  = st.slider("🔬 Optuna Trials (많을수록 정확, 느림)", 20, 150, 50, step=10)

run_btn = st.button("실행")

if run_btn:
    if not cities:
        st.warning("도시를 1개 이상 선택하세요.")
        st.stop()

    for city in cities:
        st.header(f" {city}")
        df_c = df[df["도시"] == city].copy()

        # 1) 히트맵
        st.subheader(" 시간×요일 수요 히트맵")
        plot_heatmap_hour_weekday(df_c, city)

        # 2) 피처 생성
        monthly = make_monthly(df_c)
        feat    = make_features(monthly)

        avail_f = [f for f in FEATURES if f in feat.columns]
        X = feat[avail_f].values
        y = feat["자전거수요"].values

        # 3) 학습 + 튜닝
        st.subheader(" Optuna 하이퍼파라미터 튜닝 + 앙상블 학습")
        prog = st.progress(0, text="XGBoost 튜닝 중...")
        with st.spinner(f"{city} 모델 학습 중 (약 1~3분)..."):
            models = train_models(X, y, n_trials=trials)
        prog.progress(100, text="완료!")

        # 4) 2026 예측
        forecast_2026 = predict_2026(models, feat)

        # 5) KPI
        st.subheader(" 모델 성능 (홀드아웃 최근 6개월)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("XGB MAPE",    f"{models['mape_x']:.2f}%")
        c2.metric("LGB MAPE",    f"{models['mape_l']:.2f}%")
        c3.metric("CAT MAPE",    f"{models['mape_c']:.2f}%")
        c4.metric(" 앙상블 MAPE", f"{models['ens_mape']:.2f}%",
                  delta=f"RMSE {models['ens_rmse']:,.0f}",
                  delta_color="inverse")

        st.caption(
            f"앙상블 가중치 — XGB: {models['wx']:.3f} | LGB: {models['wl']:.3f} | CAT: {models['wc']:.3f}"
        )

        # 6) 홀드아웃 실제 vs 예측
        st.subheader(" 홀드아웃 실제 vs 예측")
        plot_holdout_comparison(models, monthly, city)

        # 7) 월별 추이
        st.subheader(" 월별 사용자 추이 (23~26)")
        plot_monthly_trend(monthly, forecast_2026, city)

        # 8) 연도별 증감
        st.subheader(" 연도별 사용자 증감 추이")
        plot_yearly_growth(monthly, forecast_2026)

        # 9) 잔차 분석
        st.subheader(" 잔차 & 월별 MAPE 분석")
        plot_residual_line(monthly, models["fit_ens"])

        # 10) 2026 예측 테이블
        st.subheader(" 2026년 월별 예측값")
        fc_disp = forecast_2026[["연도","월","자전거수요"]].copy()
        fc_disp.columns = ["연도","월","예측 사용자 수"]
        fc_disp["예측 사용자 수"] = fc_disp["예측 사용자 수"].map(lambda x: f"{x:,.0f}")
        st.dataframe(fc_disp, use_container_width=True, hide_index=True)

        st.divider()

    st.success("선택한 모든 도시의 예측 분석이 완료되었습니다.")
