import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from sklearn.base import clone

# 선택 라이브러리 (없으면 해당 섹션 비활성)
try:
    import statsmodels.api as sm
except Exception:
    sm = None

try:
    import shap
except Exception:
    shap = None


# -----------------------------
# 검증용 유틸 함수들
# -----------------------------
def repeated_split_validation(clf, X, y, n_repeats=20, test_size=0.25):
    """
    여러 random_state로 train/test split 반복하면서
    AUC / Precision / Recall 분포 확인
    """
    from sklearn.model_selection import train_test_split

    rows = []
    for seed in range(n_repeats):
        model = clone(clf)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=seed
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)

        rows.append(
            {
                "seed": seed,
                "auc": roc_auc_score(y_te, prob),
                "precision": precision_score(y_te, pred, zero_division=0),
                "recall": recall_score(y_te, pred, zero_division=0),
                "brier": brier_score_loss(y_te, prob),
            }
        )
    return pd.DataFrame(rows)


def kfold_auc_validation(clf, X, y, n_splits=5):
    """
    StratifiedKFold 기반 AUC 교차검증
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for tr_idx, te_idx in skf.split(X, y):
        model = clone(clf)
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        aucs.append(roc_auc_score(y_te, prob))

    return np.array(aucs)


# --------------------------------
# 0) 기본 설정
# --------------------------------
st.set_page_config(page_title="계량적 요인 변화에 따른 이직률 예측 모델", layout="wide")
st.title("계량적 요인 변화에 따른 이직률 예측 모델")

# 잔차화에 사용할 전역 계수 (salary_std → log_salary_vs_industry)
gamma_industry_slope = None
gamma_industry_intercept = None
use_industry_resid = False  # 나중에 features 선택 시 사용

# --------------------------------
# 1) 사이드바: 업로드 & 옵션
# --------------------------------
st.sidebar.header("데이터 업로드")

uploaded_file = st.sidebar.file_uploader("ERP 데이터 파일 업로드", type=["xlsx", "csv"])
industry_file = st.sidebar.file_uploader(
    "산업 평균 임금 데이터 (CSV: year, tenure, industry_avg_wage)", type=["csv"]
)

use_industry_norm = st.sidebar.checkbox("산업 평균 임금 기반 상대임금 사용", value=True)

# --------------------------------
# 2) 산업 평균 임금 데이터 전처리
# --------------------------------
ind_ref = None
if industry_file is not None:
    try:
        industry_file.seek(0)
        ind_df = pd.read_csv(industry_file)

        # 컬럼 이름 정리 (대소문자 무시)
        cols = {c.lower(): c for c in ind_df.columns}
        year_col = cols.get("year")
        ten_col = cols.get("tenure")
        wage_col = (
            cols.get("industry_avg_wage")
            or cols.get("industry_avg_wage_year")
            or cols.get("avg_wage")
        )

        if (year_col is None) or (ten_col is None) or (wage_col is None):
            st.warning(
                "산업 평균 임금 CSV에서 year, tenure, industry_avg_wage 컬럼을 찾지 못했습니다.\n"
                "예시 헤더: year, tenure, industry_avg_wage"
            )
        else:
            ind_ref = ind_df[[year_col, ten_col, wage_col]].copy()
            ind_ref.columns = ["year", "tenure_group", "industry_avg_wage_year"]

            # 숫자형/단위 정리
            ind_ref["year"] = pd.to_numeric(ind_ref["year"], errors="coerce")
            ind_ref["industry_avg_wage_year"] = (
                ind_ref["industry_avg_wage_year"]
                .astype(str)
                .str.replace(",", "", regex=False)
            )
            ind_ref["industry_avg_wage_year"] = pd.to_numeric(
                ind_ref["industry_avg_wage_year"], errors="coerce"
            )

    except Exception as e:
        st.warning(f"산업 평균 임금 파일을 읽는 중 오류가 발생했습니다: {e}")
        ind_ref = None

# --------------------------------
# 3) ERP 데이터 로드
# --------------------------------
if uploaded_file is None:
    st.info("📂 왼쪽 사이드바에서 ERP 데이터를 업로드하면 예측을 시작합니다.")
    st.stop()

try:
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"ERP 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()

# 한글 컬럼 → 내부 변수명 매핑
col_mapping = {
    "사번": "employeeID",
    "직원번호": "employeeID",
    "성과등급": "performanceGrade",
    "근속기간": "tenure",
    "근속": "tenure",
    "성별": "gender",
    "연령": "age",
    "나이": "age",
    "통근시간": "commuteTime",
    "연봉": "salary",
    "연봉(만원)": "salary",
    "직무만족도": "jobSatisfaction",
    "퇴사여부": "attrition",
    "퇴사": "attrition",
    "기준연도": "pay_year",
    "급여연도": "pay_year",
    "부서": "department",
    "소속부서": "department",
    "팀": "department",
    "직급": "position",
    "직위": "position",
}

rename_map = {k: v for k, v in col_mapping.items() if k in data.columns}
raw = data.rename(columns=rename_map).copy()

required_cols = [
    "performanceGrade",
    "tenure",
    "gender",
    "age",
    "commuteTime",
    "salary",
    "jobSatisfaction",
    "pay_year",
    "attrition",
]

missing = [c for c in required_cols if c not in raw.columns]
if missing:
    st.error(f"다음 필수 컬럼이 없습니다: {missing}")
    st.stop()

# 0/1로 들어온다고 가정하지만 안전하게 숫자로 캐스팅
raw["attrition"] = pd.to_numeric(raw["attrition"], errors="coerce")

st.subheader("업로드 데이터 미리보기")

preview_cols = [c for c in raw.columns if c != "employeeID"]
st.dataframe(raw[preview_cols].head())

# --------------------------------
# 3-1) Salary / Tenure 분포 및 극단값 체크 (모델에는 반영 안 함)
# --------------------------------
st.subheader("Salary / Tenure 분포 및 극단값 체크 (진단용)")

with st.expander("연봉(salary) / 근속(tenure) 분포 확인 (모델에는 반영하지 않습니다)"):
    for col in ["salary", "tenure"]:
        if col in raw.columns:
            try:
                raw[col] = pd.to_numeric(raw[col], errors="coerce")
                desc = raw[col].describe(percentiles=[0.01, 0.05, 0.95, 0.99])
                st.markdown(f"**{col} 분포 요약**")
                st.table(desc.to_frame("value"))
            except Exception as e:
                st.warning(f"{col} 분포 요약 계산 중 오류: {e}")
    st.caption(
        "※ 이 정보는 단순 진단용입니다. 상/하위 1% 값이 나머지 값들과 크게 다르지 않다면, "
        "극단치로 인한 계수 왜곡은 크지 않을 가능성이 높습니다."
    )

# --------------------------------
# 4) 파생변수 생성: 성과/통근/만족도/근속구간
# --------------------------------
def map_perf(x: str) -> str:
    if x in ["S", "A", "High", "HIGH", "high"]:
        return "High"
    elif x in ["B", "Mid", "MID", "mid"]:
        return "Mid"
    elif x in ["C", "D", "Low", "LOW", "low"]:
        return "Low"
    else:
        return "Mid"


raw["perf3"] = raw["performanceGrade"].astype(str).apply(map_perf)

commute_map = {
    "≤30m": "≤60m",
    "30m<": "≤60m",
    "30분이내": "≤60m",
    "60분이내": "≤60m",
    "≤60m": "≤60m",
    "30~60m": "≤60m",
    "30-60m": "≤60m",
    "<60m": "≤60m",
    "60~90m": "61–120m",
    "≤90m": "61–120m",
    "61–120m": "61–120m",
    "90분이내": "61–120m",
    "≥120m": "≥120m",
    "120분이상": "≥120m",
    ">120m": "≥120m",
}
raw["comm3"] = raw["commuteTime"].astype(str).map(commute_map).fillna("≤60m")


def map_jobsat(x: str) -> str:
    if x in ["높음", "매우 만족", "상", "다소 높음"]:
        return "높음"
    elif x in ["보통", "중간"]:
        return "보통"
    elif x in ["다소 낮음", "낮음", "불만족", "하"]:
        return "낮음"
    else:
        return "보통"


raw["jobSat3"] = raw["jobSatisfaction"].astype(str).apply(map_jobsat)


def map_tenure_group(t):
    try:
        x = float(t)
    except Exception:
        return np.nan
    if pd.isna(x):
        return np.nan
    if x < 1:
        return "<1y"
    elif x < 3:
        return "1-3y"
    elif x < 5:
        return "3-5y"
    elif x < 10:
        return "5-10y"
    else:
        return "10y+"


raw["tenure_group"] = raw["tenure"].apply(map_tenure_group)

# 수치형 캐스팅
raw["tenure"] = pd.to_numeric(raw["tenure"], errors="coerce")
raw["age"] = pd.to_numeric(raw["age"], errors="coerce")
raw["salary"] = pd.to_numeric(raw["salary"], errors="coerce")
raw["pay_year"] = pd.to_numeric(raw["pay_year"], errors="coerce")

# --------------------------------
# 5) 산업 평균 임금 매칭 + 상대임금(외부) 계산
# --------------------------------
raw["industry_avg_wage_year"] = np.nan
raw["industry_ref_year"] = np.nan
raw["salary_vs_industry"] = np.nan
raw["log_salary_vs_industry"] = np.nan

if use_industry_norm and (ind_ref is not None):
    try:
        ref = ind_ref.dropna(subset=["year", "tenure_group", "industry_avg_wage_year"]).copy()
        if not ref.empty:

            def find_industry(row):
                py = row["pay_year"]
                tg = row["tenure_group"]
                if pd.isna(py) or pd.isna(tg):
                    return pd.Series([np.nan, np.nan])
                sub = ref[ref["tenure_group"] == tg]
                if sub.empty:
                    sub = ref
                idx = (sub["year"] - py).abs().idxmin()
                return pd.Series(
                    [sub.loc[idx, "industry_avg_wage_year"], sub.loc[idx, "year"]]
                )

            raw[["industry_avg_wage_year", "industry_ref_year"]] = raw.apply(
                find_industry, axis=1
            )

            # 천원/만원 단위 보정
            med_salary = raw["salary"].median()
            med_ind = raw["industry_avg_wage_year"].median()
            if pd.notna(med_salary) and pd.notna(med_ind) and med_ind > 0:
                if med_salary / med_ind > 5:
                    raw["industry_avg_wage_year"] *= 10
                    st.info(
                        "⚙️ 산업 평균 임금이 천원 단위로 인식되어 10배 보정했습니다 (만원 단위로 환산)."
                    )

            valid = raw["industry_avg_wage_year"] > 0
            raw.loc[valid, "salary_vs_industry"] = (
                raw.loc[valid, "salary"] / raw.loc[valid, "industry_avg_wage_year"]
            )
            raw.loc[valid, "log_salary_vs_industry"] = np.log(
                raw.loc[valid, "salary_vs_industry"]
            )
    except Exception as e:
        st.warning(f"산업 평균 임금 기반 상대임금 계산 중 오류: {e}")

# --------------------------------
# 6) 내부 직급 기준 상대임금(내부) 계산
# --------------------------------
raw["salary_vs_internal"] = np.nan
raw["log_salary_vs_internal"] = np.nan

if "position" in raw.columns:
    try:
        grp_cols = ["position", "pay_year"]
        base_tmp = raw.dropna(subset=["salary"] + grp_cols).copy()
        if not base_tmp.empty:
            mean_by_pos_year = base_tmp.groupby(grp_cols)["salary"].transform("mean")
            raw.loc[base_tmp.index, "salary_vs_internal"] = (
                base_tmp["salary"] / mean_by_pos_year
            )
            valid_int = raw["salary_vs_internal"] > 0
            raw.loc[valid_int, "log_salary_vs_internal"] = np.log(
                raw[valid_int]["salary_vs_internal"]
            )
    except Exception as e:
        st.warning(f"내부 직급 기준 상대임금 계산 중 오류: {e}")

# --------------------------------
# 7) 절대 연봉 표준화 (연도 기준)
# --------------------------------
raw["salary_std"] = np.nan


def _zscore(x):
    if x.std(ddof=0) == 0:
        return x - x.mean()
    return (x - x.mean()) / x.std(ddof=0)


try:
    raw["salary_std"] = raw.groupby("pay_year")["salary"].transform(_zscore)
except Exception:
    pass

# --------------------------------
# 7-1) 잔차화: log_salary_vs_industry를 salary_std에 대해 orthogonalization
# --------------------------------
raw["log_salary_vs_industry_resid"] = np.nan

if use_industry_norm and ("log_salary_vs_industry" in raw.columns) and ("salary_std" in raw.columns):
    mask_resid = raw["log_salary_vs_industry"].notna() & raw["salary_std"].notna()
    if mask_resid.sum() >= 10 and raw.loc[mask_resid, "salary_std"].std(ddof=0) > 0:
        try:
            A = raw.loc[mask_resid, "salary_std"].values
            B = raw.loc[mask_resid, "log_salary_vs_industry"].values
            gamma_industry_slope, gamma_industry_intercept = np.polyfit(A, B, 1)
            raw.loc[mask_resid, "log_salary_vs_industry_resid"] = (
                B - (gamma_industry_slope * A + gamma_industry_intercept)
            )
            use_industry_resid = True
            st.info(
                "log_salary_vs_industry 변수를 salary_std에 대해 선형 회귀 후 잔차(log_salary_vs_industry_resid)를 생성했습니다.\n"
                "- salary_std: 절대 연봉 수준(연도 내 z-score)\n"
                "- log_salary_vs_industry_resid: 절대 연봉 수준으로 설명되지 않는 '추가적인 산업 대비 프리미엄'"
            )
        except Exception as e:
            st.warning(f"잔차화 수행 중 오류가 발생하여 원래 log_salary_vs_industry를 사용합니다: {e}")
            use_industry_resid = False
    else:
        st.info(
            "salary_std / log_salary_vs_industry 데이터가 충분하지 않거나 분산이 거의 없어 잔차화를 생략하고, "
            "원래 log_salary_vs_industry 변수를 사용합니다."
        )
        use_industry_resid = False

for c in ["log_salary_vs_industry", "log_salary_vs_internal", "log_salary_vs_industry_resid"]:
    if c in raw.columns:
        raw[c] = raw[c].fillna(0.0)

# --------------------------------
# 8) 모델 피처 정의 및 학습
# --------------------------------
features = [
    "perf3",
    "tenure",
    "gender",
    "age",
    "comm3",
    "salary_std",
    "jobSat3",
]

# 산업 기준 상대임금: 잔차화 사용 여부에 따라 선택
if use_industry_norm and (
    (use_industry_resid and raw["log_salary_vs_industry_resid"].notna().any())
    or ((not use_industry_resid) and raw["log_salary_vs_industry"].notna().any())
):
    if use_industry_resid and raw["log_salary_vs_industry_resid"].notna().any():
        features.append("log_salary_vs_industry_resid")
    elif raw["log_salary_vs_industry"].notna().any():
        features.append("log_salary_vs_industry")

if raw["log_salary_vs_internal"].notna().any():
    features.append("log_salary_vs_internal")

num_cols = ["tenure", "age", "salary_std"]
if "log_salary_vs_industry_resid" in features:
    num_cols.append("log_salary_vs_industry_resid")
elif "log_salary_vs_industry" in features:
    num_cols.append("log_salary_vs_industry")
if "log_salary_vs_internal" in features:
    num_cols.append("log_salary_vs_internal")

cat_cols = ["perf3", "gender", "comm3", "jobSat3"]

for c in num_cols:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw[c] = raw[c].fillna(raw[c].median())
for c in cat_cols:
    raw[c] = raw[c].fillna("Unknown")

X = raw[features].copy()
Y = raw["attrition"]

y_unique = pd.Series(Y).dropna().unique()
have_labels = set(y_unique) <= {0, 1}
two_class = have_labels and (len(set(y_unique)) >= 2)

pre = ColumnTransformer(
    [
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

clf = Pipeline(
    [
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=600, C=0.7)),
    ]
)

Xtr = Ytr = None
brier_main = np.nan

if two_class:
    Xtr, Xte, ytr, yte = train_test_split(
        X, Y, test_size=0.25, random_state=42, stratify=Y
    )
    Ytr = ytr
    clf.fit(Xtr, ytr)
    probs_te = clf.predict_proba(Xte)[:, 1]
    preds = (probs_te >= 0.5).astype(int)
    auc = roc_auc_score(yte, probs_te)
    acc = accuracy_score(yte, preds)
    prec = precision_score(yte, preds)
    rec = recall_score(yte, preds)
    brier_main = brier_score_loss(yte, probs_te)
else:
    clf.fit(X)
    auc = acc = prec = rec = float("nan")

# 전체 데이터 예측
probs_all = clf.predict_proba(X)[:, 1]
raw["pred_prob"] = probs_all
raw["risk_band"] = np.where(
    probs_all >= 0.6, "High", np.where(probs_all >= 0.3, "Medium", "Low")
)

# --------------------------------
# 9) 사이드바 단건 입력 (What-if)
# --------------------------------
st.sidebar.header("단건 입력 (What-if 시뮬레이션)")
perf_in = st.sidebar.selectbox("성과 등급(3단계)", ["High", "Mid", "Low"], index=1)
tenure_in = st.sidebar.number_input("근속기간(년)", 0.0, 40.0, 4.0, step=0.1)
gender_in = st.sidebar.selectbox("성별", ["Male", "Female"])
age_in = st.sidebar.number_input("연령", 18, 70, 35)
comm_in = st.sidebar.selectbox("통근 시간(3단계)", ["≤60m", "61–120m", "≥120m"], index=0)
salary_in = st.sidebar.number_input("연봉(만원)", 0, 30000, 4200, step=100)
jobSat_in = st.sidebar.selectbox("직무만족도(3단계)", ["높음", "보통", "낮음"], index=1)
payyear_in = st.sidebar.number_input("급여/평가 기준연도(pay_year)", 2000, 2100, 2024)

# 직급 (내부 비교용) 입력
if "position" in raw.columns and raw["position"].notna().any():
    pos_raw = raw["position"].dropna().astype(str).unique()

    pos_order = [
        "Associate",
        "Senior Associate",
        "Manager",
        "Senior Manager",
        "General Manager",
    ]

    def _pos_key(x):
        return (
            pos_order.index(x)
            if x in pos_order
            else len(pos_order) + sorted(pos_raw.tolist()).index(x)
        )

    pos_sorted = sorted(pos_raw, key=_pos_key)

    default_idx = 0
    if "Manager" in pos_sorted:
        default_idx = pos_sorted.index("Manager")

    position_in = st.sidebar.selectbox(
        "직급 (내부 비교용, 예: Manager)",
        pos_sorted,
        index=default_idx,
    )
else:
    position_in = st.sidebar.text_input("직급 (내부 비교용, 예: Manager)", value="Manager")


# 9-1 연도 기준 salary_std 추정
if raw["pay_year"].notna().any():
    same_year = raw[raw["pay_year"] == payyear_in]
    if same_year.empty:
        idx_y = (raw["pay_year"] - payyear_in).abs().idxmin()
        ref_year = raw.loc[idx_y, "pay_year"]
        same_year = raw[raw["pay_year"] == ref_year]
    if same_year["salary"].std(ddof=0) == 0:
        salary_std_in = salary_in - same_year["salary"].mean()
    else:
        salary_std_in = (salary_in - same_year["salary"].mean()) / same_year[
            "salary"
        ].std(ddof=0)
else:
    salary_std_in = 0.0

# 9-2 산업 기준 상대임금 (근속구간 + 연도)
log_ratio_ind_in = 0.0
tg_in = map_tenure_group(tenure_in)
if use_industry_norm and (ind_ref is not None) and (tg_in is not None):
    ref2 = ind_ref.dropna(subset=["year", "tenure_group", "industry_avg_wage_year"])
    sub = ref2[ref2["tenure_group"] == tg_in]
    if sub.empty:
        sub = ref2
    if not sub.empty:
        idx = (sub["year"] - payyear_in).abs().idxmin()
        ind_wage = sub.loc[idx, "industry_avg_wage_year"]
        if pd.notna(ind_wage) and ind_wage > 0:
            med_salary = raw["salary"].median()
            if pd.notna(med_salary) and med_salary / ind_wage > 5:
                ind_wage *= 10
            ratio_ind = salary_in / ind_wage
            if ratio_ind > 0:
                log_ratio_ind_in = float(np.log(ratio_ind))

# 9-3 내부 직급 기준 상대임금 (같은 직급 + 연도)
log_ratio_int_in = 0.0
if "position" in raw.columns:
    base_pos = raw[raw["position"].astype(str) == str(position_in)]
    if base_pos.empty:
        base_pos = raw
    same_year_pos = base_pos[base_pos["pay_year"] == payyear_in]
    if same_year_pos.empty and not base_pos.empty:
        idx = (base_pos["pay_year"] - payyear_in).abs().idxmin()
        ref_year2 = base_pos.loc[idx, "pay_year"]
        same_year_pos = base_pos[base_pos["pay_year"] == ref_year2]
    if not same_year_pos.empty:
        mean_internal = same_year_pos["salary"].mean()
        if mean_internal > 0:
            ratio_int = salary_in / mean_internal
            if ratio_int > 0:
                log_ratio_int_in = float(np.log(ratio_int))

x_dict = {
    "perf3": perf_in,
    "tenure": tenure_in,
    "gender": gender_in,
    "age": age_in,
    "comm3": comm_in,
    "salary_std": salary_std_in,
    "jobSat3": jobSat_in,
}

if use_industry_norm:
    if use_industry_resid and ("log_salary_vs_industry_resid" in features) and (
        gamma_industry_slope is not None
    ):
        log_ratio_ind_resid_in = log_ratio_ind_in - (
            gamma_industry_slope * salary_std_in + gamma_industry_intercept
        )
        x_dict["log_salary_vs_industry_resid"] = log_ratio_ind_resid_in
    elif "log_salary_vs_industry" in features:
        x_dict["log_salary_vs_industry"] = log_ratio_ind_in

if "log_salary_vs_internal" in features:
    x_dict["log_salary_vs_internal"] = log_ratio_int_in

x_in = pd.DataFrame([x_dict])

# --------------------------------
# 10) 단건 예측
# --------------------------------
prob = float(clf.predict_proba(x_in)[0, 1])
risk = "High" if prob >= 0.6 else ("Medium" if prob >= 0.3 else "Low")

colA, colB, colC = st.columns([1.2, 0.8, 1])

with colA:
    st.subheader("단건 예측")
    st.metric("퇴사 확률", f"{prob*100:.1f}%")
    st.write(
        f"**위험도:** "
        f"{'🔴 High' if risk=='High' else '🟠 Medium' if risk=='Medium' else '🟢 Low'}"
    )
    st.progress(min(max(prob, 0), 1))

    with st.expander("퇴사 확률이 의미하는 것 보기"):
        st.markdown(
            """
**퇴사 확률**은  
현재 입력된 조건(근속, 연봉, 직무만족도, 통근시간 등)이 유지된다는 가정 하에,  
과거 데이터 패턴을 기반으로 **향후 약 1년 내 이직/퇴사할 가능성(확률)** 을 의미합니다.

- 예: 퇴사확률이 `0.73` 이라면  
  → “지금과 유사한 조건을 가진 직원 100명 중 약 73명이 1년 내 퇴사했던 패턴”에 가깝다는 뜻입니다.  
- **예언이 아니라, ‘리스크 수준’에 대한 통계적 추정치**입니다.
- 조건이 바뀌면(연봉 인상, 부서이동, 팀 이직률 개선 등) **퇴사확률도 함께 변합니다.**
"""
        )

with colC:
    st.subheader("모델 성능 (기본 split 기준)")
    if two_class:
        c1, c2 = st.columns(2)
        c1.metric("ROC-AUC", f"{auc:.3f}")
        c2.metric("Accuracy", f"{acc:.3f}")
        c1.metric("Precision", f"{prec:.3f}")
        c2.metric("Recall", f"{rec:.3f}")
        st.caption(f"Brier score (확률 보정 정도): **{brier_main:.3f}**")
    else:
        st.write("라벨이 한 클래스뿐이라 성능지표를 계산하지 않았습니다.")

# --------------------------------
# 11) 특징 기여도 (단건 기준, 상위 5)
# --------------------------------
st.subheader("특징 기여도 (단건 기준, 상위 5)")

oh = clf.named_steps["pre"].named_transformers_["cat"]
cat_names = list(oh.get_feature_names_out(cat_cols))
feat_names_tr = list(np.array(num_cols)) + cat_names

x_trans = clf.named_steps["pre"].transform(x_in)
coefs = clf.named_steps["lr"].coef_[0]
if hasattr(x_trans, "toarray"):
    x_vec = x_trans.toarray()[0]
else:
    x_vec = x_trans[0]
contrib_values = x_vec * coefs

_contrib = pd.DataFrame({"feature": feat_names_tr, "value": contrib_values})


def agg_onehots(df, base_cols, cats):
    out = []
    for c in base_cols:
        out.append(
            {"feature": c, "value": float(df.loc[df["feature"] == c, "value"].sum())}
        )
    for cat in cats:
        mask = df["feature"].str.startswith(cat + "_")
        out.append(
            {"feature": cat, "value": float(df.loc[mask, "value"].sum())}
        )
    return pd.DataFrame(out)


contrib_agg = agg_onehots(_contrib, num_cols, cat_cols)
order = [
    "jobSat3",
    "perf3",
    "salary_std",
    "log_salary_vs_industry_resid",
    "log_salary_vs_industry",
    "log_salary_vs_internal",
    "tenure",
    "age",
    "comm3",
    "gender",
]
contrib_agg = (
    contrib_agg.set_index("feature").reindex(order).dropna().reset_index()
)
contrib_top = contrib_agg.sort_values(
    by="value", key=lambda s: s.abs(), ascending=False
).head(5)

st.bar_chart(contrib_top.set_index("feature")["value"])

# --------------------------------
# 12) 조직별 리스크 (재직자 기준)
# --------------------------------
st.markdown("---")
st.subheader("조직별 이직/퇴사 리스크 현황 (재직자 기준)")

col_dept, col_pos = st.columns(2)

with col_dept:
    st.markdown("### 부서별 리스크")
    if "department" in raw.columns:
        base = raw[raw["attrition"] == 0].copy()
        if base.empty:
            st.info("재직자(attrition=0) 데이터가 없어 부서별 리스크를 표시하지 않습니다.")
        else:
            dept_mean = (
                base.groupby("department")["pred_prob"]
                .mean()
                .sort_values(ascending=False)
            )
            st.caption("부서별 평균 예측 이직/퇴사 확률 (재직자 기준)")
            st.bar_chart(dept_mean)

            dept_risk = (
                base.pivot_table(
                    index="department",
                    columns="risk_band",
                    values="pred_prob",
                    aggfunc="count",
                )
                .fillna(0)
                .reindex(columns=["High", "Medium", "Low"])
            )
            st.caption("부서별 High/Medium/Low 인원 수 (재직자 기준)")
            st.dataframe(dept_risk.astype(int))
    else:
        st.info("ERP 데이터에 부서(부서/소속부서/팀) 컬럼이 없습니다.")

with col_pos:
    st.markdown("### 직급별 리스크")
    if "position" in raw.columns:
        base = raw[raw["attrition"] == 0].copy()
        if base.empty:
            st.info("재직자(attrition=0) 데이터가 없어 직급별 리스크를 표시하지 않습니다.")
        else:
            pos_mean = base.groupby("position")["pred_prob"].mean()

            pos_order = [
                "Associate",
                "Senior Associate",
                "Manager",
                "Senior Manager",
                "General Manager",
            ]

            others = [p for p in pos_mean.index if p not in pos_order]
            index_order = [p for p in pos_order if p in pos_mean.index] + sorted(others)

            pos_mean = pos_mean.reindex(index_order).dropna()

            st.caption("직급별 평균 예측 이직/퇴사 확률 (재직자 기준)")

            chart_df = pos_mean.reindex(index_order).dropna().reset_index()
            chart_df.columns = ["position", "pred_prob"]

            chart = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("position:N", sort=index_order, title="직급"),
                    y=alt.Y("pred_prob:Q", title="평균 예측 이직/퇴사 확률"),
                )
                .properties(height=300)
            )

            st.altair_chart(chart, use_container_width=True)

            pos_risk = (
                base.pivot_table(
                    index="position",
                    columns="risk_band",
                    values="pred_prob",
                    aggfunc="count",
                )
                .fillna(0)
            )

            pos_risk = pos_risk.reindex(index_order)
            pos_risk = pos_risk.reindex(columns=["High", "Medium", "Low"])

            st.caption("직급별 High/Medium/Low 인원 수 (재직자 기준)")
            st.dataframe(pos_risk.astype(int))
    else:
        st.info("ERP 데이터에 직급(직급/직위) 컬럼이 없습니다.")

# --------------------------------
# 13) 임금 경쟁력 분석 (산업 평균 vs 내부)
# --------------------------------
st.markdown("---")
st.subheader("임금 경쟁력 분석 (산업 평균 vs 내부)")

if (not use_industry_norm) or raw["industry_avg_wage_year"].isna().all():
    st.info(
        "산업 평균 임금 CSV를 업로드하고, 사이드바에서 "
        "'산업 평균 임금 기반 상대임금 사용'을 체크하면 임금 경쟁력 분석을 볼 수 있습니다.\n\n"
        "예시 컬럼: year, tenure, industry_avg_wage"
    )
elif "position" not in raw.columns:
    st.warning(
        "ERP 데이터에 직급(직급/직위) 컬럼이 없어 내부 공정성 분석(직급 대비 상대임금)을 계산할 수 없습니다."
    )
else:
    base = raw[raw["attrition"] == 0].copy()
    if base.empty:
        st.info("재직자 데이터가 없어 임금 경쟁력을 표시할 수 없습니다.")
    else:
        if base["salary_vs_internal"].isna().all():
            base["salary_vs_internal"] = (
                base["salary"]
                / base.groupby(["position", "pay_year"])["salary"].transform("mean")
            )

        col_y, col_t = st.columns(2)

        with col_y:
            year_options = sorted(raw["pay_year"].dropna().unique())
            if len(year_options) == 0:
                st.info("임금 경쟁력 분석에 사용할 연도가 없습니다.")
                year_sel = None
            else:
                year_sel = st.selectbox(
                    "분석 기준 연도(pay_year)",
                    year_options,
                    index=len(year_options) - 1,
                )

        with col_t:
            ten_options = base["tenure_group"].dropna().unique()
            order_ten = ["<1y", "1-3y", "3-5y", "5-10y", "10y+"]
            ten_sorted = sorted(
                ten_options,
                key=lambda x: order_ten.index(x) if x in order_ten else 999,
            )
            tenure_sel = st.selectbox("근속 구간(tenure_group)", ten_sorted)

        if year_sel is not None:
            sub = base[
                (base["pay_year"] == year_sel)
                & (base["tenure_group"] == tenure_sel)
                & base["industry_avg_wage_year"].notna()
            ]
            if sub.empty:
                st.info("선택한 조건에 해당하는 재직자가 없거나 산업 평균 매칭값이 없습니다.")
            else:
                mean_salary = sub["salary"].mean()
                mean_ind = sub["industry_avg_wage_year"].mean()
                ratio_ext = mean_salary / mean_ind if mean_ind > 0 else np.nan
                ratio_int = sub["salary_vs_internal"].mean()

                if "industry_ref_year" in sub.columns and sub["industry_ref_year"].notna().any():
                    ref_year = int(round(sub["industry_ref_year"].mode().iloc[0]))
                else:
                    ref_year = year_sel

                def label_external(r):
                    if pd.isna(r):
                        return "산업 평균 정보 없음"
                    if r >= 1.05:
                        return "외부 경쟁력 ↑"
                    elif r <= 0.95:
                        return "외부 경쟁력 ↓"
                    else:
                        return "외부 경쟁력 보통"

                def label_internal(r):
                    if pd.isna(r):
                        return "내부 공정성 평가 불가"
                    if 0.97 <= r <= 1.03:
                        return "내부 공정성 양호"
                    elif r > 1.03:
                        return "상대적으로 높음 (고평가 가능성)"
                    else:
                        return "상대적으로 낮음 (저평가 가능성)"

                ext_label = label_external(ratio_ext)
                int_label = label_internal(ratio_int)

                st.markdown(
                    f"""
**요약 문장**  

- {year_sel}년 근속 **{tenure_sel}** 재직자의 평균 연봉은  
  산업 동일 근속구간 평균(참조연도: {ref_year}년)의 **{ratio_ext:.2f}배** ({ext_label})이며,  
  같은 직급 내부 평균의 **{ratio_int:.2f}배** ({int_label}) 입니다.
"""
                )

                summary_df = pd.DataFrame(
                    {
                        "지표": [
                            "사내 평균 연봉(만원)",
                            "산업 평균 연봉(만원)",
                            "산업 대비 배수",
                            "직급 내 평균 대비 배수",
                        ],
                        "값": [
                            round(mean_salary, 1),
                            round(mean_ind, 1) if not np.isnan(mean_ind) else np.nan,
                            round(ratio_ext, 2),
                            round(ratio_int, 2),
                        ],
                    }
                )
                st.table(summary_df)

                st.caption(
                    "※ 재직자(attrition=0) 기준, 동일 연도(pay_year)·근속구간(tenure_group) 내 구성원을 평균 기준으로 비교했습니다."
                )

# --------------------------------
# 14) 절대/상대 임금 × 이직확률 4분면 (보상 레버리지 분석)
# --------------------------------
st.markdown("---")
st.subheader("절대/상대 임금 × 이직확률 4분면 (어디에 돈을 써야 할까?)")

if "log_salary_vs_industry_resid" in raw.columns and raw["log_salary_vs_industry_resid"].notna().any():
    y_col = "log_salary_vs_industry_resid"
    y_label = "산업 대비 상대임금 잔차 (log)"
elif "log_salary_vs_industry" in raw.columns and raw["log_salary_vs_industry"].notna().any():
    y_col = "log_salary_vs_industry"
    y_label = "산업 대비 상대임금 (log)"
else:
    y_col = None

if (y_col is None) or raw["salary_std"].isna().all():
    st.info("salary_std와 산업 대비 상대임금 변수가 없어서 4분면 분석을 표시할 수 없습니다.")
else:
    base = raw[raw["attrition"] == 0].copy()
    if base.empty:
        st.info("재직자(attrition=0) 데이터가 없어 4분면 분석을 표시할 수 없습니다.")
    else:
        base = base.dropna(subset=["salary_std", y_col, "pred_prob"])

        if base.empty:
            st.info("salary_std, 상대임금, 예측 이직확률이 모두 있는 재직자 데이터가 없습니다.")
        else:
            x_med = base["salary_std"].median()
            y_med = base[y_col].median()

            st.caption(
                f"· X축: 절대 연봉 수준 (salary_std, 연도 내 z-score)\n"
                f"· Y축: {y_label}\n"
                "· 색상: 예측 이직확률(pred_prob, 붉을수록 위험), 점 크기: 근속(tenure)\n\n"
                "↘ 오른쪽 아래(절대 高 / 상대 低) 구간이 **이직 위험이 높은데 업계 대비 보상이 약한 그룹**으로, "
                "보상·승급 조정의 1순위 타깃이 될 수 있습니다."
            )

            hover_cols = ["tenure", "salary", "pay_year"]
            if "department" in base.columns:
                hover_cols.append("department")
            if "position" in base.columns:
                hover_cols.append("position")
            if "employeeID" in base.columns:
                hover_cols.append("employeeID")

            fig = px.scatter(
                base,
                x="salary_std",
                y=y_col,
                color="pred_prob",
                size="tenure",
                hover_data=hover_cols,
                color_continuous_scale="RdYlGn_r",
                labels={
                    "salary_std": "절대 연봉 수준 (연도 내 표준화)",
                    y_col: y_label,
                    "pred_prob": "예측 이직확률",
                    "tenure": "근속(년)",
                },
                title="절대/상대 임금 × 예측 이직확률 4분면",
            )

            fig.add_vline(x=x_med, line_dash="dash", line_color="gray")
            fig.add_hline(y=y_med, line_dash="dash", line_color="gray")

            fig.update_layout(
                height=550,
                legend_title_text="예측 이직확률",
            )

            st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# 15) 모델 심층 분석 (오즈비 · SHAP · 검증 + 최종 피처 해석 리포트)
# --------------------------------
st.markdown("---")
st.subheader("모델 심층 분석")

tab_coef, tab_shap, tab_valid = st.tabs(
    ["오즈비 · 계수표 · 최종 해석", "SHAP 분석", "검증(반복/교차)"]
)

with tab_coef:
    st.caption("로지스틱 회귀 계수 및 오즈비(OR)를 통해 변수 영향력 해석")
    if (sm is None) or (not two_class) or (Xtr is None) or (Ytr is None):
        st.warning(
            "statsmodels가 설치되어 있지 않거나, 라벨이 부족하여 계수표를 계산할 수 없습니다.\n"
            "설치: `pip install statsmodels`"
        )
    else:
        try:
            Xt = clf.named_steps["pre"].transform(Xtr)
            if hasattr(Xt, "toarray"):
                Xt = Xt.toarray()

            oh2 = clf.named_steps["pre"].named_transformers_["cat"]
            cat_names2 = list(oh2.get_feature_names_out(cat_cols))
            feat_names_tr2 = list(np.array(num_cols)) + cat_names2

            X_design = pd.DataFrame(Xt, index=Ytr.index, columns=feat_names_tr2)
            X_design = sm.add_constant(X_design, has_constant="add")

            X_design = X_design.loc[:, X_design.nunique() > 1]
            X_design = X_design.loc[:, ~X_design.T.duplicated()]

            logit_model = sm.Logit(Ytr, X_design)
            res = logit_model.fit(disp=0)

            params = res.params
            bse = res.bse
            pvals = res.pvalues
            ci = res.conf_int()

            out = pd.DataFrame(
                {
                    "feature": params.index,
                    "coef(β)": params.values,
                    "SE": bse.values,
                    "z": (params / bse).values,
                    "p": pvals.values,
                    "OR": np.exp(params.values),
                    "OR 95% LCL": np.exp(ci[0].values),
                    "OR 95% UCL": np.exp(ci[1].values),
                }
            )

            def sort_key(name: str):
                if name == "const":
                    return (0, name)
                return (1 if name in num_cols else 2, name)

            out["sort_order"] = out["feature"].map(sort_key)
            out = out.sort_values("sort_order").drop(columns="sort_order")

            st.dataframe(
                out.style.format(
                    {
                        "coef(β)": "{:.4f}",
                        "SE": "{:.4f}",
                        "z": "{:.2f}",
                        "p": "{:.4f}",
                        "OR": "{:.3f}",
                        "OR 95% LCL": "{:.3f}",
                        "OR 95% UCL": "{:.3f}",
                    }
                ),
                use_container_width=True,
            )

            st.caption(
                "- salary_std: 같은 연도 내에서 연봉이 1 표준편차 올라갈 때 퇴사위험(odds)의 변화량\n"
                "- log_salary_vs_industry_resid (또는 log_salary_vs_industry): 절대 연봉 수준을 통제한 후 남는 '추가적인 산업 대비 프리미엄'의 효과\n"
                "- log_salary_vs_internal: 같은 직급·연도 내 평균 대비 상대임금(배수)의 로그 효과"
            )

            # ---- 최종 피처 해석 리포트 (p<0.05 피처 위주) ----
            sig = out[(out["feature"] != "const") & (out["p"] < 0.05)].copy()
            sig = sig.reindex(
                sig["coef(β)"].abs().sort_values(ascending=False).index
            )

            def interpret_row(row):
                f = row["feature"]
                beta = row["coef(β)"]
                or_ = row["OR"]
                p = row["p"]
                direction = "이직 odds를 증가시키는 방향" if beta > 0 else "이직 odds를 감소시키는 방향"

                if f == "salary_std":
                    return (
                        f"**절대연봉 (salary_std)** — p={p:.3f}, OR={or_:.2f}\n"
                        f"  · 같은 연도 내에서 연봉이 1 표준편차 상승할 때, 이직 odds가 약 **{or_:.2f}배**로 변합니다 "
                        f"({direction})."
                    )
                if f == "tenure":
                    return (
                        f"**근속 (tenure)** — p={p:.3f}, OR={or_:.2f}\n"
                        f"  · 근속이 1년 늘어날 때 이직 odds가 **{or_:.2f}배**로 변합니다 "
                        f"({direction}). 보통 OR<1이면 장기근속자가 더 안정적인 계층이라는 의미입니다."
                    )
                if f in ["log_salary_vs_industry_resid", "log_salary_vs_industry"]:
                    return (
                        f"**산업 평균 대비 상대임금 ({f})** — p={p:.3f}, OR={or_:.2f}\n"
                        f"  · 산업 평균 대비 상대임금(로그)이 1 증가할 때(대략 2.7배 상승), 이직 odds가 **{or_:.2f}배**로 변합니다 "
                        f"({direction}). 업계 대비 얼마나 잘 받는지가 이직 의사에 미치는 영향으로 해석할 수 있습니다."
                    )
                if f == "log_salary_vs_internal":
                    return (
                        f"**내부 상대임금 (log_salary_vs_internal)** — p={p:.3f}, OR={or_:.2f}\n"
                        f"  · 같은 직급·연도 내에서 평균 대비 더 많이/적게 받는 정도가 1 로그단위 증가할 때, "
                        f"이직 odds가 **{or_:.2f}배**로 변합니다 ({direction}). 성과연봉제 도입 이후엔 "
                        f"내부 공정성과 이직 간 관계 해석에 핵심 지표가 될 수 있습니다."
                    )
                if f == "age":
                    return (
                        f"**나이 (age)** — p={p:.3f}, OR={or_:.2f}\n"
                        f"  · 나이가 1살 증가할 때 이직 odds가 **{or_:.2f}배**로 변합니다({direction}). "
                        f"보통 OR<1이면 연령이 높을수록 이직 가능성이 줄어드는 패턴입니다."
                    )

                for cat in cat_cols:
                    prefix = cat + "_"
                    if f.startswith(prefix):
                        level = f[len(prefix):]
                        return (
                            f"**{cat}={level}** — p={p:.3f}, OR={or_:.2f}\n"
                            f"  · 기준범주 대비 이 범주일 때 이직 odds가 **{or_:.2f}배**입니다 ({direction})."
                        )

                return (
                    f"**{f}** — p={p:.3f}, OR={or_:.2f}\n"
                    f"  · 값이 증가(또는 해당 범주)할 때 이직 odds가 **{or_:.2f}배**로 변합니다 ({direction})."
                )

            st.markdown("### 🔍 최종 피처 해석 리포트 (p<0.05 변수 기준 간이 보고서)")

            if sig.empty:
                st.write("p<0.05 수준에서 유의한 계수가 없습니다.")
            else:
                for _, row in sig.iterrows():
                    st.markdown("- " + interpret_row(row))

            with st.expander("해석 가이드 (공통 설명)"):
                st.markdown(
                    """
- OR(odds ratio)이 **1보다 크면**: 값이 증가/해당 범주일 때 이직 odds(위험도)가 증가  
- OR이 **1보다 작으면**: 값이 증가/해당 범주일 때 이직 odds가 감소  
- p-value는 계수의 통계적 유의성: 통상 0.05 이하를 “유의”로 많이 사용합니다.  

위 리포트는 모델을 다시 학습할 때마다 자동으로 갱신되며,  
연봉·근속·산업 대비 임금 수준 등 주요 계량 지표가 이직 odds에 어떤 방향과 크기로 작용하는지
간단한 HR 리뷰용 보고서로 바로 활용할 수 있습니다.
"""
                )

        except Exception as e:
            st.error(f"계수표 계산 중 오류: {e}")

with tab_shap:
    st.caption(
        "SHAP: 각 특징이 예측에 기여한 정도(절댓값 기준 변수 중요도).\n"
        "※ 막대그래프는 방향(퇴사위험 ↑/↓) 정보는 포함하지 않고, "
        "크기만 보여줍니다. 방향은 오즈비(OR) 표의 계수 부호를 기준으로 해석하세요."
    )
    if shap is None:
        st.warning("shap 라이브러리가 설치되어 있지 않습니다.\n\n설치: `pip install shap`")
    else:
        try:
            X_bg = clf.named_steps["pre"].transform(
                X.sample(min(200, len(X)), random_state=42)
            )
            model_lr = clf.named_steps["lr"]

            explainer = shap.LinearExplainer(
                model_lr,
                X_bg,
                feature_perturbation="interventional",
            )

            oh3 = clf.named_steps["pre"].named_transformers_["cat"]
            cat_names3 = list(oh3.get_feature_names_out(cat_cols))
            feat_names_tr3 = list(np.array(num_cols)) + cat_names3

            X_all_t = clf.named_steps["pre"].transform(X)
            sv_all = explainer.shap_values(X_all_t)
            if hasattr(sv_all, "toarray"):
                sv_all = sv_all.toarray()
            shap_df = pd.DataFrame(sv_all, columns=feat_names_tr3)

            global_imp = {}
            for c in num_cols:
                global_imp[c] = float(np.mean(np.abs(shap_df[c])))
            for cat in cat_cols:
                cols = [c for c in shap_df.columns if c.startswith(cat + "_")]
                if cols:
                    global_imp[cat] = float(
                        np.mean(np.abs(shap_df[cols]).sum(axis=1))
                    )
            imp_df = (
                pd.DataFrame(
                    {"feature": list(global_imp.keys()), "mean|SHAP|": list(global_imp.values())}
                )
                .sort_values("mean|SHAP|", ascending=False)
                .set_index("feature")
            )

            x_t = clf.named_steps["pre"].transform(x_in)
            sv_one = explainer.shap_values(x_t)
            if hasattr(sv_one, "toarray"):
                sv_one = sv_one.toarray()
            sv_one = sv_one[0]
            local_df = pd.DataFrame({"feature": feat_names_tr3, "shap": sv_one})

            agg_local = []
            for c in num_cols:
                agg_local.append(
                    {
                        "feature": c,
                        "shap": float(local_df.loc[local_df.feature == c, "shap"].sum()),
                    }
                )
            for cat in cat_cols:
                mask = local_df.feature.str.startswith(cat + "_")
                agg_local.append(
                    {
                        "feature": cat,
                        "shap": float(local_df.loc[mask, "shap"].sum()),
                    }
                )
            local_agg_df = (
                pd.DataFrame(agg_local)
                .set_index("feature")
                .sort_values("shap", key=lambda s: s.abs(), ascending=False)
            )

            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**전사 기준: 평균 |SHAP| (변수 중요도)**")
                st.bar_chart(imp_df)

            with g2:
                st.markdown("**현재 입력 기준: 특징별 기여도(부호 포함)**")
                st.bar_chart(local_agg_df)

            st.caption(
                "막대 길이가 길수록 해당 특징이 예측(이직/퇴사 위험)에 기여하는 정도가 큽니다.\n"
                "양수(+)는 이직/퇴사 위험을 높이고, 음수(-)는 위험을 낮추는 방향으로 작용합니다."
            )
        except Exception as e:
            st.error(f"SHAP 계산 중 오류: {e}")

with tab_valid:
    st.caption("검증 체크리스트 v2: 반복 split / k-fold 교차검증 / 보정 정도 점검")
    if not two_class:
        st.warning("이진 타깃이 부족해 검증을 수행할 수 없습니다.")
    else:
        try:
            st.markdown("### 1) 반복 train/test split 검증 (20회)")
            df_rep = repeated_split_validation(clf, X, Y, n_repeats=20, test_size=0.25)
            st.dataframe(df_rep.describe().T[["mean", "std"]])

            st.markdown(
                f"- AUC 평균: **{df_rep['auc'].mean():.3f}** (표준편차 {df_rep['auc'].std():.3f})  \n"
                f"- Precision 평균: **{df_rep['precision'].mean():.3f}**  \n"
                f"- Recall 평균: **{df_rep['recall'].mean():.3f}**  \n"
                f"- Brier score 평균: **{df_rep['brier'].mean():.3f}**"
            )

            st.markdown("### 2) Stratified K-fold AUC (k=5)")
            aucs = kfold_auc_validation(clf, X, Y, n_splits=5)
            st.write(
                f"AUC 평균: **{aucs.mean():.3f}**, 표준편차: **{aucs.std():.3f}**"
            )
            st.write("Fold별 AUC:", np.round(aucs, 3))

            st.caption(
                """
- AUC 평균이 0.70~0.80 수준이고 표준편차가 0.05 미만이면, 작은 조직(수십명) 기준에서 상당히 안정적인 모델로 볼 수 있습니다.  
- Brier score는 확률 보정 정도를 나타내며, 0.0에 가까울수록 좋습니다(0.25는 완전 랜덤 수준).
"""
            )
        except Exception as e:
            st.error(f"검증 계산 중 오류: {e}")

# --------------------------------
# 16) 개인/팀 리스크 자동 진단
# --------------------------------
st.markdown("---")
st.subheader("개인 / 팀 리스크 자동 진단")

tab_ind, tab_team = st.tabs(["개인 단위 진단 (현재 입력)", "팀/부서 단위 진단 (재직자 기준)"])


def diagnose_individual(prob, x_dict):
    """현재 What-if 입력과 예측값을 기반으로 위험 요인/보호 요인 텍스트 생성"""
    if prob >= 0.6:
        band = "High"
    elif prob >= 0.3:
        band = "Medium"
    else:
        band = "Low"

    risk_factors = []
    protect_factors = []

    # 절대연봉: 상위 z-score
    if x_dict.get("salary_std", 0) > 1.0:
        risk_factors.append("절대 연봉이 연도 내 상위 구간으로, 외부 오퍼 유인이 큰 핵심 인재군입니다.")
    elif x_dict.get("salary_std", 0) < -0.5:
        protect_factors.append("절대 연봉이 아직 높지 않아, 외부 시장에서 스카우트 타깃이 될 가능성은 상대적으로 낮습니다.")

    # 산업 대비 상대임금
    if "log_salary_vs_industry_resid" in x_dict:
        val = x_dict["log_salary_vs_industry_resid"]
        if val < -0.2:
            risk_factors.append("동종업계 평균 대비 보상 프리미엄이 낮은 편으로, 보상 불만에 따른 이직 유인이 존재합니다.")
        elif val > 0.2:
            protect_factors.append("동종업계 대비 상대적으로 좋은 보상을 받고 있어, 보상 측면의 이탈 유인은 낮은 편입니다.")
    elif "log_salary_vs_industry" in x_dict:
        val = x_dict["log_salary_vs_industry"]
        if val < 0:
            risk_factors.append("동종업계 평균 대비 보상이 낮은 편입니다.")
        elif val > 0.2:
            protect_factors.append("동종업계 평균보다 높은 수준의 보상을 받고 있습니다.")

    # 근속
    tenure_val = x_dict.get("tenure", 0)
    if tenure_val < 2:
        risk_factors.append("근속 2년 미만의 초기 재직 구간으로, 아직 조직에 완전히 정착하지 않은 상태입니다.")
    elif tenure_val >= 5:
        protect_factors.append("근속 5년 이상으로, 조직 정착도가 높은 안정층에 가깝습니다.")

    # 직무만족도
    if x_dict.get("jobSat3") == "낮음":
        risk_factors.append("직무만족도가 낮아, 보상 외 요인에서도 이탈 위험이 존재합니다.")
    elif x_dict.get("jobSat3") == "높음":
        protect_factors.append("직무만족도가 높아 현재 역할에 대한 몰입도가 높은 편입니다.")

    # 통근
    if x_dict.get("comm3") == "≥120m":
        risk_factors.append("통근 시간이 2시간 이상으로, 피로 누적으로 인한 이직 유인이 있을 수 있습니다.")

    return band, risk_factors, protect_factors


with tab_ind:
    st.markdown("### 개인 단위 자동 진단 (현재 What-if 입력 기준)")
    band, risk_factors, protect_factors = diagnose_individual(prob, x_dict)

    st.markdown(f"**예측 이직 확률:** {prob*100:.1f}%  /  **위험도 구간:** {band}")

    st.markdown("#### 주요 위험 요인")
    if not risk_factors:
        st.write("- 특별히 두드러지는 위험 요인이 감지되지는 않습니다.")
    else:
        for r in risk_factors:
            st.write(f"- {r}")

    st.markdown("#### 완충(보호) 요인")
    if not protect_factors:
        st.write("- 뚜렷한 보호 요인이 크지 않아, 조건 변화에 따라 이직 확률이 쉽게 변할 수 있는 상태입니다.")
    else:
        for r in protect_factors:
            st.write(f"- {r}")

    st.caption(
        "※ 이 섹션은 개별 구성원에 대해 '왜 위험한가 / 무엇이 보호요인인가'를 빠르게 요약해 보여주는 용도로 설계되었습니다. "
        "정책 결정 시에는 팀/직무/조직 단위 맥락과 함께 해석해야 합니다."
    )


with tab_team:
    st.markdown("### 팀 / 부서 단위 리스크 자동 진단 (재직자 기준)")

    if "department" not in raw.columns:
        st.info("ERP 데이터에 부서(부서/소속부서/팀) 컬럼이 없어 팀/부서 진단을 수행할 수 없습니다.")
    else:
        base = raw[raw["attrition"] == 0].copy()
        if base.empty:
            st.info("재직자(attrition=0) 데이터가 없어 팀/부서 진단을 수행할 수 없습니다.")
        else:
            # 집계용 보조 컬럼
            if "salary_vs_industry" in base.columns and base["salary_vs_industry"].notna().any():
                base["salary_vs_industry_fill"] = base["salary_vs_industry"]
            else:
                base["salary_vs_industry_fill"] = np.nan

            dept_summary = (
                base.groupby("department")
                .agg(
                    n=("pred_prob", "count"),
                    mean_prob=("pred_prob", "mean"),
                    high_cnt=("risk_band", lambda s: (s == "High").sum()),
                    med_cnt=("risk_band", lambda s: (s == "Medium").sum()),
                    low_cnt=("risk_band", lambda s: (s == "Low").sum()),
                    mean_tenure=("tenure", "mean"),
                    mean_salary_std=("salary_std", "mean"),
                    mean_industry_ratio=("salary_vs_industry_fill", "mean"),
                )
                .reset_index()
            )

            dept_summary["high_ratio"] = dept_summary["high_cnt"] / dept_summary["n"]
            dept_summary = dept_summary.sort_values("mean_prob", ascending=False)

            st.dataframe(
                dept_summary.style.format(
                    {
                        "mean_prob": "{:.3f}",
                        "high_ratio": "{:.2f}",
                        "mean_tenure": "{:.1f}",
                        "mean_salary_std": "{:.2f}",
                        "mean_industry_ratio": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )

            st.markdown("#### 상위 위험 부서 요약 (Top 3)")

            top_k = dept_summary.head(3)
            if top_k.empty:
                st.write("요약할 부서가 없습니다.")
            else:
                for _, row in top_k.iterrows():
                    dept = row["department"]
                    prob_mean = row["mean_prob"]
                    high_ratio = row["high_ratio"]
                    tenure_mean = row["mean_tenure"]
                    ind_ratio = row["mean_industry_ratio"]

                    txt = (
                        f"- **{dept}** — 평균 이직확률 **{prob_mean:.2f}**, "
                        f"High 구간 비중 **{high_ratio:.0%}**, "
                        f"평균 근속 **{tenure_mean:.1f}년**"
                    )
                    if not np.isnan(ind_ratio):
                        txt += f", 산업 평균 대비 연봉 **{ind_ratio:.2f}배** 수준"

                    st.markdown(txt)

            st.caption(
                "※ 이 표는 각 부서별 평균 이직확률과 High/Medium/Low 분포를 보여줍니다. "
                "평균 이직확률과 High 비중이 높은 부서를 우선순위로 두고, "
                "근속·보상 수준을 함께 고려하여 리텐션 정책을 설계할 수 있습니다."
            )
