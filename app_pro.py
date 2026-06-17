import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from lightgbm import LGBMRegressor

st.set_page_config(layout="wide", page_title="따릉이 통합 관제 플랫폼")

@st.cache_data
def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "final_table_B_fixed.csv")

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['bike_count_x'] = df['gender_F'] + df['gender_M']
    return df

df = load_data()
df['log_gender_M'] = np.log1p(df['gender_M'])
df['log_gender_F'] = np.log1p(df['gender_F'])

# 1. 학습 로직
features = ['hour', 'is_weekend', 'temp', 'is_holiday']

model_total = LGBMRegressor(n_estimators=1000, random_state=42).fit(df[features], df['bike_count_x'])
model_foreign = LGBMRegressor(n_estimators=1000, random_state=42).fit(df[features], df['foreigner_bike_count'])
model_M = LGBMRegressor(n_estimators=1000, learning_rate=0.01).fit(df[features], df['log_gender_M'])
model_F = LGBMRegressor(n_estimators=1000, learning_rate=0.01).fit(df[features], df['log_gender_F'])

# 2. UI 설정
st.title("📊 따릉이 통합 관제 플랫폼")
zone_1_2 = st.columns([1, 2])

with zone_1_2[0]:
    with st.container(border=True):
        st.subheader("⚙️ 관제 설정")
        selected_date = st.date_input("기준 날짜 선택", df['date'].min())
        selected_hour = st.slider("현재 시간", 0, 23, 12)
        target_date = pd.to_datetime(selected_date)
        is_weekend_val = 1 if target_date.dayofweek >= 5 else 0
        is_holiday_val = 1 if target_date in df[df['is_holiday']==1]['date'].values else 0

# 3. 예측 로직
future_hours = [(selected_hour + i) % 24 for i in range(24)]
results = pd.DataFrame({'hour': future_hours, 'Time': [f"{h}시" for h in future_hours]})
results['is_weekend'] = is_weekend_val
results['is_holiday'] = is_holiday_val

def get_precise_temp(target_date, target_hour, df):
    subset = df[(df['date'].dt.month == target_date.month) & (df['date'].dt.day == target_date.day) & (df['hour'] == target_hour)]
    return subset['temp'].mean() if not subset.empty else df['temp'].mean()

results['temp'] = [get_precise_temp(target_date, h, df) for h in future_hours]

results['bike_count_x'] = model_total.predict(results[features])
results['foreigner_bike_count'] = model_foreign.predict(results[features])
results['gender_M'] = np.expm1(model_M.predict(results[features]))
results['gender_F'] = np.expm1(model_F.predict(results[features]))

# 4. 시각화 출력 (IndentationError 해결)
with zone_1_2[1]:
    with st.container(border=True):
        st.subheader("🔮 현재 시점 관제 지표")
        current_pred = results[results['hour'] == selected_hour].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{selected_hour}시 예상 수요", f"{int(current_pred['bike_count_x'])} 대")
        c2.metric("평균 기온", f"{results['temp'].mean():.1f} °C")
        c3.metric("기준 날짜", str(selected_date))

    with st.container(border=True):
        st.subheader("📈 성별 수요 예측")
        fig_gender = px.line(results, x='Time', y=['gender_M', 'gender_F'], title="남녀 따릉이 수요 예측")
        st.plotly_chart(fig_gender, use_container_width=True)

    with st.container(border=True):
        st.subheader("📈 외국인 수요 예측")
        fig_foreign = px.line(results, x='Time', y=['foreigner_bike_count'], title="외국인 따릉이 수요 예측")
        st.plotly_chart(fig_foreign, use_container_width=True)
