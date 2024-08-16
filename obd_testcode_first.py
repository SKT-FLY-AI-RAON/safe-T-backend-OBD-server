import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib


# 저장된 전처리 객체 불러오기
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')

model = joblib.load('isolation_forest_model.pkl')  # 여기에서 model을 불러옴


columns = ['ENGINE_RPM', 'SPEED', 'ENGINE_COOLANT_TEMP']

# 새로운 데이터 불러오기
df_test = pd.read_csv('kaggle_modified_with_anomalies.csv')

# 필요한 열만 선택
# columns = ['RPM', 'Speed', 'Coolant', 'Load', 'ThrottlePos', 'PedalPos', 'FuelStatus', 'FuelPressure', 'AirFlow']
df_test_selected = df_test[columns]

# 숫자형 데이터만 남기기 위해 비숫자형 데이터가 포함된 행 제거
df_test_selected = df_test_selected.apply(pd.to_numeric, errors='coerce')

# NaN이 포함된 행 제거 (비숫자형 데이터를 가진 행이 제거됨)
df_test_selected = df_test_selected.dropna()


# 동일한 방식으로 전처리 적용
new_data_imputed = imputer.transform(df_test_selected)
new_data_scaled = scaler.transform(new_data_imputed)

# 전처리된 데이터를 DataFrame으로 변환
test_processed = pd.DataFrame(new_data_scaled, columns=df_test_selected.columns)


# 모델에 데이터 적용 및 이상치 예측
test_processed['anomaly'] = model.predict(test_processed)

# 이상치 탐지 결과: -1은 이상치, 1은 정상 데이터
test_processed['anomaly'] = test_processed['anomaly'].map({1: 0, -1: 1})

# 이상치 데이터만 선택
anomalies = test_processed[test_processed['anomaly'] == 1]

# 결과 확인
anomalies.head()





# # 저장된 모델 불러오기 및 예측
# model = joblib.load('isolation_forest_model.pkl')
# predictions = model.predict(new_data_processed)

# print(f"새로운 데이터에 대한 예측 결과: {predictions}")
