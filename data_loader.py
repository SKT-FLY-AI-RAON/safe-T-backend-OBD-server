import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(file_path, target_columns):
    """
    데이터 파일 경로와 대상 컬럼을 받아 데이터를 로드하고,
    해당 컬럼만 선택하여 반환합니다.
    """
    df = pd.read_csv(file_path)
    return df[target_columns].values

def scale_data(X_train):
    """
    학습 데이터를 받아 StandardScaler로 스케일링을 수행하고,
    스케일러 객체와 스케일링된 데이터를 반환합니다.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return scaler, X_train_scaled

def preprocess_data(df, scaler):
    """
    테스트 데이터를 받아 학습 데이터에 맞춰 스케일링을 수행하고 반환합니다.
    """
    return scaler.transform(df)
