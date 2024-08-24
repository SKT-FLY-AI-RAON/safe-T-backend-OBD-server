import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from data_loader import load_data, preprocess_data  # 필요한 경우 사용
from model_builder import IsolationTreeEnsemble
from trainer import find_TPR_threshold

# Set seaborn style
sns.set(style="whitegrid")

def plot_anomaly_results(df_test, y_test, y_pred, features):
    plt.figure(figsize=(14, 12))

    for i, feature in enumerate(features):
        plt.subplot(len(features), 1, i + 1)
        sns.lineplot(x=df_test.index, y=df_test[feature], label=f'{feature} (Data)', color='royalblue', linewidth=1.5)

        # Highlight actual anomalies as shaded regions
        for j in range(len(y_test)):
            if y_test[j] == 1:
                plt.axvspan(j, j+1, color='orange', alpha=0.2)  # Use lighter shading for anomaly regions

        # Plot predicted anomalies
        anomalies = np.where(y_pred == 1)[0]
        plt.scatter(anomalies, df_test[feature].iloc[anomalies], color='red', label='Predicted Anomaly', marker='o', s=50, zorder=3)

        plt.title(f'{feature} with Anomalies', fontsize=14, weight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True)

    plt.tight_layout(pad=1.5)
    plt.suptitle('Anomaly Detection Results', fontsize=18, weight='bold', y=1.02)
    sns.despine()  # Remove top and right spines for a cleaner look
    plt.show()

def main():
    # 데이터 로드 및 전처리
    train_data_path = 'C:\\Users\\SKT033\\PycharmProjects\\Isolation_Forest\\data\\sorted_combined_file_no_duplicates.csv'  # 학습용 정상 데이터
    df_train = pd.read_csv(train_data_path)
    target_columns = ['FuelStatus', 'PedalPos', 'ThrottlePos', 'Load', 'Speed', 'RPM']
    X_train = df_train[target_columns].values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # IsolationTreeEnsemble 모델 학습
    sample_size = 256
    ensemble = IsolationTreeEnsemble(sample_size=sample_size, n_trees=100)
    ensemble.fit(X_train_scaled)

    # 모델과 스케일러 저장
    with open('isolation_forest_model.pkl', 'wb') as model_file:
        pickle.dump(ensemble, model_file)
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

    # 테스트 데이터 로드 및 전처리
    test_data_path = 'C:\\Users\\SKT033\\PycharmProjects\\Isolation_Forest\\data\\scenario_2_confused_labeled_with_fuel.csv'  # 테스트 데이터
    df_test = pd.read_csv(test_data_path)
    X_test = df_test[target_columns].values
    y_test = df_test['Label'].values
    X_test_scaled = scaler.transform(X_test)

    # 이상 점수 계산 및 임계값 설정
    anomaly_scores = ensemble.anomaly_score(X_test_scaled)
    desired_TPR = 0.95
    threshold, FPR = find_TPR_threshold(y_test, anomaly_scores, desired_TPR)

    # 새로운 임계값으로 예측
    y_pred = ensemble.predict_from_anomaly_scores(anomaly_scores, threshold)

    # 성능 평가
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    # 이상 탐지된 위치 시각화
    plot_anomaly_results(df_test, y_test, y_pred, target_columns)

if __name__ == "__main__":
    main()
