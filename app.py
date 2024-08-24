import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import paho.mqtt.client as mqtt
from flask import Flask, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import pickle
from model_builder import *
from trainer import find_TPR_threshold
import requests

# Flask 애플리케이션 생성
app = Flask(__name__)
CORS(app)

# Flask 트리거 API 경로 설정 (페달 서버 API)
START_PEDAL_URL = "http://localhost:6000/start-pedal-model"
STOP_PEDAL_URL = "http://localhost:6000/stop-pedal-model"

# Swagger 설정
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={'app_name': "OBD Anomaly Detection API"}
)

# Swagger UI 경로 추가
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# 저장된 전처리 객체 및 모델 불러오기
with open('isolation_forest_model.pkl', 'rb') as model_file:
    ensemble = pickle.load(model_file)  # IsolationTreeEnsemble 모델 불러오기
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)  # Scaler 불러오기

# 이상 탐지 상태를 추적하는 플래그
anomaly_detected = False

# MQTT 클라이언트 콜백 함수 설정
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe("obd/topic")  # OBD 데이터를 받는 토픽을 구독

def on_message(client, userdata, msg):
    global anomaly_detected
    try:
        # MQTT로 수신한 데이터를 JSON 형식으로 변환하여 처리
        data = eval(msg.payload.decode('utf-8'))
        print(f"Received data: {data}")

        # 필요한 열만 선택하여 리스트로 변환 (MQTT로 받은 실시간 데이터)
        data_list = [[data['FuelStatus'], data['PedalPos'], data['ThrottlePos'], data['Load'], data['Speed'], data['RPM']]]

        # 데이터 전처리 (Scaler 적용)
        data_scaled = scaler.transform(data_list)

        # 모델을 통해 이상 점수 계산 및 임계값 설정
        anomaly_scores = ensemble.anomaly_score(data_scaled)
        threshold = 0.5  # 설정된 기준값 (적절히 조정 가능)
        anomaly = 1 if ensemble.predict_from_anomaly_scores(anomaly_scores, threshold)[0] == 1 else 0

        # 결과 출력
        print(f"Anomaly: {anomaly}, Data: {data_list}")

        # 이상치가 감지되면 페달 모델 시작 트리거 발생
        if anomaly == 1 and not anomaly_detected:
            anomaly_detected = True
            print("Anomaly detected! Sending start-pedal-model request...")
            requests.post(START_PEDAL_URL)

        # 이상치가 해소되면 페달 모델 종료 트리거 발생
        elif anomaly == 0 and anomaly_detected:
            anomaly_detected = False
            print("No anomaly detected. Sending stop-pedal-model request...")
            requests.post(STOP_PEDAL_URL)

    except Exception as e:
        print(f"Error processing message: {e}")

# Flask 엔드포인트 - 페달 모델 시작
@app.route('/start-pedal-model')
def start_pedal_model():
    return jsonify({"message": "Pedal model started!"})

# Flask 엔드포인트 - 페달 모델 종료
@app.route('/stop-pedal-model')
def stop_pedal_model():
    return jsonify({"message": "Pedal model stopped!"})

# 기본 경로 설정 ("/")
@app.route('/')
def index():
    return "OBD 모델 API 서버가 실행 중입니다."

# Flask 서버 시작
if __name__ == '__main__':
    # MQTT 클라이언트 생성 및 콜백 함수 연결
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # MQTT 브로커에 연결 (브로커 주소와 포트를 적절히 설정)
    client.connect("3.35.30.20", 1883, 60)

    # Flask 서버를 비동기적으로 실행 (MQTT는 백그라운드에서 계속 실행됨)
    client.loop_start()
    app.run(debug=True, use_reloader=False, port=5001)
