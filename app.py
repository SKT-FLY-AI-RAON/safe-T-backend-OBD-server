
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.ensemble import IsolationForest
# import joblib
# import paho.mqtt.client as mqtt
# import requests
# from flask import Flask, jsonify
# from flask_cors import CORS
# from flask_swagger_ui import get_swaggerui_blueprint

# # Flask 애플리케이션 생성
# app = Flask(__name__)
# CORS(app)

# # Flask 트리거 API 경로 설정
# START_PEDAL_URL = "http://localhost:5000/start-pedal-model"
# STOP_PEDAL_URL = "http://localhost:5000/stop-pedal-model"

# # Swagger 설정
# SWAGGER_URL = '/swagger'  # Swagger UI에 접근할 URL
# API_URL = '/static/swagger.json'  # Swagger 문서의 경로

# swaggerui_blueprint = get_swaggerui_blueprint(
#     SWAGGER_URL,
#     API_URL,
#     config={  # Swagger UI 설정
#         'app_name': "OBD Anomaly Detection API"
#     }
# )

# # Swagger UI 경로 추가
# app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# # 저장된 전처리 객체 및 모델 불러오기
# imputer = joblib.load('imputer.pkl')
# scaler = joblib.load('scaler.pkl')
# model = joblib.load('isolation_forest_model.pkl')

# # 이상 탐지 상태를 추적하는 플래그
# anomaly_detected = False

# # MQTT 클라이언트 콜백 함수 설정
# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code " + str(rc))
#     client.subscribe("obd/data")

# def on_message(client, userdata, msg):
#     global anomaly_detected
#     # MQTT로 들어온 데이터를 JSON 형식으로 가정
#     data = eval(msg.payload.decode('utf-8'))

#     # 필요한 열만 선택
#     data_list = [[data['ENGINE_RPM'], data['SPEED'], data['ENGINE_COOLANT_TEMP']]]

#     # 데이터 전처리 (Imputer와 Scaler 적용)
#     data_imputed = imputer.transform(data_list)
#     data_scaled = scaler.transform(data_imputed)

#     # 모델을 통해 이상치 탐지
#     prediction = model.predict(data_scaled)
#     anomaly = 1 if prediction[0] == -1 else 0  # 1: 이상치, 0: 정상

#     # 결과 출력
#     print(f"Anomaly: {anomaly}, Data: {data_list}")

#     # 이상치가 감지되면 페달 모델 시작 트리거 발생
#     if anomaly == 1 and not anomaly_detected:
#         anomaly_detected = True
#         print("Anomaly detected! Starting pedal model...")
#         requests.post(START_PEDAL_URL)

#     # 이상치가 해소되면 페달 모델 종료 트리거 발생
#     elif anomaly == 0 and anomaly_detected:
#         anomaly_detected = False
#         print("No anomaly detected. Stopping pedal model...")
#         requests.post(STOP_PEDAL_URL)

# # Flask 엔드포인트 - 페달 모델 시작
# @app.route('/start-pedal-model')
# def start_pedal_model():
#     """
#     트리거: 페달 모델 시작
#     ---
#     responses:
#       200:
#         description: 페달 모델이 시작되었습니다.
#     """
#     return jsonify({"message": "Pedal model started!"})

# # Flask 엔드포인트 - 페달 모델 종료
# @app.route('/stop-pedal-model')
# def stop_pedal_model():
#     """
#     트리거: 페달 모델 종료
#     ---
#     responses:
#       200:
#         description: 페달 모델이 중지되었습니다.
#     """
#     return jsonify({"message": "Pedal model stopped!"})

# # Flask 서버 시작
# if __name__ == '__main__':
#     # MQTT 클라이언트 생성 및 콜백 함수 연결
#     client = mqtt.Client()
#     client.on_connect = on_connect
#     client.on_message = on_message

#     # MQTT 브로커에 연결 (브로커 주소와 포트를 적절히 설정하세요)
#     client.connect("3.35.30.20", 1883, 60)

#     # Flask 서버를 비동기적으로 실행 (MQTT는 백그라운드에서 계속 실행됨)
#     client.loop_start()
#     app.run(debug=True, use_reloader=False)


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib
import paho.mqtt.client as mqtt
import requests
from flask import Flask, jsonify
from flask_cors import CORS
from flask_swagger_ui import get_swaggerui_blueprint
import ast

# Flask 애플리케이션 생성
app = Flask(__name__)
CORS(app)

# Flask 트리거 API 경로 설정 (페달 서버 API)
START_PEDAL_URL = "http://localhost:5001/start-pedal-model"
STOP_PEDAL_URL = "http://localhost:5001/stop-pedal-model"

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
imputer = joblib.load('imputer.pkl')
scaler = joblib.load('scaler.pkl')
model = joblib.load('isolation_forest_model.pkl')

# 이상 탐지 상태를 추적하는 플래그
anomaly_detected = False

# MQTT 클라이언트 콜백 함수 설정
def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT broker with result code {rc}")
    client.subscribe("test/topic")  # OBD 데이터를 받는 토픽을 구독

def on_message(client, userdata, msg):
    global anomaly_detected
    try:
        # MQTT로 수신한 데이터를 JSON 형식으로 변환하여 처리
        data = eval(msg.payload.decode('utf-8'))
        # data = ast.literal_eval(msg.payload.decode('utf-8'))
        print(data)

        # 필요한 열만 선택하여 리스트로 변환
        # data_list = [[data['ENGINE_RPM'], data['SPEED'], data['ENGINE_COOLANT_TEMP']]]
        data_list = [[data['rpm'], data['speed'], data['throttle']]]


        # 데이터 전처리 (SimpleImputer와 MinMaxScaler 적용)
        data_imputed = imputer.transform(data_list)
        data_scaled = scaler.transform(data_imputed)

        # 모델을 통해 이상 탐지
        prediction = model.predict(data_scaled)
        anomaly = 1 if prediction[0] == -1 else 0  # 1: 이상치, 0: 정상

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
    """
    트리거: 페달 모델 시작
    ---
    responses:
      200:
        description: 페달 모델이 시작되었습니다.
    """
    return jsonify({"message": "Pedal model started!"})

# Flask 엔드포인트 - 페달 모델 종료
@app.route('/stop-pedal-model')
def stop_pedal_model():
    """
    트리거: 페달 모델 종료
    ---
    responses:
      200:
        description: 페달 모델이 중지되었습니다.
    """
    return jsonify({"message": "Pedal model stopped!"})

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
    app.run(debug=True, use_reloader=False)
