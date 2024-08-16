import paho.mqtt.client as mqtt
import json
import time
import random

# MQTT 브로커 정보
broker_address = "3.35.30.20"  # Flask 서버와 동일한 브로커 주소
port = 1883
topic = "test/topic"  # 동일한 토픽으로 메시지 전송

# MQTT 클라이언트 생성
client = mqtt.Client()

# MQTT 브로커에 연결
client.connect(broker_address, port=port)

# 테스트 데이터를 MQTT로 전송하는 함수
def send_test_data():
    # 무작위로 OBD 데이터 생성
    rpm = random.randint(500, 7000)  # RPM은 500에서 7000 사이
    speed = random.randint(0, 180)   # 속도는 0에서 180 km/h 사이
    throttle = random.uniform(0, 100)  # 스로틀은 0%에서 100% 사이
    time_info = time.strftime("%Y-%m-%d %H:%M:%S")  # 현재 시간
    
    # 테스트용 OBD 데이터
    obd_data = {
        'rpm': rpm,                  # 엔진 RPM
        'speed': speed,              # 속도
        'throttle': throttle,        # 스로틀 포지션
        'time': time_info            # 시간 정보
    }

    # 데이터를 JSON 형식으로 변환
    message = json.dumps(obd_data)

    # MQTT 메시지 전송
    client.publish(topic, message)
    print(f"Sent data: {message}")

# 주기적으로 데이터를 전송 (5초 간격)
if __name__ == "__main__":
    try:
        while True:
            send_test_data()
            time.sleep(5)
    except KeyboardInterrupt:
        print("Publisher stopped.")
