import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os

# === Configurações ===
MODEL_PATH = 'modelo.h5'
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
PREDICT_INTERVAL_SEC = 0.18  # tempo entre predições
MIN_CONFIDENCE = 0.6
IMG_SIZE = (128, 128)  # tamanho usado no treino

# === Cores ===
CORES = [
    (255, 255, 255), (0, 0, 0), (0, 0, 255), (0, 255, 0),
    (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (255, 165, 0), (128, 0, 128), (128, 128, 128), (139, 69, 19)
]

# === Carrega modelo ===
modelo = tf.keras.models.load_model(MODEL_PATH)
print("Input shape do modelo:", modelo.input_shape)
print("Output shape do modelo:", modelo.output_shape)

# === Descobre classes a partir da pasta de treino ===
TRAIN_PATH = 'train'
CLASS_NAMES = sorted(os.listdir(TRAIN_PATH))  # ordena alfabeticamente
print("Classes detectadas:", CLASS_NAMES)

# === Inicializa MediaPipe Hands ===
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    static_image_mode=False
)

# === Controle de predição ===
last_predict_time = 0.0

def predict_roi(frame_roi):
    """Recebe ROI BGR e retorna (classe_nome, confidence)"""
    try:
        roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
        roi_normalized = roi_resized.astype('float32') / 255.0
        x = np.expand_dims(roi_normalized, axis=0)
        preds = modelo.predict(x, verbose=0)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        if conf < MIN_CONFIDENCE:
            return None, conf
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f'Classe_{idx}'
        return name, conf
    except Exception as e:
        print("Erro na predição:", e)
        return None, 0.0

# === Função principal ===
def main():
    global last_predict_time

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Não foi possível abrir a câmera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)

        # Predição
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                xs = [int(lm.x * w) for lm in hand_landmarks.landmark]
                ys = [int(lm.y * h) for lm in hand_landmarks.landmark]
                xmin, xmax = max(0, min(xs)-15), min(w, max(xs)+15)
                ymin, ymax = max(0, min(ys)-15), min(h, max(ys)+15)
                roi = frame[ymin:ymax, xmin:xmax]

                now = time.time()
                if roi.size > 0 and now - last_predict_time >= PREDICT_INTERVAL_SEC:
                    letra, conf = predict_roi(roi)
                    last_predict_time = now
                    if letra:
                        cv2.putText(frame, f"{letra}: {conf:.2f}", (xmin, max(20, ymin-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        cv2.imshow("Reconhecimento de Sinais", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
