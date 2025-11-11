import cv2
import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
import json
import time
import os


#algumas configs padrões
MODEL_PATH = 'modelo.h5'
CAMERA_ID = 0
FRAME_WIDTH = 720
FRAME_HEIGHT = 480
FPS = 30
PREDICT_INTERVAL_SEC = 0.18  #5 predições/segundo
MIN_CONFIDENCE = 0.6
IMG_SIZE = (128, 128)

#cores
CORES = [
    (255, 255, 255),  # branco
    (0, 0, 0),        # preto
    (0, 0, 255),      # vermelho
    (0, 255, 0),      # verde
    (255, 0, 0),      # azul
    (255, 255, 0),    # amarelo
    (255, 0, 255),    # magenta
    (0, 255, 255),    # ciano
    (255, 165, 0),    # laranja
    (128, 0, 128),    # roxo
    (128, 128, 128),  # cinza
    (139, 69, 19)     # marrom
]

#carregando modelo
try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado:", MODEL_PATH)
    print("input_shape esperado:", modelo.input_shape)
except Exception as e:
    print("Erro ao carregar modelo:", e)
    modelo = None

#classes
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
CLASS_NAMES = CLASS_NAMES[:21]
print(f"Classes para predição: {CLASS_NAMES}")

#config do mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7,
    static_image_mode=False
)
#variaveis
verificar_coordenadas = False
verificar_desenhando = False
pontos_anteriores_desenhos = None
todos_pontos = []
historico_pontos = []
indice_cor = 0
cor_pintura = CORES[indice_cor]
estado_desenho = 'Parado'
last_predict_time = 0.0
landmark = False


def dedo_levantado(mao, dedo_tipo, w, h):
    dedos = {
        'Indicador': {'ponta': mp_hands.HandLandmark.INDEX_FINGER_TIP, 'junta': mp_hands.HandLandmark.INDEX_FINGER_PIP},
        'Meio': {'ponta': mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 'junta': mp_hands.HandLandmark.MIDDLE_FINGER_PIP},
        'Anelar': {'ponta': mp_hands.HandLandmark.RING_FINGER_TIP, 'junta': mp_hands.HandLandmark.RING_FINGER_PIP},
        'Mindinho': {'ponta': mp_hands.HandLandmark.PINKY_TIP, 'junta': mp_hands.HandLandmark.PINKY_PIP},
        'Dedão': {'ponta': mp_hands.HandLandmark.THUMB_TIP, 'junta': mp_hands.HandLandmark.THUMB_IP}
    }
    try:
        ponta = mao.landmark[dedos[dedo_tipo]['ponta']]
        junta = mao.landmark[dedos[dedo_tipo]['junta']]
    except Exception:
        return False

    y_junta = junta.y * h
    y_ponta = ponta.y * h

    if dedo_tipo == 'Dedão':
        x_ponta = ponta.x * w
        x_junta = junta.x * w
        return x_ponta < x_junta
    else:
        return y_ponta < y_junta


def pad(img):
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    padded = cv2.copyMakeBorder(img, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


def preprocessamento(frame_roi):
    if frame_roi is None or frame_roi.size == 0:
        return None
    square = pad(frame_roi)
    square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(square_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)
    roi_normalized = roi_resized.astype('float32') / 255.0

    #ajustar canais
    expected_shape = getattr(modelo, 'input_shape', None)
    if expected_shape is not None:
        channels = expected_shape[-1]
        if channels == 1:
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_RGB2GRAY)
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            return np.expand_dims(roi_gray, axis=0)
    return np.expand_dims(roi_normalized, axis=0)


def predict(frame_roi, debug_save=False):
    global modelo, CLASS_NAMES
    if modelo is None:
        return None, None
    x = preprocessamento(frame_roi)
    if x is None:
        return None, None
    try:
        preds = modelo.predict(x, verbose=0)[0]
        topk = min(3, len(preds))
        inds = preds.argsort()[-topk:][::-1]
        resultados = []
        for i in inds:
            conf = float(preds[i])
            name = CLASS_NAMES[i] if 0 <= i < len(CLASS_NAMES) else f'Classe_{i}'
            resultados.append((name, conf, int(i)))
        if debug_save and (len(resultados) == 0 or resultados[0][1] < MIN_CONFIDENCE):
            os.makedirs('debug_rois', exist_ok=True)
            ts = int(time.time() * 1000)
            cv2.imwrite(f'debug_rois/roi_{ts}.png', frame_roi)
        return resultados, preds
    except Exception as e:
        print("Erro na predição:", e)
        return None, None

def main():
    global verificar_coordenadas, verificar_desenhando, pontos_anteriores_desenhos
    global todos_pontos, historico_pontos, cor_pintura, indice_cor, estado_desenho
    global last_predict_time

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print("Não foi possível abrir a câmera")
        return

    print("Iniciando captura. Pressione 'q' para sair.")

    while True:
        ret, cam = cap.read()
        if not ret:
            print("Frame não lido da câmera.")
            break

        cam = cv2.flip(cam, 1)
        h, w, _ = cam.shape
        frame_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

        #desenha histórico de strokes
        for stroke in historico_pontos:
            for i in range(1, len(stroke)):
                cv2.line(cam, stroke[i - 1], stroke[i], cor_pintura, 5)
        for i in range(1, len(todos_pontos)):
            cv2.line(cam, todos_pontos[i - 1], todos_pontos[i], cor_pintura, 5)

        results = hands_detector.process(frame_rgb)
        dedos_index = {}
        dedos_estado = {}

        estado_desenho = 'Desenhando' if verificar_desenhando else 'Parado'

        if results.multi_hand_landmarks:
            for mao in results.multi_hand_landmarks:
                if landmark:
                    mp_drawing.draw_landmarks(cam, mao, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing_styles.get_default_hand_landmarks_style(),
                                          mp_drawing_styles.get_default_hand_connections_style())
                lm_array = [(int(lm.x * w), int(lm.y * h)) for lm in mao.landmark]

                xs = [p[0] for p in lm_array]
                ys = [p[1] for p in lm_array]
                xmin, xmax = max(0, min(xs) - 15), min(w, max(xs) + 15)
                ymin, ymax = max(0, min(ys) - 15), min(h, max(ys) + 15)

                #roi expandida
                scale = 1.4
                box_w = xmax - xmin
                box_h = ymax - ymin
                cx = xmin + box_w // 2
                cy = ymin + box_h // 2
                size_box = int(max(box_w, box_h) * scale)
                xmin_e = max(0, cx - size_box // 2)
                ymin_e = max(0, cy - size_box // 2)
                xmax_e = min(w, cx + size_box // 2)
                ymax_e = min(h, cy + size_box // 2)

                frame_analise = cam[ymin_e:ymax_e, xmin_e:xmax_e].copy()
                if frame_analise is None or frame_analise.size == 0:
                    continue

                now = time.time()
                if now - last_predict_time >= PREDICT_INTERVAL_SEC:
                    resultados, preds_raw = predict(frame_analise, debug_save=True)
                    last_predict_time = now
                    if resultados:
                        top1_name, top1_conf, _ = resultados[0]
                        cv2.putText(cam, f"{top1_name}: {top1_conf:.2f}",
                                    (xmin_e, max(20, ymin_e - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        for i, (nm, conf, _) in enumerate(resultados):
                            cv2.putText(cam, f"{i+1}. {nm}:{conf:.2f}",
                                        (xmin_e, ymax_e + 20 + i * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        print("Top preds:", resultados)

                cv2.rectangle(cam, (xmin_e, ymin_e), (xmax_e, ymax_e), CORES[0], 1)

                #detecta estado dos dedos
                dedos = {
                    'Indicador': {'ponta': mp_hands.HandLandmark.INDEX_FINGER_TIP},
                    'Meio': {'ponta': mp_hands.HandLandmark.MIDDLE_FINGER_TIP},
                    'Anelar': {'ponta': mp_hands.HandLandmark.RING_FINGER_TIP},
                    'Mindinho': {'ponta': mp_hands.HandLandmark.PINKY_TIP},
                    'Dedão': {'ponta': mp_hands.HandLandmark.THUMB_TIP}
                }

                for nome_dedo, info in dedos.items():
                    ponto = mao.landmark[info['ponta']]
                    x = int(ponto.x * w)
                    y = int(ponto.y * h)
                    levantado = dedo_levantado(mao, nome_dedo, w, h)
                    estado = 'Levantado' if levantado else 'Dobrado'
                    dedos_estado[nome_dedo] = levantado
                    dedos_index[nome_dedo] = {'X': x, 'Y': y, 'Estado': estado}
                    if verificar_coordenadas:
                        cv2.putText(cam, f'{estado}', (x - 30, y - 75),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORES[0], 1)

                pontos_atuais = (int(mao.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                                 int(mao.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))

                if verificar_desenhando and (not dedos_estado.get('Meio', True)):
                    if pontos_anteriores_desenhos is not None:
                        dx = pontos_atuais[0] - pontos_anteriores_desenhos[0]
                        dy = pontos_atuais[1] - pontos_anteriores_desenhos[1]
                        dist = math.hypot(dx, dy)
                        if dist > 3:
                            todos_pontos.append(pontos_atuais)
                            cv2.line(cam, pontos_anteriores_desenhos, pontos_atuais, cor_pintura, 5)
                    else:
                        todos_pontos.append(pontos_atuais)
                    pontos_anteriores_desenhos = pontos_atuais
                else:
                    if pontos_anteriores_desenhos is not None and len(todos_pontos) > 0:
                        historico_pontos.append(todos_pontos.copy())
                        todos_pontos = []
                    pontos_anteriores_desenhos = None

        #caixa lateral de comandos
        cv2.rectangle(cam, (1000, 0), (1280, 210), CORES[1], -1)
        cv2.rectangle(cam, (1000, 0), (1280, 210), CORES[0], 2)
        cv2.putText(cam, "COMANDOS", (1020, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CORES[0], 2)
        cv2.putText(cam, "Q - Fechar   D - Desenhar", (1020, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES[0], 1)
        cv2.putText(cam, "X - Limpar    M - Mudar Cor", (1020, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES[0], 1)
        cv2.putText(cam, "C - Coordenadas", (1020, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES[0], 1)
        cor_texto_aviso = CORES[3] if verificar_desenhando else CORES[2]
        cv2.putText(cam, estado_desenho, (1020, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_texto_aviso, 2)

        cv2.imshow('Camera', cam)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Saindo...")
            break
        elif key == ord('c'):
            verificar_coordenadas = not verificar_coordenadas
        elif key == ord('d'):
            verificar_desenhando = not verificar_desenhando
            estado_desenho = 'Desenhando' if verificar_desenhando else 'Parado'
            if not verificar_desenhando and todos_pontos:
                historico_pontos.append(todos_pontos.copy())
                todos_pontos = []
            pontos_anteriores_desenhos = None
        elif key == ord('x'):
            historico_pontos = []
            todos_pontos = []
            pontos_anteriores_desenhos = None
        elif key == ord('m'):
            indice_cor = (indice_cor + 1) % len(CORES)
            cor_pintura = CORES[indice_cor]
            print("Cor alterada:", cor_pintura)
        elif key == ord('l'):
            landmark = not landmark

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
