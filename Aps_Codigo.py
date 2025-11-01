# realtime_hand_app_fixed.py
"""
Versão corrigida e com melhorias:
- Salvamento correto de strokes ao parar de desenhar.
- Leitura robusta de class_indices.json.
- Pré-processamento consistente do ROI (BGR->RGB, pad->square, resize, normalize).
- Proteções contra ROI inválida / índices fora do range.
- Mensagens de debug úteis.
"""
import cv2
import mediapipe as mp
import math
import numpy as np
import tensorflow as tf
import json
import time
import os

# ------- Configurações -------
MODEL_PATH = 'modelo.h5'
CLASS_INDICES_PATH = 'class_indices.json'
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
PREDICT_INTERVAL_SEC = 0.18  # 5-6 predições por segundo
MIN_CONFIDENCE = 0.6
IMG_SIZE = (128, 128)

# Cores (BGR)
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

# ------- Carrega modelo e nomes de classes -------
try:
    modelo = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado:", MODEL_PATH)
except Exception as e:
    print("Erro ao carregar modelo:", e)
    modelo = None

# Carrega class indices de maneira robusta
if os.path.exists(CLASS_INDICES_PATH):
    try:
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        # class_indices pode ser {"W": 0, "X": 1, ...} ou similar
        # vamos construir uma lista onde index -> nome da classe
        if isinstance(class_indices, dict):
            max_idx = max(class_indices.values())
            CLASS_NAMES = [None] * (max_idx + 1)
            for name, idx in class_indices.items():
                if 0 <= idx <= max_idx:
                    CLASS_NAMES[idx] = name
            # substitui None por nome generico (se houver)
            for i in range(len(CLASS_NAMES)):
                if CLASS_NAMES[i] is None:
                    CLASS_NAMES[i] = f'Classe_{i}'
        else:
            CLASS_NAMES = [str(x) for x in class_indices]
        print("Nomes de classes carregados do JSON.")
    except Exception as e:
        print("Erro ao ler class_indices.json:", e)
        CLASS_NAMES = [f'Classe_{i}' for i in range(21)]
        print("Usando nomes genéricos.")
else:
    CLASS_NAMES = [f'Classe_{i}' for i in range(21)]
    print("Arquivo class_indices.json não encontrado. Usando nomes genéricos (ajuste se necessário).")

# ------- MediaPipe Hands config -------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7,
    static_image_mode=False
)

# ------- Estado do app -------
verificar_coordenadas = False
verificar_desenhando = False
pontos_anteriores_desenhos = None
todos_pontos = []
historico_pontos = []
indice_cor = 0
cor_pintura = CORES[indice_cor]
estado_desenho = 'Parado'

last_predict_time = 0.0

# --- Funções utilitárias ---
def dedo_levantado(mao, dedo_tipo, w, h):
    """Retorna True se dedo estiver levantado (heurística simples)."""
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
        # leva em conta espelhamento (flip horizontal)
        return x_ponta < x_junta
    else:
        return y_ponta < y_junta

def pad_to_square(img):
    """Recebe imagem (H,W,3) e retorna imagem quadrada com padding preto centralizada."""
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    padded = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return padded

def preprocess_roi_for_model(frame_roi):
    """Recebe ROI BGR e devolve array pronto para model.predict (1, H, W, 3)."""
    # Convert BGR -> RGB
    roi_rgb = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
    # Pad to square to preserve aspect ratio
    roi_square = pad_to_square(roi_rgb)
    # Resize
    roi_resized = cv2.resize(roi_square, IMG_SIZE, interpolation=cv2.INTER_AREA)
    # Normalize
    roi_normalized = roi_resized.astype('float32') / 255.0
    # Batch dim
    x = np.expand_dims(roi_normalized, axis=0)
    return x

def predict_roi(frame_roi):
    """Recebe ROI em BGR e retorna (classe_nome, confidence) ou (None, conf)."""
    global modelo, CLASS_NAMES
    if modelo is None:
        return None, 0.0
    try:
        x = preprocess_roi_for_model(frame_roi)
        preds = modelo.predict(x, verbose=0)
        idx = int(np.argmax(preds))
        conf = float(np.max(preds))
        if conf < MIN_CONFIDENCE:
            return None, conf
        name = CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else f'Classe_{idx}'
        return name, conf
    except Exception as e:
        print("Erro na predição:", e)
        return None, 0.0

# ------- Main loop -------
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

        # desenha histórico de strokes e o stroke atual
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
                lm_array = []
                for lm in mao.landmark:
                    lm_array.append((int(lm.x * w), int(lm.y * h)))

                xs = [p[0] for p in lm_array]
                ys = [p[1] for p in lm_array]
                xmin, xmax = max(0, min(xs) - 15), min(w, max(xs) + 15)
                ymin, ymax = max(0, min(ys) - 15), min(h, max(ys) + 15)

                # Proteção ROI
                if xmax - xmin <= 10 or ymax - ymin <= 10:
                    # ROI muito pequena -> pular
                    continue

                frame_analise = cam[ymin:ymax, xmin:xmax].copy()
                if frame_analise.size == 0:
                    continue

                # Predição em intervalos definidos
                now = time.time()
                if now - last_predict_time >= PREDICT_INTERVAL_SEC:
                    nome, conf = predict_roi(frame_analise)
                    last_predict_time = now
                    if nome is not None:
                        cv2.putText(cam, f"{nome}: {conf:.2f}", (xmin, max(20, ymin - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        # para debugging, opcional:
                        # cv2.putText(cam, f"Sem conf >= {MIN_CONFIDENCE:.2f}", (xmin, max(20, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        pass

                # Desenha retângulo de ROI (opcional)
                cv2.rectangle(cam, (xmin, ymin), (xmax, ymax), CORES[0], 1)

                # verifica dedos
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
                    dedos_index[nome_dedo] = {'X': x, 'Y': y, 'Dedo': nome_dedo, 'Estado': estado}
                    cor = CORES[3] if levantado else CORES[2]
                    if verificar_coordenadas:
                        cv2.putText(cam, f'X:{x}', (x - 30, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORES[0], 1)
                        cv2.putText(cam, f'Y:{y}', (x - 30, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORES[0], 1)
                        cv2.putText(cam, f'{estado}', (x - 30, y - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORES[0], 1)

                # Lógica de desenho usando dedo indicador como ponta e dedo médio como gatilho
                pontos_atuais = (int(mao.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w),
                                 int(mao.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h))

                # Se estamos em modo desenhar E dedo medio está ABAIXADO (não levantado)
                if verificar_desenhando and (not dedos_estado.get('Meio', True)):
                    # dedo medio ABAIXADO => desenhar
                    if pontos_anteriores_desenhos is not None:
                        dx = pontos_atuais[0] - pontos_anteriores_desenhos[0]
                        dy = pontos_atuais[1] - pontos_anteriores_desenhos[1]
                        dist = math.hypot(dx, dy)
                        if dist > 3:  # filtro para reduzir jitter
                            todos_pontos.append(pontos_atuais)
                            cv2.line(cam, pontos_anteriores_desenhos, pontos_atuais, cor_pintura, 5)
                    else:
                        todos_pontos.append(pontos_atuais)
                    pontos_anteriores_desenhos = pontos_atuais
                else:
                    # parou de desenhar: salva stroke (nota: removi checagem que dependia de verificar_desenhando)
                    if pontos_anteriores_desenhos is not None and len(todos_pontos) > 0:
                        historico_pontos.append(todos_pontos.copy())
                        todos_pontos = []
                    pontos_anteriores_desenhos = None

        # Desenha painel de info dedos
        cv2.rectangle(cam, (0, 0), (280, 150), CORES[1], -1)
        cv2.rectangle(cam, (0, 0), (280, 150), CORES[0], 2)
        pos_y = 25
        for i, (nome_dedo, info) in enumerate(dedos_index.items()):
            texto = f"{nome_dedo}: ({info['X']},{info['Y']}) {info['Estado']}"
            cv2.putText(cam, texto, (10, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CORES[0], 1)
            pos_y += 25

        # Caixa de comandos
        cv2.rectangle(cam, (1000, 0), (1280, 210), CORES[1], -1)
        cv2.rectangle(cam, (1000, 0), (1280, 210), CORES[0], 2)
        cv2.putText(cam, "COMANDOS", (1020, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CORES[0], 2)
        cv2.putText(cam, "Q - Fechar   D - Desenhar", (1020, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES[0], 1)
        cv2.putText(cam, "X - Limpar    M - Mudar Cor", (1020, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES[0], 1)
        cv2.putText(cam, "C - Coordenadas", (1020, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CORES[0], 1)
        cv2.putText(cam, "Para desenhar: ative 'D' e mantenha dedo medio ABAIXADO", (1020, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.37, CORES[2], 1)
        cor_texto_aviso = CORES[3] if verificar_desenhando else CORES[2]
        cv2.putText(cam, estado_desenho, (1020, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_texto_aviso, 2)

        cv2.imshow('Camera', cam)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Saindo...")
            break
        elif key == ord('c'):
            verificar_coordenadas = not verificar_coordenadas
            print("Coordenadas:", verificar_coordenadas)
        elif key == ord('d'):
            verificar_desenhando = not verificar_desenhando
            estado_desenho = 'Desenhando' if verificar_desenhando else 'Parado'
            # ao desativar, salva stroke atual
            if not verificar_desenhando:
                if todos_pontos:
                    historico_pontos.append(todos_pontos.copy())
                    todos_pontos = []
                pontos_anteriores_desenhos = None
            print("Desenhar:", verificar_desenhando)
        elif key == ord('x'):
            historico_pontos = []
            todos_pontos = []
            pontos_anteriores_desenhos = None
            print("Tela limpa")
        elif key == ord('m'):
            indice_cor = (indice_cor + 1) % len(CORES)
            cor_pintura = CORES[indice_cor]
            print("Cor alterada:", cor_pintura)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
