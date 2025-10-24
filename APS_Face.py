import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
from deepface import DeepFace
import numpy as np

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FPS, 120)
face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

face_config = face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7
)
mostra_pontos_faciais = False
REGIOES = {
    "Olho Direito": [33, 133, 160, 159, 158, 144, 153, 154, 155, 246],
    "Olho Esquerdo": [362, 263, 387, 386, 385, 373, 380, 381, 382, 466],
    "Sobrancelha Direita": [70, 63, 105, 66, 107],
    "Sobrancelha Esquerda": [336, 296, 334, 293, 300],
    "Boca Externa": [61, 291, 78, 308, 191, 80, 81, 82, 13, 312],
    "Boca Interna": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
    "Nariz": [6, 197, 195, 4, 2, 45],
    "Mandíbula": list(range(0, 17))
}

def traduz_emocao(emocao):
    match emocao:
        case 'sad':
            return 'triste'
        case 'happy':
            return 'feliz'
        case 'disgust':
            return 'enojado'
        case 'fear':
            return 'medo'
        case 'surprise':
            return 'surpreso'
        case 'angry': 
            return 'raiva'
        case 'neutral':
            return 'neutro'

while True:
    ret, cam = video.read()
    if not ret:
        print("Erro ao abrir a câmera")
        break

    h, w, _ = cam.shape
    panel_width = 300
    cv2.rectangle(cam, (w - panel_width, 0), (w, h), (30, 30, 30), -1)

    img_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    detect = face_config.process(img_rgb)
    
    if detect.multi_face_landmarks:
        for face in detect.multi_face_landmarks:
            #captura range da face detectada
            x = [coordenada.x*w for coordenada in face.landmark]
            y = [coordenada.y*h for coordenada in face.landmark]

            x_min = int(min(x))-10
            x_max = int(max(x))+10
            y_min = int(min(y))-10
            y_max = int(max(y))+10

            frame_para_analisar = cam[y_min:y_max, x_min:x_max]

            #desenha retangulo ao redor da face detecada
            cv2.rectangle(cam, (x_min,y_min), (x_max,y_max), (255,0,0),2)

            #analisa expressão da face detectada
            frame_para_analise = cv2.resize(frame_para_analisar, (224,224))
            frame_para_analise = cv2.cvtColor(frame_para_analise, cv2.COLOR_BGR2GRAY)
            frame_para_analise = cv2.equalizeHist(frame_para_analise)
            frame_para_analise = cv2.cvtColor(frame_para_analise, cv2.COLOR_GRAY2BGR)
            analise_expressao = DeepFace.analyze(frame_para_analisar, actions=['emotion'],enforce_detection=False, detector_backend='mediapipe')
            resultado_expressao = analise_expressao[0]['dominant_emotion']
            cv2.putText(cam, f'Emocao: {traduz_emocao(resultado_expressao)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)
            if mostra_pontos_faciais:
                mp_draw.draw_landmarks(
                image=cam,
                landmark_list=face,
                connections=face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_style.get_default_face_mesh_tesselation_style()
                )

            y_offset = 30
            for regiao, indices in REGIOES.items():
                lm = face.landmark[indices[0]]
                x = int(lm.x * w)
                y = int(lm.y * h)
                z = round(lm.z, 3)

                texto = f"{regiao}: ({x}, {y}, {z})"
                cv2.putText(
                    cam,
                    texto,
                    (w - panel_width + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA
                )
                y_offset += 25

    cv2.imshow("Câmera", cam)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        mostra_pontos_faciais = not mostra_pontos_faciais
video.release()
cv2.destroyAllWindows()