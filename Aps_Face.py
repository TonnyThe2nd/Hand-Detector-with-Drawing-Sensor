from deepface import DeepFace
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from tkinter import Tk, simpledialog
import tkinter as tk
import concurrent.futures 
import os

#inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

#variáveis de controle
mostrar_landmarks = False
emotion_history = deque(maxlen=10)
emocao_atual = "Identificando"
confianca = 0
PATH_IMAGENS_ROSTO = "./rostos_salvos"
nome_pessoa = ''

#configurar câmera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 680)
cap.set(cv2.CAP_PROP_FPS, 60)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def preprocessamento(face_roi):
    try:
        #converter BGR para RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        #redimensionar para o tamanho esperado pelo DeepFace
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        return face_resized
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return None

def analize_emocao(face_img):
    try:
        #usar análise direta do deepface
        result = DeepFace.analyze(
            face_img,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True,
            align=True
        )
        #emoção dominante
        emotion = result[0]['dominant_emotion']
        confianca = result[0]['emotion'][emotion]
        
        return emotion, confianca
        
    except Exception as e:
        print(f"Erro na análise: {e}")
        return "erro", 0
    
def reconhecer_pessoa(face_img):
    try:
        if not os.path.exists(PATH_IMAGENS_ROSTO):
            os.makedirs(PATH_IMAGENS_ROSTO)
            return "Nenhum rosto cadastrado"
        
        imagens = os.listdir(PATH_IMAGENS_ROSTO)
        if not imagens:
            return "Nenhum rosto cadastrado"
        
        for imagem in imagens:
            if imagem.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    caminho_imagem = os.path.join(PATH_IMAGENS_ROSTO, imagem)
                    
                    result = DeepFace.verify(
                        face_img, 
                        caminho_imagem,
                        model_name='VGG-Face',
                        detector_backend='opencv',
                        enforce_detection=False,
                        distance_metric='cosine'
                    )
                    
                    if result['verified']:
                        nome = os.path.splitext(imagem)[0]
                        return nome
                        
                except Exception as e:
                    print(f"Erro ao verificar {imagem}: {e}")
                    continue
        
        return "Desconhecido"
        
    except Exception as e:
        print(f"Erro no reconhecimento: {e}")
        return "Erro no reconhecimento"
# def tratamento_imagem(path):
#     for imagens in os.listdir(path):
#         imagem = cv2.imread(os.path.join(path,imagens))
#         imagem_tratada = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
#         cv2.imwrite(f"{path}/{imagens}", imagem_tratada)
 
def salvar_rosto(frame):
    #criar uma janela temporária com input de nome
    root = tk.Tk()
    root.withdraw()
    
    #pedir o nome 
    nome = simpledialog.askstring("Salvar Rosto", "Digite o nome para salvar o rosto:")
    
    if nome:
        #remover espaços e caracteres especiais do nome do arquivo
        nome_arquivo = "".join(c for c in nome if c.isalnum() or c in (' ', '-', '_')).rstrip()
        nome_arquivo = nome_arquivo.replace(' ', '_')
        
        caminho_arquivo = f"{PATH_IMAGENS_ROSTO}/{nome_arquivo}.jpg"
        #garante imagem rgb salva para reconhecimento posterior
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow("imagem teste", frame) # Comentado para evitar janela extra
        cv2.imwrite(caminho_arquivo, frame_rgb)
        # tratamento_imagem(PATH_IMAGENS_ROSTO) # Removido, pois é redundante
        print(f"Rosto salvo como: {caminho_arquivo}")    
    
    root.destroy()

emotion_analysis_future = None
recognition_future = None

frame_count = 0
face_roi_global = None
processed_face_global = None
pessoa_reconhecida_global = "Nenhum rosto"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar frame")
        break
    
    #espelhar o frame
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    #converter BGR para RGB para mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #processando resultados da detecção de face
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            #extrair coordenadas dos landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
            
            #calcular bounding box
            x_coords = [p[0] for p in landmarks]
            y_coords = [p[1] for p in landmarks]
            
            x_min = max(0, min(x_coords) - 30)
            y_min = max(0, min(y_coords) - 30)
            x_max = min(w, max(x_coords) + 30)
            y_max = min(h, max(y_coords) + 30)
            
            #desenhar retângulo ao redor do rosto
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            #desenhar nome da pessoa em cima do rosto
            texto_nome = f"Nome: {pessoa_reconhecida_global}"
            (text_width, text_height), baseline = cv2.getTextSize(texto_nome, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            #calcular posição do texto (acima do retângulo)
            texto_x = x_min
            texto_y = y_min - 10  
            
            #garantir que o texto não saia da tela
            if texto_y < text_height + 5:
                texto_y = y_max + text_height + 5
            
            
            #texto do nome
            cv2.putText(frame, texto_nome, (texto_x, texto_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            #desenhar landmarks se mostrar for true
            if mostrar_landmarks:
                for landmark in landmarks:
                    cv2.circle(frame, landmark, 1, (0, 0, 255), -1)
            
            #extrair ROI do rosto para análise
            face_roi = frame[y_min:y_max, x_min:x_max]
            face_roi_global = face_roi
            
            if face_roi.size > 0:
                try:
                    #pré-processar a imagem do rosto
                    processed_face = preprocessamento(face_roi)
                    processed_face_global = processed_face
                    
                    if processed_face is not None:

                        if frame_count % 10 == 0 and (emotion_analysis_future is None or emotion_analysis_future.done()):
                            emotion_analysis_future = executor.submit(analize_emocao, processed_face)

                        if frame_count % 30 == 0 and (recognition_future is None or recognition_future.done()):
                            recognition_future = executor.submit(reconhecer_pessoa, processed_face)

                        if emotion_analysis_future and emotion_analysis_future.done():
                            emotion, conf = emotion_analysis_future.result()
                            if emotion != "erro" and conf > 20:
                                emotion_history.append(emotion)
                                emocao_atual = emotion
                                confianca = conf
                            else:
                                emocao_atual = "Baixa Confiança"
                                confianca = 0
                            emotion_analysis_future = None 

                        if recognition_future and recognition_future.done():
                            pessoa_reconhecida_global = recognition_future.result()
                            recognition_future = None 
                    else:
                        emocao_atual = "Erro Processamento"
                        confianca = 0
                        
                except Exception as e:
                    print(f"Erro ao submeter/obter resultados da análise: {e}")
                    emocao_atual = "Erro Análise"
                    confianca = 0
    
    else:
        emocao_atual = "Nenhum Rosto"
        confianca = 0
        pessoa_reconhecida_global = "Nenhum rosto"
    
    #exibir informações na tela
    text_color = (0, 255, 0) if confianca > 50 else (0, 165, 255) if confianca > 20 else (0, 0, 255)
    
    #texto de emoção principal
    cv2.putText(frame, f"Emocao: {emocao_atual}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    #texto de confiança
    cv2.putText(frame, f"Confianca: {confianca:.1f}%", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    #instruções
    cv2.putText(frame, "M: Landmarks  S: Salvar  Q: Sair", (10, h - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    #barra de confiança dinâmica
    if confianca > 0:
        bar_width = 200
        bar_height = 10
        confianca_width = int((confianca / 100) * bar_width)
        cv2.rectangle(frame, (10, 70), (10 + bar_width, 70 + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 70), (10 + confianca_width, 70 + bar_height), text_color, -1)
    
    cv2.imshow('Detecao de Emocoes', frame)
    
    frame_count += 1
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        mostrar_landmarks = not mostrar_landmarks
    elif key == ord('s'):
        if face_roi_global is not None and face_roi_global.size > 0:
            salvar_rosto(face_roi_global)
        else:
            print("Nenhum rosto detectado para salvar!")

executor.shutdown(wait=True) 
cap.release()
cv2.destroyAllWindows()
