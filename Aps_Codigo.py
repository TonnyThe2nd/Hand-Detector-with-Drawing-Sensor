import cv2
import mediapipe as mp
import math
import numpy

#IA que reconhece mão
mp_maos = mp.solutions.hands
#Desenho do mapa da mão
mp_desenho = mp.solutions.drawing_utils
#Estilo do desenho do mapa da mão
mp_estilo_desenho = mp.solutions.drawing_styles
#Configuração do detector
maos_detec_config = mp_maos.Hands(
    max_num_hands = 1, #Num maximo de mãos reconhecidas
    min_detection_confidence = 0.8, #Certeza de conhecimento da mão
    min_tracking_confidence = 0.7, #Confiança de tracking
    static_image_mode = False #config para tela dinamica
)
#variaveis e arrays
verificar_coordenadas = False
verificar_desenhando = False
pontos_anteriores_desenhos = None
todos_pontos = []
historico_pontos = []

cores = (
    (255,255,255), #branco
    (0,0,0), #preto
    (0,0,255), #vermelho
    (0,255,0) #verde
)
indice = 0
cor_pintura = cores[indice]
video = cv2.VideoCapture(0)
#Configurações do tamanho do frame e do fps
video.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FPS, 60)
#função para verificar dedos levantados
def dedo_levantado(mao, dedo_tipo, w, h):
     dedos = {
                'Indicador': {
                    'ponta': mp_maos.HandLandmark.INDEX_FINGER_TIP,
                    'junta': mp_maos.HandLandmark.INDEX_FINGER_PIP
                },
                'Meio': {
                    'ponta': mp_maos.HandLandmark.MIDDLE_FINGER_TIP,
                    'junta': mp_maos.HandLandmark.MIDDLE_FINGER_PIP
                },
                'Anelar': {
                    'ponta': mp_maos.HandLandmark.RING_FINGER_TIP,
                    'junta': mp_maos.HandLandmark.RING_FINGER_PIP
                },
                'Mindinho': {
                    'ponta': mp_maos.HandLandmark.PINKY_TIP,
                    'junta': mp_maos.HandLandmark.PINKY_PIP
                },
                'Dedão': {
                    'ponta': mp_maos.HandLandmark.THUMB_TIP,
                    'junta': mp_maos.HandLandmark.THUMB_IP  
                }
            }
     ponto_ponta = mao.landmark[dedos[dedo_tipo]['ponta']]
     ponto_junta = mao.landmark[dedos[dedo_tipo]['junta']]

     y_junta = ponto_junta.y*h
     y_ponta = ponto_ponta.y*h

     if dedo_tipo == 'Dedão':
         x_ponta = ponto_ponta.x*w
         x_junta = ponto_junta.x*w
         return x_ponta < x_junta
     else:
        levantado = y_junta > y_ponta
        return levantado
while True:
    ret, cam = video.read()
    #Verifica se a câmera foi encontrada
    if not ret: 
        print("Não foi possível abrir a câmera")
        break
    #Inverte a câmera para ficar na posição certa
    cam = cv2.flip(cam, 1)

    #mantem frame em rgb
    frame_rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    #desenha todos os pixels dentro do array
    for i in historico_pontos:
        for j in range(1, len(i)):
            cv2.line(cam, i[j-1], i[j],cor_pintura, 5)
    for i in range(1, len(todos_pontos)):
        cv2.line(cam, todos_pontos[i-1], todos_pontos[i],cor_pintura, 5)
                   
    #processa frame e identifica a mão
    detectar_mao = maos_detec_config.process(frame_rgb)
    dedos_index = {}
    #verifica se mão foi detectada
    if detectar_mao.multi_hand_landmarks:
        #para cada traço detectado na mão
        for mao in detectar_mao.multi_hand_landmarks:
            #desenha a conexão entre os dedos e o centro da mão
            mp_desenho.draw_landmarks(cam, mao, mp_maos.HAND_CONNECTIONS)
            h, w, c = cam.shape
            dedos = {
                'Indicador': {
                    'ponta': mp_maos.HandLandmark.INDEX_FINGER_TIP
                },
                'Meio': {
                    'ponta': mp_maos.HandLandmark.MIDDLE_FINGER_TIP
                },
                'Anelar': {
                    'ponta': mp_maos.HandLandmark.RING_FINGER_TIP
                },
                'Mindinho': {
                    'ponta': mp_maos.HandLandmark.PINKY_TIP
                },
                'Dedão': {
                    'ponta': mp_maos.HandLandmark.THUMB_TIP
                }
            }
            dedos_estado = {}
            #itera sobre o dicionario dedos
            for nome_dedo, index in dedos.items():
                #mapeia os index para cada dedo detectado
                ponto = mao.landmark[index['ponta']]
                #valores (x,y) da posição de cada dedo
                x = int(ponto.x * w)
                y = int(ponto.y * h)
                levantado =dedo_levantado(mao, nome_dedo, w, h)
                estado = 'Levantado' if levantado else 'Dobrado'
                dedos_estado[nome_dedo] = levantado
                #informações de cada dedo
                dedos_index[nome_dedo] = {
                    'X' : x,
                    'Y' : y,
                    'Dedo' : nome_dedo,
                    'Estado' :  estado
                }
                #alteração visual para cada dedo na tela
                cor = cores[3] if levantado else cores[2]
                cv2.circle(cam, (x,y), 10, cor, -1)
                if verificar_coordenadas:
                    cv2.putText(cam, f'X: {x}', (x-30,y-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[0], 2)
                    cv2.putText(cam, f'Y: {y}', (x-30,y-55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[0], 2)
                    cv2.putText(cam, f'{estado}', (x-30,y-75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[0], 2)
            #infos dos dedos
            
            #Verifica posição atual do dedo indicador
            pontos_atuais = (int(mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].x*w),int(mao.landmark[mp_maos.HandLandmark.INDEX_FINGER_TIP].y*h))
            if verificar_desenhando and not dedos_estado['Meio']:
                if pontos_anteriores_desenhos is not None:
                    # pequeno filtro de movimento para reduzir jitter
                    dx = pontos_atuais[0] - pontos_anteriores_desenhos[0]
                    dy = pontos_atuais[1] - pontos_anteriores_desenhos[1]
                    dist = math.hypot(dx, dy)
                    if dist > 3:  # ajuste esse valor conforme necessidade
                        todos_pontos.append(pontos_atuais)
                        cv2.line(cam, pontos_anteriores_desenhos, pontos_atuais, cor_pintura, 5)
                else:
                    # primeiro ponto do stroke atual (não desenha linha, só registra)
                    todos_pontos.append(pontos_atuais)
                pontos_anteriores_desenhos = pontos_atuais
            else:
                # quando o dedo do meio é levantado (parou de desenhar),
                if verificar_desenhando and pontos_anteriores_desenhos is not None and len(todos_pontos) > 0:
                    historico_pontos.append(todos_pontos.copy())
                    todos_pontos.clear()
                pontos_anteriores_desenhos = None
    cv2.rectangle(cam, (0, 150), (280, 0), cores[1], -1)
    cv2.rectangle(cam, (0, 150), (280, 0), cores[0], 2)
    pos_y = 25
    for i, (nome_dedo, info) in enumerate(dedos_index.items()):
        texto = f"{nome_dedo}: ({info['X']},{info['Y']}) / {info['Estado']}"
        cv2.putText(cam, texto, (10, pos_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
        pos_y += 25
    #Caixa de tags de comando
    cv2.rectangle(cam, (1000, 170), (1280, 0), cores[1], -1)
    cv2.rectangle(cam, (1000, 170), (1280, 0), cores[0], 2)
    cv2.putText(cam, "COMANDOS", (1020, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cores[0], 2)
    cv2.putText(cam, "Q-Fechar  D-Desenhar", (1020, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
    cv2.putText(cam, "X-Apagar M - Mudar Cor", (1020, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
    cv2.putText(cam, "PARA DESENHAR:", (1020, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cores[2], 1)
    cv2.putText(cam, "1. Aperte 'D' para ativar", (1020, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
    cv2.putText(cam, "2. Dedo meio ABAIXADO = Desenha", (1020, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
    cv2.putText(cam, "3. Dedo meio LEVANTADO = Para", (1020, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
    cv2.putText(cam, "4. Aperte 'D' novamente", (1020, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
    cv2.putText(cam, "para desativar modo", (1020, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cores[0], 1)
    cv2.imshow('Camera', cam)

    #verifica se clicks no teclados foram dados
    key = cv2.waitKey(1) & 0xFF
    if key ==  ord('q'):
        print("Saindo da câmera")
        break
    elif key == ord('c'):
        verificar_coordenadas = not verificar_coordenadas
    elif key == ord('d'):
        verificar_desenhando = not verificar_desenhando
        if not verificar_desenhando:
            if todos_pontos:
                historico_pontos.append(todos_pontos)
                todos_pontos = []
            pontos_anteriores_desenhos= None
    elif key == ord('x'):
        historico_pontos = []
        todos_pontos = []
        pontos_anteriores_desenhos = None
    elif key == ord('m'):
        indice += 1
        if indice >= len(cores):
            indice = 0
        cor_pintura = cores[indice] 
video.release()
cv2.destroyAllWindows()

