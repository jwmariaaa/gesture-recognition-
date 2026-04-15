import cv2
import mediapipe as mp
import csv
import os
import math
import numpy as np
import pygame

# Inicializando o Pygame para o uso do som
pygame.mixer.init()

# =========================================================================
# 1. FUNÇÕES DO ALGORITMO DE RECONHECIMENTO (Mantidas)
# =========================================================================
def normalizar_pontos(pontos_xy):
    base_x, base_y = pontos_xy[0], pontos_xy[1]
    pontos_transladados = []
    
    for i in range(0, len(pontos_xy), 2):
        pontos_transladados.append(pontos_xy[i] - base_x)
        pontos_transladados.append(pontos_xy[i+1] - base_y)
        
    distancias = [math.sqrt(pontos_transladados[i]**2 + pontos_transladados[i+1]**2) for i in range(0, len(pontos_transladados), 2)]
    max_dist = max(distancias) if max(distancias) > 0 else 1.0
    return [p / max_dist for p in pontos_transladados]

def carregar_dataset(csv_file):
    dataset = []
    if not os.path.exists(csv_file):
        return dataset
    with open(csv_file, mode='r') as f:
        reader = csv.reader(f)
        next(reader, None) # Pula o cabeçalho
        for row in reader:
            if len(row) >= 43:
                dataset.append({
                    "label": row[0].strip().upper(),
                    "pontos": normalizar_pontos([float(v) for v in row[1:43]])
                })
    return dataset

def reconhecer_gesto(pontos_atuais, dataset):
    if not dataset: return None
    
    pontos_norm = normalizar_pontos(pontos_atuais)
    melhor_label = None
    menor_distancia = float('inf')
    
    for item in dataset:
        distancia = sum((p_atual - p_data)**2 for p_atual, p_data in zip(pontos_norm, item["pontos"]))
        if distancia < menor_distancia:
            menor_distancia = distancia
            melhor_label = item["label"]
            
    if menor_distancia < 1.2:  # Threshold mais tolerante para gestos complexos como o Fox
        return melhor_label
    return None

# =========================================================================
# 2. FUNÇÕES DE EFEITOS VISUAIS, IMAGEM E REDE (Novas)
# =========================================================================
def tocar_som(som_obj):
    """ Toca o arquivo de som (evitando travar o frame) e garante que o som exista """
    if som_obj is not None:
        som_obj.play()

def mostrar_imagem(frame, imagem_overlay, largura_desejada=150):
    """ Cola uma imagem transparente por cima do vídeo, no canto inferior direito. """
    if imagem_overlay is None:
        return frame
    
    alt_original, larg_original = imagem_overlay.shape[:2]
    alt_redimensionada = int(alt_original * (largura_desejada / larg_original))
    img_redim = cv2.resize(imagem_overlay, (largura_desejada, alt_redimensionada))
    
    h_frame, w_frame = frame.shape[:2]
    y_ini = h_frame - alt_redimensionada - 10
    y_fim = h_frame - 10
    x_ini = w_frame - largura_desejada - 10
    x_fim = w_frame - 10
    
    # Faz uma mistura inteligente do Canal Alpha (transparência)
    if img_redim.shape[2] == 4:
        alpha_img = img_redim[:, :, 3] / 255.0
        alpha_fundo = 1.0 - alpha_img
        for c in range(3):
            frame[y_ini:y_fim, x_ini:x_fim, c] = (alpha_img * img_redim[:, :, c] + alpha_fundo * frame[y_ini:y_fim, x_ini:x_fim, c])
    else:
        frame[y_ini:y_fim, x_ini:x_fim] = img_redim
    return frame

def aplicar_efeito(frame, hand_landmarks, cor):
    """ Cria um efeito de glow (aura brilhante) ao redor da mão. """
    h, w, _ = frame.shape
    aura_layer = np.zeros_like(frame, dtype=np.uint8)
    
    # Desenha bolas grandes e pintadas na aura_layer onde tem mão
    for ponto in hand_landmarks.landmark:
        px, py = int(ponto.x * w), int(ponto.y * h)
        cv2.circle(aura_layer, (px, py), 25, cor, -1)
    
    # Borra intensamente essas bolas para parecerem pura luz ou fumaça colorida
    aura_layer = cv2.GaussianBlur(aura_layer, (61, 61), 0)
    
    # Mescla = Adiciona o layer de luz acima da câmera
    cv2.addWeighted(aura_layer, 0.6, frame, 1.0, 0, frame)

def aplicar_tremor(frame, intensidade=15):
    """ Cria um efeito de terremoto / tremor na câmera """
    h, w = frame.shape[:2]
    dx = np.random.randint(-intensidade, intensidade)
    dy = np.random.randint(-intensidade, intensidade)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # borderMode copia as bordas para não ficar o fundo preto quando a imagem treme
    return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

# =========================================================================
# 3. CONFIGURAÇÕES DOS ASSETS (Evita carregar dentro do loop e travar)
# =========================================================================
ASSETS_CONFIG = {
    "TIGER": {
        "imgpath": "naruto_hand.png", "sompath": "naruto_sound.mp3", 
        "cor": (255, 0, 0) # Azul (BGR OpenCV)
    },
    "DIVINE DOGS": {
        "imgpath": "megumi_hand.png", "sompath": "megumi_sound.mp3", 
        "cor": (0, 0, 0) # Preto
    },
    "FOX": {
        "imgpath": "aki_hand.png", "sompath": "aki_sound.mp3", 
        "cor": (0, 0, 255) # Vermelho
    }
}

ASSETS_MEMORIA = {}
for chave, prop in ASSETS_CONFIG.items():
    # Inicializando IMAGENS
    img = cv2.imread(prop["imgpath"], cv2.IMREAD_UNCHANGED) if os.path.exists(prop["imgpath"]) else None
    # Inicializando SONS
    snd = pygame.mixer.Sound(prop["sompath"]) if os.path.exists(prop["sompath"]) else None
    
    ASSETS_MEMORIA[chave] = {"imagem": img, "som": snd, "cor": prop["cor"]}

# =========================================================================
# 4. ROTINA PRINCIPAL DO MEDIAPIPE E LOOP
# =========================================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

csv_file = "hand_landmarks.csv"
dataset = carregar_dataset(csv_file)

cap = cv2.VideoCapture(0)
ultimo_gesto_visto = None  # <-- Variável de Estado para controlar o som repetitivo

while True:
    sucesso, frame = cap.read()
    if not sucesso: continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultado = hands.process(frame_rgb)
    h, w, _ = frame.shape
    
    gesto_identificado_agora = None
    gesto_chave_encontrada = None
    
    if resultado.multi_hand_landmarks:
        for hand_landmarks in resultado.multi_hand_landmarks:
            
            # Reconhecimento
            pontos = [val for p in hand_landmarks.landmark for val in (p.x, p.y)]
            gesto_identificado_agora = reconhecer_gesto(pontos, dataset)
            cor_atual = (0, 255, 0) # Default verde
            
            # Descobrir se temos assets vinculados
            if gesto_identificado_agora:
                for chave in ASSETS_MEMORIA:
                    if chave in gesto_identificado_agora.upper():
                        gesto_chave_encontrada = chave
                        cor_atual = ASSETS_MEMORIA[chave]["cor"]
                        break
            
            # Chama a função de EFEITO DE AURA visual e pinta os landmarks
            aplicar_efeito(frame, hand_landmarks, cor_atual)
            
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=cor_atual, thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=1, circle_radius=1)
            )

    # ---------------------------
    # Lógica do Estado (Sons e Textos)
    # ---------------------------
    if gesto_chave_encontrada:
        dados_asset = ASSETS_MEMORIA[gesto_chave_encontrada]
        
        # Só toca o som se FOR UM GESTO NOVO (evita som repicado contínuo)
        if gesto_chave_encontrada != ultimo_gesto_visto:
            tocar_som(dados_asset["som"])
            
        # Determinar posição do Texto Estilizado acima da Imagem inferior direita
        larg_img = 300 # Imagem maior, como pedido
        y_texto = h - 20 
        
        if dados_asset["imagem"] is not None:
            prop_alt = int(dados_asset["imagem"].shape[0] * (larg_img / dados_asset["imagem"].shape[1]))
            y_texto -= (prop_alt + 20) # Sobe o texto para não cruzar
            mostrar_imagem(frame, dados_asset["imagem"], larg_img)
            
        # Posicionar o nome alinhadinho no canto direito
        txt_size, _ = cv2.getTextSize(gesto_identificado_agora, cv2.FONT_HERSHEY_DUPLEX, 1, 2)
        x_texto = w - txt_size[0] - 20
        
        # Efeito de Letra
        Sombra = (255, 255, 255) if dados_asset["cor"] == (0,0,0) else (0,0,0)
        cv2.putText(frame, gesto_identificado_agora, (x_texto+3, y_texto+3), cv2.FONT_HERSHEY_DUPLEX, 1, Sombra, 3, cv2.LINE_AA)
        cv2.putText(frame, gesto_identificado_agora, (x_texto, y_texto), cv2.FONT_HERSHEY_DUPLEX, 1, dados_asset["cor"], 2, cv2.LINE_AA)
        

    ultimo_gesto_visto = gesto_chave_encontrada


    if gesto_chave_encontrada == "FOX":
        frame = aplicar_tremor(frame, intensidade=20)





    cv2.imshow('Câmera Super Poderes!', frame)
    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
         break
    elif tecla == ord('d'):
         dataset = carregar_dataset(csv_file)
         print(f"Dataset recarregado! Total de poses: {len(dataset)}")
         
    elif tecla == ord('s'):
         if resultado.multi_hand_landmarks:
             primeira_mao = resultado.multi_hand_landmarks[0]
             label = input("\nDigite o rótulo da posição (ex: fox, rasengan) e pressione Enter: ")
             linha_csv = [label.strip()]
             for ponto in primeira_mao.landmark:
                 linha_csv.extend([ponto.x, ponto.y])
             with open(csv_file, mode='a', newline='') as f:
                 writer = csv.writer(f)
                 writer.writerow(linha_csv)
             print(f"-> Gesto '{label}' salvo com sucesso! Pressione a tecla 'd' no vídeo para atualizar!")
         else:
             print("\nColoque a mão na frente da câmera primeiro para poder salvar os pontos!\n")
    
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
