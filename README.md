# 🖐️ Sistema de Reconhecimento de Gestos com Super Poderes

Um sistema de visão computacional em tempo real que detecta gestos da mão pela webcam e ativa efeitos visuais, sonoros e animações inspirados em três animes — construído com MediaPipe, OpenCV e Pygame.

Nota: Utilizei o AntiGravity e, dentro dele, um agente de IA para gerar o código em Python. O objetivo principal era entender como a visão computacional funciona e, a partir disso, desenvolver algo que eu considero divertido.

---

## ✨ Demonstração

Ao realizar um gesto reconhecido na frente da câmera, o sistema:
- Exibe uma **aura colorida** ao redor da mão
- Toca um **efeito sonoro** temático
- Mostra uma **imagem** do personagem correspondente
- Aplica **efeitos visuais** exclusivos por gesto

| Gesto | Personagem | Anime | Efeito |
|-------|-----------|-------|--------|
| TIGER | Naruto | Naruto | Aura azul |
| DIVINE DOGS | Megumi | Jujutsu Kaisen | Aura preta |
| FOX | Aki | Chainsaw Man | Aura vermelha + tremor de câmera |

---

## 🧠 Como funciona

O reconhecimento **não usa uma rede neural pré-treinada** — ele foi construído com um algoritmo de comparação de distância euclidiana. O processo tem três etapas:

**1. Detecção:** O MediaPipe localiza 21 pontos da mão em cada frame da câmera.

**2. Normalização:** As coordenadas brutas são transformadas matematicamente para que o gesto seja reconhecido independente do tamanho da mão ou posição na tela.

**3. Comparação:** Os pontos normalizados são comparados com os gestos salvos no dataset (`hand_landmarks.csv`). O gesto mais próximo matematicamente é o reconhecido.

---

## 🚀 Como rodar

**Pré-requisitos:** Python 3.10+ e uma webcam.

> 💡 **Sem webcam?** Sem problema! Você pode usar o celular como câmera virtual. Recomendo o **[Iriun Webcam](https://iriun.com/)** (mais simples, via Wi-Fi) ou o **[DroidCam](https://www.dev47apps.com/)** (mais estável, via USB).

Clone o repositório:
```bash
git clone https://github.com/jwmariaaa/gesture-recognition-.git
cd gesture-recognition-
```

Instale as dependências:
```bash
pip install -r requirements.txt
pip install pygame numpy
```

Execute o programa:
```bash
python hand_tracking.py
```

---

## ⌨️ Controles durante a execução

| Tecla | Ação |
|-------|------|
| `q` | Encerra o programa |
| `s` | Salva a pose atual da mão no dataset (treinar novo gesto) |
| `d` | Recarrega o dataset sem reiniciar o programa |

---

## ➕ Como adicionar novos gestos

O sistema é extensível — você pode treinar novos gestos sem modificar o código:

1. Rode o programa e posicione a mão fazendo o gesto desejado
2. Pressione `s` e digite um nome para o gesto no terminal
3. Pressione `d` no vídeo para recarregar o dataset
4. O gesto já está ativo!

Para vincular imagem, som e cor ao novo gesto, adicione uma entrada no dicionário `ASSETS_CONFIG` dentro do `hand_tracking.py`.

---

## 📁 Estrutura do projeto

```
gesture-recognition/
├── hand_tracking.py        # Script principal
├── hand_landmarks.csv      # Dataset de gestos treinados
├── requirements.txt        # Dependências do projeto
├── aki_hand.png            # Imagem do gesto - Aki (Chainsaw Man)
├── megumi_hand.png         # Imagem do gesto - Megumi (Jujutsu Kaisen)
└── naruto_hand.png         # Imagem do gesto - Naruto
```

---

## 🛠️ Tecnologias utilizadas

- **[MediaPipe](https://mediapipe.dev/)** — detecção e rastreamento dos 21 pontos da mão em tempo real
- **[OpenCV](https://opencv.org/)** — captura da câmera, efeitos visuais e renderização dos frames
- **[Pygame](https://www.pygame.org/)** — reprodução dos efeitos sonoros sem travar o vídeo
- **[NumPy](https://numpy.org/)** — operações matemáticas para os efeitos de aura e tremor

---

## 👩‍💻 Autora

Desenvolvido por **[@jwmariaaa](https://github.com/jwmariaaa)**  
Estudante de Análise e Desenvolvimento de Sistemas — 1º semestre
