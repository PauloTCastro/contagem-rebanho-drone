# 🐄 Contagem de Rebanho por Drone

Detecção e **contagem automática de gado** em pastagens a partir de imagens aéreas
(drone/VANT). O objetivo é contar todos os animais de uma pastagem inteira em
segundos, substituindo a contagem manual que é lenta, sujeita a erro e cara em
propriedades grandes.

O projeto usa **YOLO** (Ultralytics) para detecção e **SAHI** (*Slicing Aided
Hyper Inference*) para lidar com o principal desafio de imagem aérea: os animais
são objetos muito pequenos dentro de uma imagem enorme.

> Projeto de portfólio em Visão Computacional. Foco em reprodutibilidade,
> avaliação quantitativa e um serviço de inferência utilizável (API + demo).

---

## 📌 Índice

- [Motivação](#-motivação)
- [O desafio técnico](#-o-desafio-técnico-por-que-não-é-só-rodar-um-yolo)
- [Abordagem](#-abordagem)
- [Estrutura do repositório](#-estrutura-do-repositório)
- [Dataset](#-dataset)
- [Como usar](#-como-usar)
- [Avaliação](#-avaliação)
- [Roadmap](#-roadmap)
- [Referências](#-referências)

---

## 🎯 Motivação

Contar rebanho manualmente numa pastagem grande é demorado e impreciso. Um drone
sobrevoa a área, captura ortomosaicos ou fotos de alta resolução, e o modelo
retorna a contagem com as detecções sobrepostas para conferência visual. Casos de
uso: gestão de rebanho, conferência de compra/venda, monitoramento sanitário e
auditoria.

## 🧠 O desafio técnico (por que não é só "rodar um YOLO")

Numa foto de drone de, digamos, `5000 × 4000` px, cada boi pode ocupar
`~40 × 25` px. Se você redimensionar a imagem inteira para a entrada do detector
(ex.: `640 × 640`), cada animal vira um punhado de pixels e o modelo simplesmente
não os enxerga. Esse é o problema clássico de **detecção de objetos pequenos em
imagens de alta resolução**.

A solução adotada é **inferência por fatiamento (tiling)** com a biblioteca
[SAHI](https://github.com/obss/sahi):

1. A imagem é dividida em blocos sobrepostos (ex.: `1024 × 1024`, 20% de overlap).
2. O detector roda em cada bloco, onde os animais têm tamanho "normal".
3. As detecções de todos os blocos são reprojetadas para a imagem original e
   mescladas com **NMS**, eliminando duplicatas nas bordas dos blocos.

Isso melhora drasticamente o recall em objetos pequenos sem precisar de um modelo
maior.

## 🏗 Abordagem

| Etapa | Escolha | Observação |
|---|---|---|
| Detector | YOLOv8 / YOLO11 (Ultralytics) | Rápido, fácil de treinar, boa base |
| Inferência aérea | SAHI (tiling + NMS) | Núcleo do projeto |
| Contagem | nº de detecções após NMS | Métrica de negócio |
| Avaliação | mAP (detecção) + **MAE/RMSE** (contagem) | Contagem é o que importa |
| Serviço | FastAPI + demo em Streamlit | Upload da imagem → contagem + imagem anotada |

> **Alternativa considerada:** para pastagens com aglomeração extrema (currais,
> bebedouros), regressão de *mapa de densidade* (estilo CSRNet) tende a superar a
> detecção. Fica registrado no [Roadmap](#-roadmap).

## 📁 Estrutura do repositório

```
contagem-rebanho-drone/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── data.yaml            # config do dataset no formato YOLO
├── data/
│   └── README.md            # como organizar os dados (não versionados)
├── src/
│   ├── prepare_dataset.py   # organiza/split e (opcional) fatia p/ treino
│   ├── train.py             # treino do YOLO
│   ├── infer.py             # inferência com SAHI + contagem  ⭐
│   └── evaluate.py          # MAE/RMSE de contagem + mAP
├── api/
│   └── main.py              # API FastAPI
├── app_streamlit.py         # demo interativa
└── notebooks/
    └── 01_eda.ipynb         # análise exploratória
```

## 🗂 Dataset

Este repositório **não versiona os dados**. Opções para obtê-los:

- **Roboflow Universe** buscar por *"cattle aerial"*, *"livestock drone"*,
  *"cow counting"*. Vários datasets públicos já vêm no formato YOLO.
- **Anotação própria** capturar imagens com drone e anotar em
  [CVAT](https://www.cvat.ai/) ou [Roboflow](https://roboflow.com/). Recomendado
  para dados representativos do seu bioma/tipo de gado.
- **Aumento sintético** para robustez, aplicar variações de altitude, ângulo,
  iluminação e sombra.

Esperado apenas **uma classe**: `boi` (ou `cattle`). Organize no formato YOLO
conforme [`data/README.md`](data/README.md).

> ⚠️ Confira a **licença de uso** de qualquer dataset público antes de usar em
> portfólio ou produção.

## 🚀 Como usar

### 1. Instalação

```bash
git clone https://github.com/PauloTCastro/contagem-rebanho-drone.git
cd contagem-rebanho-drone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Preparar o dataset

```bash
python src/prepare_dataset.py \
  --raw-dir data/raw \
  --out-dir data/dataset \
  --tile 1024 --overlap 0.2   # fatia imagens grandes para o treino
```

### 3. Treinar

```bash
python src/train.py --data configs/data.yaml --model yolov8m.pt --epochs 100 --imgsz 1024
```

### 4. Inferir (contagem em imagem aérea completa)

```bash
python src/infer.py \
  --weights runs/detect/train/weights/best.pt \
  --source data/exemplo/pasto.jpg \
  --slice 1024 --overlap 0.2 \
  --out assets/resultado.jpg
# -> imprime a contagem e salva a imagem anotada
```

### 5. Avaliar

```bash
python src/evaluate.py \
  --weights runs/detect/train/weights/best.pt \
  --data configs/data.yaml
# -> mAP@0.5, mAP@0.5:0.95, MAE e RMSE de contagem
```

### 6. Servir

```bash
# API
uvicorn api.main:app --reload
# Demo
streamlit run app_streamlit.py
```

## 📊 Avaliação

O projeto não avalia só detecção — o que interessa ao pecuarista é o **erro de
contagem**:

- **MAE** (Erro Absoluto Médio): em média, quantos animais de diferença por imagem.
- **RMSE**: penaliza erros grandes, útil para detectar falhas graves.
- **mAP@0.5 / mAP@0.5:0.95**: qualidade das detecções (suporte técnico).

*(Preencha com seus resultados após treinar.)*

| Métrica | Valor |
|---|---|
| mAP@0.5 | — |
| mAP@0.5:0.95 | — |
| MAE (contagem) | — |
| RMSE (contagem) | — |

## 🗺 Roadmap

- [ ] Baseline YOLO sem tiling (para evidenciar o ganho do SAHI)
- [ ] Ablação: tamanho de bloco vs. overlap vs. recall/tempo
- [ ] Regressão por mapa de densidade para zonas de aglomeração
- [ ] Suporte a ortomosaico (GeoTIFF) com georreferência das detecções
- [ ] Deploy AWS (ECS Fargate + S3 + API Gateway)
- [ ] Rastreamento em vídeo de voo (contagem sem dupla contagem entre frames)

## 📚 Referências

- Ultralytics YOLO — https://docs.ultralytics.com
- SAHI: *Slicing Aided Hyper Inference* — Akyon et al., 2022 — https://github.com/obss/sahi
- CSRNet (contagem por densidade) — Li et al., CVPR 2018

---

**Autor:** Paulo de Tarso Castro Silva ·
[GitHub](https://github.com/PauloTCastro) ·

Licença: MIT
