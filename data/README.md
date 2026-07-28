# Organização dos dados

Os dados **não são versionados** (ver `.gitignore`). Organize localmente assim.

## Dados brutos (antes do preparo)

```
data/raw/
├── images/        # imagens aéreas (.jpg/.png)
└── labels/        # anotações no formato YOLO (.txt), um por imagem
```

Formato de cada linha do `.txt` (YOLO, coordenadas normalizadas 0–1):

```
<classe> <x_centro> <y_centro> <largura> <altura>
```

Como só há uma classe, `<classe>` é sempre `0`.

## Dataset preparado (após `prepare_dataset.py`)

```
data/dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Onde conseguir dados

- **Roboflow Universe** — buscar "cattle aerial", "livestock drone", "cow counting".
- **Anotação própria** — CVAT ou Roboflow, a partir de fotos do seu drone.

> ⚠️ Verifique a licença de qualquer dataset público antes de usar.
