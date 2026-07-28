"""
Prepara o dataset de contagem de gado para treino com YOLO.

Faz duas coisas:
  1. Divide (split) imagens+labels em train/val/test.
  2. (Opcional) Fatia imagens grandes em blocos menores para o treino, ajustando
     as bounding boxes. Isso ajuda o detector a aprender objetos pequenos.

Espera dados brutos no formato:
    data/raw/images/*.jpg   e   data/raw/labels/*.txt   (YOLO, classe 0)

Uso:
    python src/prepare_dataset.py --raw-dir data/raw --out-dir data/dataset \
        --tile 1024 --overlap 0.2
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def listar_pares(raw_dir: Path):
    """Retorna pares (imagem, label) existentes."""
    img_dir, lbl_dir = raw_dir / "images", raw_dir / "labels"
    pares = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in IMG_EXTS:
            continue
        lbl = lbl_dir / f"{img.stem}.txt"
        pares.append((img, lbl if lbl.exists() else None))
    return pares


def ler_labels(lbl_path):
    """Lê labels YOLO -> lista de (classe, xc, yc, w, h) normalizados."""
    if lbl_path is None or not lbl_path.exists():
        return []
    linhas = []
    for ln in lbl_path.read_text().strip().splitlines():
        if not ln.strip():
            continue
        c, xc, yc, w, h = map(float, ln.split())
        linhas.append((int(c), xc, yc, w, h))
    return linhas


def fatiar_imagem(img_path, lbl_path, tile, overlap, out_img_dir, out_lbl_dir):
    """
    Fatia uma imagem grande em blocos 'tile x tile' com sobreposição.
    Reprojeta as boxes para cada bloco (descarta boxes com centro fora do bloco).
    Salva os blocos que contêm ao menos um animal.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[aviso] não consegui ler {img_path}")
        return 0
    H, W = img.shape[:2]
    labels = ler_labels(lbl_path)

    # converte labels normalizados -> pixels absolutos (x1,y1,x2,y2)
    boxes_px = []
    for c, xc, yc, w, h in labels:
        bw, bh = w * W, h * H
        cx, cy = xc * W, yc * H
        boxes_px.append((c, cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))

    step = int(tile * (1 - overlap))
    salvos = 0
    for y0 in range(0, max(1, H - tile + step), step):
        for x0 in range(0, max(1, W - tile + step), step):
            x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
            x0c, y0c = x1 - tile, y1 - tile  # garante blocos de tamanho fixo
            x0c, y0c = max(0, x0c), max(0, y0c)
            crop = img[y0c:y1, x0c:x1]
            ch, cw = crop.shape[:2]

            novas = []
            for c, bx1, by1, bx2, by2 in boxes_px:
                # centro da box dentro do bloco?
                mcx, mcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                if not (x0c <= mcx < x1 and y0c <= mcy < y1):
                    continue
                # recorta a box aos limites do bloco
                nx1, ny1 = max(bx1, x0c), max(by1, y0c)
                nx2, ny2 = min(bx2, x1), min(by2, y1)
                w_n, h_n = (nx2 - nx1) / cw, (ny2 - ny1) / ch
                xc_n = ((nx1 + nx2) / 2 - x0c) / cw
                yc_n = ((ny1 + ny2) / 2 - y0c) / ch
                if w_n > 0 and h_n > 0:
                    novas.append((c, xc_n, yc_n, w_n, h_n))

            if not novas:  # só guarda blocos com gado
                continue
            nome = f"{img_path.stem}_{x0c}_{y0c}"
            cv2.imwrite(str(out_img_dir / f"{nome}.jpg"), crop)
            (out_lbl_dir / f"{nome}.txt").write_text(
                "\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                          for c, xc, yc, w, h in novas)
            )
            salvos += 1
    return salvos


def copiar_par(img_path, lbl_path, out_img_dir, out_lbl_dir):
    shutil.copy2(img_path, out_img_dir / img_path.name)
    dst = out_lbl_dir / f"{img_path.stem}.txt"
    if lbl_path and lbl_path.exists():
        shutil.copy2(lbl_path, dst)
    else:
        dst.write_text("")  # imagem sem gado (negativo)


def main():
    ap = argparse.ArgumentParser(description="Prepara dataset YOLO de contagem de gado")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/dataset"))
    ap.add_argument("--splits", nargs=3, type=float, default=[0.8, 0.1, 0.1],
                    help="proporções train val test")
    ap.add_argument("--tile", type=int, default=0,
                    help="tamanho do bloco p/ fatiar (0 = não fatiar)")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    pares = listar_pares(args.raw_dir)
    if not pares:
        raise SystemExit(f"Nenhuma imagem encontrada em {args.raw_dir/'images'}")
    random.shuffle(pares)

    n = len(pares)
    n_tr = int(n * args.splits[0])
    n_val = int(n * args.splits[1])
    grupos = {
        "train": pares[:n_tr],
        "val": pares[n_tr:n_tr + n_val],
        "test": pares[n_tr + n_val:],
    }

    for split, itens in grupos.items():
        out_img = args.out_dir / "images" / split
        out_lbl = args.out_dir / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        total_blocos = 0
        for img_path, lbl_path in itens:
            if args.tile > 0:
                total_blocos += fatiar_imagem(
                    img_path, lbl_path, args.tile, args.overlap, out_img, out_lbl
                )
            else:
                copiar_par(img_path, lbl_path, out_img, out_lbl)
        extra = f" ({total_blocos} blocos)" if args.tile > 0 else ""
        print(f"[{split}] {len(itens)} imagens{extra}")

    print(f"\nDataset pronto em: {args.out_dir}")


if __name__ == "__main__":
    main()
