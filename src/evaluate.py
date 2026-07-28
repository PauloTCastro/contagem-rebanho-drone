"""
Avaliação do modelo de contagem de gado.

Reporta duas coisas:
  1. mAP (qualidade da detecção) via validador do Ultralytics.
  2. MAE e RMSE de CONTAGEM — o que realmente importa para o caso de uso:
     compara o nº de animais previstos vs. o real, por imagem.

A contagem usa inferência fatiada (SAHI), igual à produção.

Uso:
    python src/evaluate.py --weights best.pt --data configs/data.yaml \
        --split val --slice 1024 --overlap 0.2
"""
import argparse
import math
from pathlib import Path

import yaml
from ultralytics import YOLO

from infer import contar


def carregar_split(data_yaml, split):
    """Retorna (dir_imagens, dir_labels) do split escolhido."""
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    raiz = (Path(data_yaml).parent / cfg["path"]).resolve()
    img_dir = raiz / cfg.get(split, f"images/{split}")
    lbl_dir = Path(str(img_dir).replace("images", "labels"))
    return img_dir, lbl_dir


def contar_gt(lbl_path):
    """Conta linhas (animais) num arquivo de label YOLO."""
    if not lbl_path.exists():
        return 0
    return sum(1 for ln in lbl_path.read_text().splitlines() if ln.strip())


def main():
    ap = argparse.ArgumentParser(description="Avaliação - contagem de gado")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--slice", type=int, default=1024, dest="slice_size")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # ---- 1. mAP (detecção) ----
    print("== Detecção (Ultralytics) ==")
    metrics = YOLO(args.weights).val(data=args.data, split=args.split)
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")

    # ---- 2. Contagem (MAE/RMSE) com SAHI ----
    print("\n== Contagem (SAHI) ==")
    img_dir, lbl_dir = carregar_split(args.data, args.split)
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    imagens = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]

    erros = []
    for img in imagens:
        pred, _ = contar(args.weights, img, args.slice_size,
                         args.overlap, args.conf, args.device)
        gt = contar_gt(lbl_dir / f"{img.stem}.txt")
        erros.append(pred - gt)
        print(f"{img.name}: previsto={pred}  real={gt}  erro={pred - gt:+d}")

    if erros:
        n = len(erros)
        mae = sum(abs(e) for e in erros) / n
        rmse = math.sqrt(sum(e * e for e in erros) / n)
        print(f"\nImagens avaliadas: {n}")
        print(f"MAE  (contagem): {mae:.2f}")
        print(f"RMSE (contagem): {rmse:.2f}")
    else:
        print("Nenhuma imagem encontrada para avaliar.")


if __name__ == "__main__":
    main()
