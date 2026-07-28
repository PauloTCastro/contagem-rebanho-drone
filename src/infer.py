"""
Inferência de contagem de gado em imagem aérea completa, usando SAHI.

Núcleo do projeto: imagens de drone são grandes e os animais, pequenos. Rodar o
detector na imagem inteira redimensionada perde os animais. O SAHI fatia a imagem
em blocos, detecta em cada um e mescla os resultados com NMS — recuperando os
objetos pequenos.

Uso:
    python src/infer.py --weights best.pt --source pasto.jpg \
        --slice 1024 --overlap 0.2 --conf 0.3 --out assets/resultado.jpg
"""
import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def contar(weights, source, slice_size=1024, overlap=0.2, conf=0.3,
           device="cpu", out=None):
    """Roda inferência fatiada e retorna (contagem, lista_de_boxes)."""
    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",   # em versões antigas do SAHI: "yolov8"
        model_path=str(weights),
        confidence_threshold=conf,
        device=device,
    )

    resultado = get_sliced_prediction(
        image=str(source),
        detection_model=model,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",          # remove duplicatas nas bordas dos blocos
        postprocess_match_metric="IOU",
        postprocess_match_threshold=0.5,
        verbose=0,
    )

    boxes = resultado.object_prediction_list
    return len(boxes), boxes


def anotar_e_salvar(source, boxes, contagem, out_path):
    """Desenha as caixas e o total na imagem e salva."""
    img = cv2.imread(str(source))
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.bbox.to_xyxy())
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    texto = f"Total: {contagem}"
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    cv2.rectangle(img, (10, 10), (20 + tw, 30 + th), (0, 0, 0), -1)
    cv2.putText(img, texto, (15, 25 + th), cv2.FONT_HERSHEY_SIMPLEX,
                1.4, (0, 255, 0), 3)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser(description="Contagem de gado (SAHI + YOLO)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="imagem aérea")
    ap.add_argument("--slice", type=int, default=1024, dest="slice_size")
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="assets/resultado.jpg")
    args = ap.parse_args()

    contagem, boxes = contar(
        args.weights, args.source, args.slice_size,
        args.overlap, args.conf, args.device,
    )
    anotar_e_salvar(args.source, boxes, contagem, args.out)

    print(f"\n🐄 Animais detectados: {contagem}")
    print(f"🖼  Imagem anotada salva em: {args.out}")


if __name__ == "__main__":
    main()
