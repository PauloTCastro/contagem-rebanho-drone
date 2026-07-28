"""
Treina um detector YOLO (Ultralytics) para contagem de gado.

Uso:
    python src/train.py --data configs/data.yaml --model yolov8m.pt \
        --epochs 100 --imgsz 1024 --batch 16
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description="Treino YOLO - contagem de gado")
    ap.add_argument("--data", default="configs/data.yaml")
    ap.add_argument("--model", default="yolov8m.pt",
                    help="checkpoint base (ex.: yolov8n/s/m.pt, yolo11m.pt)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="use blocos grandes: objetos pequenos precisam de resolução")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="'0', 'cpu', etc. (auto se None)")
    ap.add_argument("--name", default="contagem_gado")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # augmentations úteis p/ imagem aérea:
        degrees=180,      # animais aparecem em qualquer orientação
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        patience=25,      # early stopping
    )
    print("Treino concluído. Pesos em runs/detect/<name>/weights/best.pt")


if __name__ == "__main__":
    main()
