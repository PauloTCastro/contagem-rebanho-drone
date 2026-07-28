"""
API de contagem de gado por drone.

Sobe um serviço FastAPI que recebe uma imagem aérea e devolve a contagem de
animais (JSON) e a imagem anotada (PNG).

Executar:
    uvicorn api.main:app --reload
    # docs interativas em http://localhost:8000/docs

Variáveis de ambiente:
    WEIGHTS   caminho do checkpoint (default: runs/detect/contagem_gado/weights/best.pt)
    DEVICE    'cpu' ou '0' (default: cpu)
"""
import base64
import os
import sys
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# permite importar de src/
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from infer import contar, anotar_e_salvar  # noqa: E402

WEIGHTS = os.getenv("WEIGHTS", "runs/detect/contagem_gado/weights/best.pt")
DEVICE = os.getenv("DEVICE", "cpu")

app = FastAPI(
    title="Contagem de Rebanho por Drone",
    description="Detecção e contagem de gado em imagens aéreas (YOLO + SAHI).",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "weights": WEIGHTS, "existe": Path(WEIGHTS).exists()}


@app.post("/contar")
async def contar_endpoint(
    file: UploadFile = File(...),
    slice_size: int = 1024,
    overlap: float = 0.2,
    conf: float = 0.3,
    retornar_imagem: bool = True,
):
    """Recebe uma imagem aérea e retorna a contagem de animais."""
    if not Path(WEIGHTS).exists():
        return JSONResponse(
            status_code=503,
            content={"erro": f"Pesos não encontrados em '{WEIGHTS}'. "
                             f"Treine o modelo ou ajuste a env WEIGHTS."},
        )

    conteudo = await file.read()
    with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix,
                                     delete=False) as tmp:
        tmp.write(conteudo)
        origem = tmp.name

    contagem, boxes = contar(WEIGHTS, origem, slice_size, overlap, conf, DEVICE)

    resposta = {"contagem": contagem, "arquivo": file.filename}

    if retornar_imagem:
        saida = origem + "_anotada.jpg"
        anotar_e_salvar(origem, boxes, contagem, saida)
        img_bytes = cv2.imencode(".jpg", cv2.imread(saida))[1].tobytes()
        resposta["imagem_anotada_base64"] = base64.b64encode(img_bytes).decode()
        Path(saida).unlink(missing_ok=True)

    Path(origem).unlink(missing_ok=True)
    return resposta
