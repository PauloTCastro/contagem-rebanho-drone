"""
Demo interativa da contagem de gado por drone.

Executar:
    streamlit run app_streamlit.py
"""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from infer import contar, anotar_e_salvar  # noqa: E402

st.set_page_config(page_title="Contagem de Rebanho por Drone", page_icon="🐄")
st.title("🐄 Contagem de Rebanho por Drone")
st.caption("Detecção e contagem de gado em imagens aéreas · YOLO + SAHI")

with st.sidebar:
    st.header("Configuração")
    weights = st.text_input("Pesos do modelo",
                            os.getenv("WEIGHTS",
                                      "runs/detect/contagem_gado/weights/best.pt"))
    slice_size = st.select_slider("Tamanho do bloco (SAHI)",
                                  [512, 640, 768, 1024, 1280], value=1024)
    overlap = st.slider("Sobreposição dos blocos", 0.0, 0.5, 0.2, 0.05)
    conf = st.slider("Confiança mínima", 0.1, 0.9, 0.3, 0.05)
    device = st.selectbox("Dispositivo", ["cpu", "0"])

arquivo = st.file_uploader("Envie uma imagem aérea",
                           type=["jpg", "jpeg", "png", "tif", "tiff"])

if arquivo:
    if not Path(weights).exists():
        st.error(f"Pesos não encontrados: '{weights}'. Treine o modelo primeiro.")
        st.stop()

    with tempfile.NamedTemporaryFile(suffix=Path(arquivo.name).suffix,
                                     delete=False) as tmp:
        tmp.write(arquivo.read())
        origem = tmp.name

    with st.spinner("Detectando animais..."):
        contagem, boxes = contar(weights, origem, slice_size, overlap, conf, device)
        saida = origem + "_out.jpg"
        anotar_e_salvar(origem, boxes, contagem, saida)

    st.metric("Total de animais detectados", contagem)
    st.image(cv2.cvtColor(cv2.imread(saida), cv2.COLOR_BGR2RGB),
             caption="Detecções", use_container_width=True)

    Path(origem).unlink(missing_ok=True)
    Path(saida).unlink(missing_ok=True)
