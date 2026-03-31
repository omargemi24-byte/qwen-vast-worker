from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
import torch
import torchaudio
import io
import base64
import os
import re
import textwrap
import numpy as np
import sys
from qwen_tts import Qwen3TTSModel

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Cargando modelo Qwen3-TTS-0.6B...", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )
    print("Application startup complete.", flush=True)
    print("Application startup complete.", file=sys.stderr, flush=True)
    yield

app = FastAPI(lifespan=lifespan)

def agrupacion_inteligente(texto, limite=220):
    frases = re.split(r'(?<=[.!?])\s+', texto)
    bloques, bloque_actual = [], ""
    for f in frases:
        f = f.strip()
        if not f:
            continue
        if len(f) > limite:
            if bloque_actual:
                bloques.append(bloque_actual.strip())
                bloque_actual = ""
            bloques.extend(textwrap.wrap(f, width=limite))
            continue
        if len(bloque_actual) + len(f) > limite and bloque_actual:
            bloques.append(bloque_actual.strip())
            bloque_actual = f
        else:
            bloque_actual += " " + f if bloque_actual else f
    if bloque_actual:
        bloques.append(bloque_actual.strip())
    return bloques

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    text = data["text"]
    lang = data.get("language", "Spanish")
    batch_size = data.get("batch_size", 12)

    audio_bytes = base64.b64decode(data["voice_ref_audio_b64"])
    with open("temp_ref.wav", "wb") as f:
        f.write(audio_bytes)

    prompt = model.create_voice_clone_prompt("temp_ref.wav", data["voice_ref_text"])
    bloques = agrupacion_inteligente(text)

    audios_generados = []
    sr = None

    with torch.inference_mode():
        for i in range(0, len(bloques), batch_size):
            lote = bloques[i:i + batch_size]
            lista_idiomas = [lang] * len(lote)
            wavs, sr = model.generate_voice_clone(lote, language=lista_idiomas, voice_clone_prompt=prompt)
            for wav in wavs:
                audios_generados.append(wav.cpu().numpy())

    audio_final = np.concatenate(audios_generados)

    buf = io.BytesIO()
    torchaudio.save(buf, torch.tensor(audio_final).unsqueeze(0), sr, format="wav")
    buf.seek(0)  # ← imprescindible: resetea el puntero al inicio para leer

    os.remove("temp_ref.wav")
    return {"audio_b64": base64.b64encode(buf.getvalue()).decode('utf-8')}
