from fastapi import FastAPI, Request, HTTPException
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
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON inválido")

    text = data.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Campo 'text' requerido")

    lang = data.get("language", "Spanish")
    batch_size = data.get("batch_size", 12)
    voice_ref_audio_b64 = data.get("voice_ref_audio_b64")
    voice_ref_text = data.get("voice_ref_text", "")  # ← Opcional ahora

    # --- BENCHMARK / MODO SIN REFERENCIA ---
    # El SDK de Vast.ai no envía voice_ref_text en el benchmark.
    # Si falta la referencia de voz, generamos sin clonar (TTS directo).
    if not voice_ref_text or not voice_ref_audio_b64:
        with torch.inference_mode():
            bloques = agrupacion_inteligente(text)
            lista_idiomas = [lang] * len(bloques)
            wavs, sr = model.generate(bloques, language=lista_idiomas)
            audios = [w.cpu().numpy() for w in wavs]

        audio_final = np.concatenate(audios)
        buf = io.BytesIO()
        torchaudio.save(buf, torch.tensor(audio_final).unsqueeze(0), sr, format="wav")
        buf.seek(0)
        return {"audio_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}

    # --- MODO NORMAL: Voice Clone ---
    try:
        audio_bytes = base64.b64decode(voice_ref_audio_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="voice_ref_audio_b64 inválido")

    ref_path = f"temp_ref_{os.getpid()}.wav"  # ← Nombre único para evitar race conditions
    try:
        with open(ref_path, "wb") as f:
            f.write(audio_bytes)

        prompt = model.create_voice_clone_prompt(ref_path, voice_ref_text)
        bloques = agrupacion_inteligente(text)

        audios_generados = []
        sr = None

        with torch.inference_mode():
            for i in range(0, len(bloques), batch_size):
                lote = bloques[i:i + batch_size]
                lista_idiomas = [lang] * len(lote)
                wavs, sr = model.generate_voice_clone(
                    lote, language=lista_idiomas, voice_clone_prompt=prompt
                )
                for wav in wavs:
                    audios_generados.append(wav.cpu().numpy())
    finally:
        if os.path.exists(ref_path):
            os.remove(ref_path)

    audio_final = np.concatenate(audios_generados)
    buf = io.BytesIO()
    torchaudio.save(buf, torch.tensor(audio_final).unsqueeze(0), sr, format="wav")
    buf.seek(0)
    return {"audio_b64": base64.b64encode(buf.getvalue()).decode("utf-8")}
