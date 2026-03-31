from fastapi import FastAPI, Request
import torch
import soundfile as sf
import io
import base64
import os
import re
import textwrap
import numpy as np
from qwen_tts import Qwen3TTSModel

app = FastAPI()
model = None

@app.on_event("startup")
def load_model():
    global model
    print("Cargando modelo Qwen3-TTS-0.6B...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base", 
        device_map="cuda:0", 
        dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    print("Application startup complete.") 

def agrupacion_inteligente(texto, limite=220):
    frases = re.split(r'(?<=[.!?])\s+', texto)
    bloques, bloque_actual = [], ""
    for f in frases:
        f = f.strip()
        if not f: continue
        if len(f) > limite:
            if bloque_actual: bloques.append(bloque_actual.strip()); bloque_actual = ""
            bloques.extend(textwrap.wrap(f, width=limite))
            continue
        if len(bloque_actual) + len(f) > limite and bloque_actual:
            bloques.append(bloque_actual.strip()); bloque_actual = f
        else:
            bloque_actual += " " + f if bloque_actual else f
    if bloque_actual: bloques.append(bloque_actual.strip())
    return bloques

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    text = data["text"]
    lang = data.get("language", "Spanish")
    
    # Reconstruir el audio de referencia
    audio_bytes = base64.b64decode(data["voice_ref_audio_b64"])
    with open("temp_ref.wav", "wb") as f:
        f.write(audio_bytes)
        
    prompt = model.create_voice_clone_prompt("temp_ref.wav", data["voice_ref_text"])
    bloques = agrupacion_inteligente(text)
    
    audios_generados = []
    with torch.inference_mode():
        for b in bloques:
            wavs, sr = model.generate_voice_clone(b, language=lang, voice_clone_prompt=prompt)
            audios_generados.append(wavs[0].cpu().numpy())
            
    audio_final = np.concatenate(audios_generados)
    buf = io.BytesIO()
    sf.write(buf, audio_final, sr, format='WAV')
    
    os.remove("temp_ref.wav")
    return {"audio_b64": base64.b64encode(buf.getvalue()).decode('utf-8')}