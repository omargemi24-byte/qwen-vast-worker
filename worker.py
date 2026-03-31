import subprocess
import io
import base64
import time
import numpy as np
import soundfile as sf
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# 🛠️ Redirigir la salida del servidor al archivo log
log_file = open("server.log", "w", buffering=1)
subprocess.Popen(
    ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
    stdout=log_file,
    stderr=log_file
)

# Esperamos 3 segundos para que el proceso OS arranque antes de que el SDK empiece a leer el log
time.sleep(3)

# 🧠 Crear un audio de silencio en RAM para el benchmark
def get_dummy_wav_b64():
    buf = io.BytesIO()
    sf.write(buf, np.zeros(12000), 24000, format='WAV')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

DUMMY_AUDIO_B64 = get_dummy_wav_b64()

def dummy_benchmark():
    return {
        "text": "Hello world.",
        "language": "English",
        "voice_ref_audio_b64": DUMMY_AUDIO_B64, 
        "voice_ref_text": "Hello"
    }

config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8000,
    model_log_file="server.log",
    handlers=[
        HandlerConfig(
            route="/generate",
            allow_parallel_requests=False, 
            max_queue_time=600.0,
            workload_calculator=lambda p: float(len(p.get("text", ""))),
            benchmark_config=BenchmarkConfig(
                generator=dummy_benchmark,
                runs=2,
                concurrency=1
            )
        )
    ],
    log_action_config=LogActionConfig(
        on_load=["Application startup complete.", "INFO:     Application startup complete."], 
        on_error=["RuntimeError", "Traceback", "ERROR"],
        on_info=["INFO"]
    )
)

Worker(config).run()
