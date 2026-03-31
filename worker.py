import subprocess
import io
import base64
import numpy as np
import soundfile as sf
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# 🛠️ CORRECCIÓN: Redirigir la salida del servidor al archivo log
log_file = open("server.log", "w")
subprocess.Popen(["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"], stdout=log_file, stderr=log_file)

# 🧠 SOLUCIÓN AUTOMÁTICA: Crear un audio de silencio en RAM para el test
def get_dummy_wav_b64():
    buf = io.BytesIO()
    # Genera 0.5 segundos de silencio a 24000Hz
    sf.write(buf, np.zeros(12000), 24000, format='WAV')
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Guardamos el audio falso en una variable
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
        # 🛠️ CORRECCIÓN: Compatibilidad total con los logs de Uvicorn
        on_load=["Application startup complete.", "INFO:     Application startup complete."], 
        on_error=["RuntimeError", "Traceback", "ERROR"],
        on_info=["INFO"]
    )
)

Worker(config).run()
