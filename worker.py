import subprocess
import time
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# Lanzar uvicorn ANTES de cualquier import pesado
log_file = open("server.log", "w", buffering=1)
subprocess.Popen(
    ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
    stdout=log_file,
    stderr=log_file
)

# Dar tiempo suficiente al proceso OS para arrancar
time.sleep(5)

# ✅ Imports pesados DENTRO de la función — no bloquean el arranque de uvicorn
def get_dummy_wav_b64():
    import io
    import base64
    import numpy as np
    import soundfile as sf
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
        on_error=["RuntimeError", "Traceback", "CUDA error", "Killed"],
        on_info=["INFO"]
    )
)

Worker(config).run()
