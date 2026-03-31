import subprocess
import io
import base64
import wave
import time
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# Lanzar uvicorn ANTES de cualquier otra cosa
log_file = open("server.log", "w", buffering=1)
subprocess.Popen(
    ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info"],
    stdout=log_file,
    stderr=log_file
)

time.sleep(5)

# ✅ WAV de silencio con módulo built-in, sin soundfile ni numpy
def get_dummy_wav_b64():
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)       # Mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(24000)   # 24kHz
        wf.writeframes(b'\x00' * 24000 * 2)  # 1 segundo de silencio
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
