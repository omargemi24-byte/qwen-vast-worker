import subprocess
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# 🛠️ CORRECCIÓN: Redirigir la salida del servidor al archivo log
log_file = open("server.log", "w")
subprocess.Popen(["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"], stdout=log_file, stderr=log_file)

def dummy_benchmark():
    return {
        "text": "Hello world.",
        "language": "English",
        "voice_ref_audio_b64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA=", 
        "voice_ref_text": "Hello"
    }

config = WorkerConfig(
    model_server_url="http://127.0.0.1",
    model_server_port=8000,
    model_log_file="server.log", # 🛠️ CORRECCIÓN: Usar ruta local
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
        on_load=["Application startup complete.", "INFO:     Application startup complete."], # <-- AÑADIDO LOG DE UVICORN
        on_error=["RuntimeError", "Traceback", "ERROR"],
        on_info=["INFO"]
    )
)

Worker(config).run()
