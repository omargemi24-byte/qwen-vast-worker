import subprocess
from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

subprocess.Popen(["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "8000"])

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
    model_log_file="/app/server.log", 
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
        on_load=["Application startup complete."], 
        on_error=["RuntimeError", "Traceback"],
        on_info=["INFO"]
    )
)

Worker(config).run()