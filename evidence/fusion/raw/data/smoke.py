"""Bounded TP2 Graph development smoke; always clean up its own service group."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["B0", "B1"], default="B0")
    parser.add_argument("--model", default="/data/shared_upload/models/vspec/Qwen3.5-35B-A3B")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--batch-invariant", choices=["0", "1"], default="1")
    parser.add_argument("--optimization", choices=["both", "persistent-matmul-empty", "linear-swiglu-graph"], default="both")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    run = root / "results" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + args.arm)
    run.mkdir(parents=True)
    state = {"arm": args.arm, "evidence_level": "development-smoke", "status": "starting"}
    save(root / "status.json", {**state, "run": str(run.relative_to(root))})
    env = os.environ.copy()
    env.update({"VLLM_PLUGINS": "ascend", "VLLM_BATCH_INVARIANT": args.batch_invariant,
                "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "VLLM_CACHE_ROOT": str(root / "cache" / args.arm)})
    if args.arm == "B1":
        env["VLLM_PLUGINS"] += ",operator_optimizations"
        env["VLLM_HUST_OPERATOR_OPTIMIZATIONS"] = "persistent-matmul-empty,linear-swiglu-graph" if args.optimization == "both" else args.optimization
        env["VLLM_CACHE_ROOT"] += "/" + args.optimization
    else:
        env.pop("VLLM_HUST_OPERATOR_OPTIMIZATIONS", None)
    argv = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
            "--served-model-name", "research-model", "--host", "127.0.0.1", "--port", str(args.port),
            "--dtype", "bfloat16", "--tensor-parallel-size", "2", "--max-model-len", "4096",
            "--max-num-seqs", "4", "--max-num-batched-tokens", "4096", "--block-size", "128",
            "--gpu-memory-utilization", "0.85", "--no-enable-prefix-caching",
            "--limit-mm-per-prompt", '{"image":0,"video":0}',
            "--compilation-config", '{"cudagraph_capture_sizes":[1,2,4]}']
    packages = {n: metadata.version(n) for n in ["torch", "torch-npu", "vllm", "vllm-ascend", "triton-ascend", "vllm-hust-opset"]}
    manifest = {"argv": argv, "packages": packages, "python": sys.version,
                "direct_urls": {n: json.loads(metadata.distribution(n).read_text("direct_url.json") or "null") for n in ["vllm", "vllm-ascend", "vllm-hust-opset"]},
                "runtime_origin": "server-version source snapshots; deployment probe records active import paths and fork commits",
                "environment": {k: env[k] for k in ["VLLM_PLUGINS", "VLLM_BATCH_INVARIANT", "TORCH_DEVICE_BACKEND_AUTOLOAD", "VLLM_CACHE_ROOT", "VLLM_HUST_OPERATOR_OPTIMIZATIONS"] if k in env}}
    for name in ["config.json", "tokenizer_config.json", "model.safetensors.index.json"]:
        path = Path(args.model) / name
        manifest[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(run / "manifest.json", manifest)
    service = None
    started = time.monotonic()
    try:
        with (run / "server.log").open("w") as log:
            service = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(run / "owned-process.json", {"pid": service.pid, "pgid": service.pid})
            deadline = started + 900
            base = f"http://127.0.0.1:{args.port}"
            while time.monotonic() < deadline:
                if service.poll() is not None:
                    raise RuntimeError(f"server exited during startup: {service.returncode}")
                tail = (run / "server.log").read_text(errors="replace")[-16000:]
                if "WorkerProc hit an exception" in tail or "Engine core initialization failed" in tail:
                    raise RuntimeError("worker startup failed; see server.log")
                try:
                    with urllib.request.urlopen(base + "/health", timeout=3) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(3)
            else:
                raise TimeoutError("900-second startup deadline exceeded")
            state["startup_seconds"] = time.monotonic() - started
            prompt = {"model": "research-model", "messages": [{"role": "user", "content": "请简要介绍 Transformer 的核心原理。"}],
                      "temperature": 0, "seed": 0, "max_tokens": 128,
                      "chat_template_kwargs": {"enable_thinking": False}}
            save(run / "request.json", prompt)
            req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(prompt).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=240) as response:
                output = json.load(response)
            save(run / "response.json", output)
            state["status"] = "completed-smoke"
            state["task_quality_verified"] = False
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        if service is not None:
            try:
                os.killpg(service.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                service.wait(timeout=25)
            except subprocess.TimeoutExpired:
                os.killpg(service.pid, signal.SIGKILL)
                service.wait(timeout=10)
            state["server_exit_code"] = service.returncode
        state["elapsed_seconds"] = time.monotonic() - started
        save(run / "result.json", state)
        save(root / "status.json", {**state, "run": str(run.relative_to(root))})
        print(json.dumps(state, ensure_ascii=False), flush=True)
    return 0 if state["status"] == "completed-smoke" else 1


if __name__ == "__main__":
    raise SystemExit(main())
