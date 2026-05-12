"""
《猫语心声》 —— 本地LLM推理服务

技术策划案v2 §4.5 的完整实现：
- llama-cpp-python 加载 DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF
- 支持进程内推理 + HTTP Server 双模式
- 异步生成、超时控制、健康检查
- 模型: DeepSeek-R1-Distill-Qwen-1.5B, 量化: Q4_0 (~1.0GB)

启动命令（llama.cpp server模式，备选）:
  llama-server -m "llm_integration/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF/deepseek-r1-distill-qwen-1.5b-q4_0.gguf" \
      --host 127.0.0.1 --port 8080 -c 2048 -t 8 -b 512
"""

from __future__ import annotations
import os
import sys
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

# 添加项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import numpy as np

from .config import (
    LLMConfig, MODEL_PATH, MODEL_DIR, MODEL_FILENAME,
    LLAMA_CPP_N_CTX, LLAMA_CPP_N_THREADS, LLAMA_CPP_N_BATCH,
    LLAMA_CPP_MAX_TOKENS, LLAMA_CPP_TEMPERATURE, LLAMA_CPP_TOP_P,
    LLAMA_CPP_TOP_K, LLAMA_CPP_REPEAT_PENALTY, LLAMA_CPP_STOP,
    LLM_TIMEOUT_MS, MAX_CONSECUTIVE_FAILURES, LLM_HEALTH_CHECK_INTERVAL,
)


@dataclass
class LLMResponse:
    """LLM推理响应"""
    text: str
    tokens_generated: int
    latency_ms: float
    finish_reason: str = "stop"  # stop | length | timeout | error
    from_cache: bool = False


class LLMService:
    """
    本地LLM推理服务。

    两种工作模式：
    1. 进程内模式（默认）：llama-cpp-python 直接加载模型推理
    2. HTTP Server模式：连接 llama.cpp server 的 HTTP API

    特性：
    - 异步生成（线程池）
    - 超时控制（默认150ms）
    - 健康检查
    - 生成队列 + 并发控制
    """

    def __init__(self, config: LLMConfig = None):
        self.cfg = config or LLMConfig()
        self._model = None
        self._model_loaded = False
        self._use_server = self.cfg.use_server
        self._healthy = False
        self._consecutive_failures = 0
        self._total_requests = 0
        self._total_success = 0
        self._total_timeout = 0
        self._total_error = 0
        self._latencies_ms: List[float] = []

        # 并发控制
        self._active_generations = 0
        self._gen_queue = queue.Queue(maxsize=self.cfg.queue_size)
        self._gen_lock = threading.Lock()

        # 统计
        self._lock = threading.Lock()

    # ═══════════════════════════════════════════
    #  模型加载
    # ═══════════════════════════════════════════

    def load_model(self) -> bool:
        """
        加载量化模型到内存。

        返回: 是否加载成功
        """
        if self._model_loaded:
            return True

        if self._use_server:
            print(f"[LLMService] 使用HTTP Server模式: "
                  f"{self.cfg.server_host}:{self.cfg.server_port}")
            self._model_loaded = True
            self._healthy = self._check_server_health()
            return self._healthy

        # 进程内模式
        if not os.path.exists(self.cfg.model_path):
            print(f"[LLMService] 模型文件不存在: {self.cfg.model_path}")
            return False

        try:
            from llama_cpp import Llama

            print(f"[LLMService] 加载模型: {self.cfg.model_filename}")
            print(f"  量化: {self.cfg.quantization}, 预计内存: "
                  f"{self.cfg.estimated_memory_gb}GB")
            print(f"  ctx={self.cfg.n_ctx}, threads={self.cfg.n_threads}, "
                  f"batch={self.cfg.n_batch}")

            self._model = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_threads=self.cfg.n_threads,
                n_batch=self.cfg.n_batch,
                verbose=False,
            )

            self._model_loaded = True
            self._healthy = True
            print(f"[LLMService] 模型加载完成")
            return True

        except ImportError:
            print("[LLMService] llama-cpp-python 未安装。请运行:")
            print("  pip install llama-cpp-python")
            return False
        except Exception as e:
            print(f"[LLMService] 模型加载失败: {e}")
            return False

    def unload_model(self):
        """卸载模型释放内存"""
        if self._model is not None:
            del self._model
            self._model = None
        self._model_loaded = False
        self._healthy = False

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    # ═══════════════════════════════════════════
    #  推理接口
    # ═══════════════════════════════════════════

    def generate(self, prompt: str,
                 max_tokens: int = None,
                 temperature: float = None,
                 top_p: float = None,
                 top_k: int = None,
                 repeat_penalty: float = None,
                 stop: List[str] = None,
                 timeout_ms: float = None,
                 ) -> LLMResponse:
        """
        同步生成文本（带超时）。

        参数:
            prompt: 完整提示词（含chat template）
            max_tokens: 最大生成token数（默认64）
            temperature: 温度（默认0.7）
            timeout_ms: 超时阈值ms（默认150ms）

        返回: LLMResponse
        """
        max_tokens = max_tokens or self.cfg.max_tokens
        temperature = temperature or self.cfg.temperature
        top_p = top_p or self.cfg.top_p
        top_k = top_k or self.cfg.top_k
        repeat_penalty = repeat_penalty or self.cfg.repeat_penalty
        stop = stop or self.cfg.stop_sequences
        timeout_ms = timeout_ms or self.cfg.llm_timeout_ms

        self._total_requests += 1

        t_start = time.perf_counter()

        try:
            if self._use_server:
                raw_text, finish_reason = self._generate_via_server(
                    prompt, max_tokens, temperature, top_p, stop, timeout_ms,
                )
            elif self._model_loaded:
                raw_text, finish_reason = self._generate_in_process(
                    prompt, max_tokens, temperature, top_p, top_k,
                    repeat_penalty, stop, timeout_ms,
                )
            else:
                return LLMResponse(
                    text="", tokens_generated=0,
                    latency_ms=0, finish_reason="error",
                )

            latency = (time.perf_counter() - t_start) * 1000

            with self._lock:
                self._total_success += 1
                self._consecutive_failures = 0
                self._latencies_ms.append(latency)
                if len(self._latencies_ms) > 1000:
                    self._latencies_ms = self._latencies_ms[-500:]

            return LLMResponse(
                text=raw_text.strip(),
                tokens_generated=len(raw_text),  # 粗略估计
                latency_ms=round(latency, 1),
                finish_reason=finish_reason,
            )

        except TimeoutError:
            with self._lock:
                self._total_timeout += 1
                self._consecutive_failures += 1
            return LLMResponse(
                text="", tokens_generated=0,
                latency_ms=timeout_ms,
                finish_reason="timeout",
            )
        except Exception as e:
            with self._lock:
                self._total_error += 1
                self._consecutive_failures += 1
            return LLMResponse(
                text="", tokens_generated=0,
                latency_ms=(time.perf_counter() - t_start) * 1000,
                finish_reason=f"error:{str(e)[:50]}",
            )

    def generate_async(self, prompt: str,
                       callback: Callable[[LLMResponse], None],
                       **kwargs):
        """
        异步生成（在后台线程执行，完成后回调）。

        用于不阻塞游戏主循环的场景。
        """
        def _worker():
            result = self.generate(prompt, **kwargs)
            try:
                callback(result)
            except Exception as e:
                print(f"[LLMService] 回调异常: {e}")

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        return t

    def _generate_in_process(self, prompt: str, max_tokens: int,
                              temperature: float, top_p: float, top_k: int,
                              repeat_penalty: float, stop: List[str],
                              timeout_ms: float) -> Tuple[str, str]:
        """进程内llama-cpp-python推理"""
        result = {}
        exception = None

        def _run():
            try:
                output = self._model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repeat_penalty=repeat_penalty,
                    stop=stop,
                    echo=False,
                    stream=False,
                )
                result["text"] = output["choices"][0]["text"]
                result["finish_reason"] = output["choices"][0].get(
                    "finish_reason", "stop"
                )
            except Exception as e:
                nonlocal exception
                exception = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=timeout_ms / 1000.0)

        if t.is_alive():
            raise TimeoutError(f"LLM推理超时 ({timeout_ms}ms)")

        if exception:
            raise exception

        return result.get("text", ""), result.get("finish_reason", "stop")

    def _generate_via_server(self, prompt: str, max_tokens: int,
                              temperature: float, top_p: float,
                              stop: List[str], timeout_ms: float
                              ) -> Tuple[str, str]:
        """通过HTTP API调用llama.cpp server"""
        import urllib.request
        import json

        url = f"http://{self.cfg.server_host}:{self.cfg.server_port}/completion"
        body = json.dumps({
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(url, data=body, headers={
            "Content-Type": "application/json",
        })

        try:
            with urllib.request.urlopen(req, timeout=timeout_ms / 1000.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                text = data.get("content", "")
                reason = data.get("stop_type", "stop")
                return text, reason
        except Exception as e:
            raise RuntimeError(f"HTTP Server调用失败: {e}")

    # ═══════════════════════════════════════════
    #  健康检查
    # ═══════════════════════════════════════════

    def health_check(self) -> bool:
        """检查LLM服务是否健康可用"""
        if self._use_server:
            self._healthy = self._check_server_health()
        else:
            if not self._model_loaded:
                self._healthy = False
            elif self._consecutive_failures >= self.cfg.max_consecutive_failures:
                self._healthy = False
            else:
                self._healthy = True
        return self._healthy

    def _check_server_health(self) -> bool:
        """检查HTTP Server健康状态"""
        try:
            import urllib.request
            url = (f"http://{self.cfg.server_host}:"
                   f"{self.cfg.server_port}/health")
            with urllib.request.urlopen(url, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def should_fallback(self) -> bool:
        """判断是否应降级到模板库"""
        if not self._model_loaded and not self._use_server:
            return True
        if self._consecutive_failures >= self.cfg.max_consecutive_failures:
            return True
        if not self.health_check():
            return True
        return False

    # ═══════════════════════════════════════════
    #  统计
    # ═══════════════════════════════════════════

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def stats(self) -> Dict:
        with self._lock:
            latencies = self._latencies_ms.copy()

        return {
            "model": self.cfg.model_filename,
            "quantization": self.cfg.quantization,
            "model_loaded": self._model_loaded,
            "healthy": self._healthy,
            "use_server": self._use_server,
            "total_requests": self._total_requests,
            "total_success": self._total_success,
            "total_timeout": self._total_timeout,
            "total_error": self._total_error,
            "consecutive_failures": self._consecutive_failures,
            "success_rate": (
                self._total_success / max(1, self._total_requests)
            ),
            "avg_latency_ms": (
                sum(latencies) / len(latencies) if latencies else 0
            ),
            "p50_latency_ms": (
                sorted(latencies)[len(latencies) // 2] if latencies else 0
            ),
            "p95_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.95)]
                if len(latencies) >= 20 else 0
            ),
            "p99_latency_ms": (
                sorted(latencies)[int(len(latencies) * 0.99)]
                if len(latencies) >= 100 else 0
            ),
        }


# ═══════════════════════════════════════════
#  启动指南（供部署参考）
# ═══════════════════════════════════════════

DEPLOYMENT_GUIDE = """
═══════════════════════════════════════════
  本地LLM推理服务部署指南
═══════════════════════════════════════════

【模型信息】
  模型名称: DeepSeek-R1-Distill-Qwen-1.5B
  量化格式: Q4_0 (4-bit)
  文件大小: ~1.0GB
  文件路径: llm_integration/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF/
  文件名: deepseek-r1-distill-qwen-1.5b-q4_0.gguf

【方式一：Python进程内推理（推荐开发用）】
  1. 安装依赖:
     pip install llama-cpp-python

  2. 在代码中直接使用:
     from llm_integration.llm_service import LLMService
     from llm_integration.config import LLMConfig

     cfg = LLMConfig()
     service = LLMService(cfg)
     service.load_model()

     response = service.generate("你好")
     print(response.text)

【方式二：llama.cpp HTTP Server（推荐生产用）】
  1. 编译或下载 llama.cpp (https://github.com/ggerganov/llama.cpp)

  2. 启动服务器:
     llama-server \\
       -m "llm_integration/DeepSeek-R1-Distill-Qwen-1.5B-Q4_0-GGUF/deepseek-r1-distill-qwen-1.5b-q4_0.gguf" \\
       --host 127.0.0.1 \\
       --port 8080 \\
       -c 2048 \\
       -t 8 \\
       -b 512

  3. 配置使用Server模式:
     cfg.use_server = True
     cfg.server_host = "127.0.0.1"
     cfg.server_port = 8080

【推理参数说明】
  n_ctx (上下文窗口): 2048 tokens
    - 心声生成仅需~500 tokens，2048 留足余量
  n_threads (CPU线程): 8
    - 根据CPU核心数调整，一般设为核心数
  n_batch (批处理): 512
    - prompt并行处理token数，越大越快但吃内存
  max_tokens (最大生成): 64 tokens
    - 心声约20中文字，64 tokens 足够
  temperature: 0.7
    - 控制创造力，0.7 适中
  top_p: 0.9, top_k: 40
    - nucleus sampling参数
  repeat_penalty: 1.1
    - 轻微惩罚重复

【性能预估（CPU, 8线程）】
  模型加载时间: ~3-5秒
  单次推理延迟: 50-150ms（取决于prompt长度）
  内存占用: ~1.0GB
  支持并发: 建议≤3个同时请求

【故障排查】
  - 模型加载失败: 检查文件路径、GGUF文件完整性
  - 推理超时: 调大LLM_TIMEOUT_MS或减少n_ctx
  - 内存不足: 确认系统有≥2GB可用内存
  - 生成乱码: 检查prompt格式是否符合chat template
═══════════════════════════════════════════
"""
