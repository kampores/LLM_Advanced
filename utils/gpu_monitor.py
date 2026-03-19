"""GPU 메모리 모니터링 유틸리티"""

import torch
import gc


def print_gpu_memory(tag=""):
    """현재 GPU 메모리 사용량을 출력합니다."""
    if not torch.cuda.is_available():
        print(f"[{tag}] GPU를 사용할 수 없습니다.")
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"[{tag}] GPU 메모리: {allocated:.1f}GB 사용 / {total:.1f}GB 전체 (예약: {reserved:.1f}GB)")


def clear_gpu_memory():
    """GPU 메모리를 정리합니다."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 메모리 정리 완료. 현재: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
