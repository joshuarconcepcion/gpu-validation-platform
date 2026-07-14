import time
import cupy as cp 

# helper function to avoid repeated logic for each workload function:
def _run_workload(workload_name: str, matrix_size: int, duration_seconds: int) -> dict:
    start = time.time()
    operations = 0

    # matrix multiplication logic:
    a = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32) # created outside loop to save time on every operation iteration
    b = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)

    while time.time() - start < duration_seconds:
        cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()  # wait for GPU to finish before next iteration, otherwise "queue" speed tested rather than compute speed
        operations += 1

    peak_memory_used_mb = cp.get_default_memory_pool().used_bytes() / 1024 / 1024 # returns used bytes in memory pool and converts to mb

    cp.get_default_memory_pool().free_all_blocks()  # release GPU memory after workload

    return {
        "workload_name": workload_name,
        "duration_seconds": duration_seconds,
        "operations_per_second": round(operations / duration_seconds, 2),
        "peak_memory_used_mb": round(peak_memory_used_mb, 2),
    }


def light_workload() -> dict:
    return _run_workload("light", matrix_size=1024, duration_seconds=10)


def medium_workload() -> dict:
    return _run_workload("medium", matrix_size=4096, duration_seconds=30)


def stress_workload() -> dict:
    return _run_workload("stress", matrix_size=8192, duration_seconds=60)
