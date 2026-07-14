import time
import pynvml # python nvidia management library
from dataclasses import dataclass 


@dataclass
class GPUMetrics: # holds "snapshot" of gpu metric data + used dataclass instead of dict for types (safety when passing GPUMetrics to different files)
    timestamp: float # captures current time of data snapshot
    gpu_utilization_pct: float # gets percent of current gpu utilization
    memory_used_mb: float # used memory
    memory_total_mb: float # total memory
    temperature_c: float # temperature of gpu core in celsius
    power_draw_w: float 
    fan_speed_pct: float 
    clock_graphics_mhz: float # measures speed of running shader cores
    clock_memory_mhz: float # measures speed of VRAM (GDDR6X for 3090TI)


class GPUInstrumentation:
    def __init__(self, device_index: int = 0):
        pynvml.nvmlInit() # loads nvml library and initalizes connection to driver
        self._gpu = pynvml.nvmlDeviceGetHandleByIndex(device_index) # using opaque token to hold gpu (device_index = 0 to target first gpu)

    def collect(self) -> GPUMetrics:
        util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu) # returns gpu and memory utilization but only using gpu here
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu)
        temp = pynvml.nvmlDeviceGetTemperature(self._gpu, pynvml.NVML_TEMPERATURE_GPU) # second parameter targets gpu core temperature specifically
        power = pynvml.nvmlDeviceGetPowerUsage(self._gpu) / 1000.0 # given in milliwatts, divided to return watts
        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(self._gpu) # some gpu fans not exposed through nvml
        except Exception:
            fan = 0.0 # prevents crashing if fan not detected
        # returns clock speed as MHz:
        clock_gr = pynvml.nvmlDeviceGetClockInfo(self._gpu, pynvml.NVML_CLOCK_GRAPHICS) 
        clock_mem = pynvml.nvmlDeviceGetClockInfo(self._gpu, pynvml.NVML_CLOCK_MEM)
        return GPUMetrics(
            timestamp=time.time(),
            gpu_utilization_pct=float(util.gpu),
            memory_used_mb=mem.used / 1024 / 1024, # given as bytes, divided by 1024 twice to get mb (b -> kb -> mb)
            memory_total_mb=mem.total / 1024 / 1024,
            temperature_c=float(temp),
            power_draw_w=round(power, 1),
            fan_speed_pct=float(fan),
            clock_graphics_mhz=float(clock_gr),
            clock_memory_mhz=float(clock_mem),
        )

    def close(self):
        pynvml.nvmlShutdown() # releases connection to driver

    # makes sure nvmlShutdown is always called after block ends using with statement in test_instrumentation.py:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
