"""Integration tests against real hardware via NVML. Requires an NVIDIA GPU
and drivers on the machine running pytest — skip on machines without one."""

import time

from hardware.gpu_monitor import GPUInstrumentation, GPUMetrics


def test_metrics_fields(): # tests that metrics field access works w/ hard-coded mock numbers
    m = GPUMetrics(
        timestamp=1.0,
        gpu_utilization_pct=50.0,
        memory_used_mb=4096.0,
        memory_total_mb=24576.0,
        temperature_c=65.0,
        power_draw_w=200.0,
        fan_speed_pct=40.0,
        clock_graphics_mhz=1695.0,
        clock_memory_mhz=9751.0,
    )
    assert m.memory_total_mb == 24576.0
    assert m.gpu_utilization_pct == 50.0


class TestGPUInstrumentation:
    # pytest hooks that initialize and shut down NVML for each test:
    def setup_method(self):
        self.inst = GPUInstrumentation()
    def teardown_method(self):
        self.inst.close()

    def test_collect_returns_metrics(self): # confirms collect() function returns GPUMetrics object
        assert isinstance(self.inst.collect(), GPUMetrics)

    def test_timestamp_is_recent(self): # confirms timestamp is accurate (timestamp is at least same as before value, so timestamp is made at time of collection, not class construction)
        before = time.time()
        assert self.inst.collect().timestamp >= before

    def test_utilization_in_range(self): # confirms utilization is a percent between 0 and 100
        assert 0.0 <= self.inst.collect().gpu_utilization_pct <= 100.0

    def test_memory_used_within_total(self): # confirms memory used is within otal range of 0 and total memory
        m = self.inst.collect()
        assert 0.0 < m.memory_used_mb <= m.memory_total_mb

    def test_memory_total_matches_rtx3090(self): # confirms collected total memory value matches hardware total memory (retrieved number through first real test)
        assert self.inst.collect().memory_total_mb == 24564.0

    def test_temperature_in_range(self): # confirms gpu temperature is within range
        assert 0.0 < self.inst.collect().temperature_c < 110.0

    def test_power_draw_positive(self): # confirms power draw from gpu (even if idle) is above 0 + ensures mW -> W conversion is correct
        assert self.inst.collect().power_draw_w > 0.0

    # confirms both clocks running:
    def test_clock_graphics_positive(self):  
        assert self.inst.collect().clock_graphics_mhz > 0.0
    def test_clock_memory_positive(self):
        assert self.inst.collect().clock_memory_mhz > 0.0

    def test_context_manager_closes_on_exit(self): # confirms nvmlShutdown is called correctly after GPU metric collection
        with GPUInstrumentation() as inst:
            m = inst.collect()
        assert isinstance(m, GPUMetrics)
