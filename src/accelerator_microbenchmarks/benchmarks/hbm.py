"""HBM bandwidth microbenchmarks supporting memory bandwidth kernels (Copy, Scale, Add, Triad)."""

import collections
import copy
import dataclasses
import math
import random
from typing import Any, Callable, Sequence
from accelerator_microbenchmarks.core import base
from accelerator_microbenchmarks.core import constants
from accelerator_microbenchmarks.core import registry
from accelerator_microbenchmarks.core import report
from accelerator_microbenchmarks.core import system
from accelerator_microbenchmarks.core import utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np


class HBMKernelOp(constants.ParamEnum):
  """Operation types supported by HBM bandwidth microbenchmarks."""

  COPY = "copy"
  SCALE = "scale"
  ADD = "add"
  TRIAD = "triad"
  READ_ONLY = "read_only"
  WRITE_ONLY = "write_only"
  READ = "read"  # Alias for READ_ONLY.
  WRITE = "write"  # Alias for WRITE_ONLY.


@dataclasses.dataclass(frozen=True)
class HBMKernelSpec:
  """Specification for an HBM benchmark kernel.

  Attributes:
    name: Name of the kernel operation (e.g. copy, scale, add, triad).
    kernel_fn: Callable implementing the computation (receives tuple of args and
      scalar).
    num_inputs: Number of input arrays required.
    num_arrays: Total number of array transfers (reads + writes).
    num_flops_per_element: Multiplier for FLOP calculation (e.g. 1.0 or 2.0 for
      triad).
  """

  name: str
  kernel_fn: Callable[..., Any]
  num_inputs: int
  num_arrays: int
  num_flops_per_element: float = 1.0


HBM_KERNELS: dict[HBMKernelOp, HBMKernelSpec] = {
    HBMKernelOp.COPY: HBMKernelSpec(
        name=HBMKernelOp.COPY.value,
        kernel_fn=lambda args, scalar: args[0] + 1.0,
        num_inputs=1,
        num_arrays=2,  # 1 read (x), 1 write (y)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.SCALE: HBMKernelSpec(
        name=HBMKernelOp.SCALE.value,
        kernel_fn=lambda args, scalar: args[0] * scalar,
        num_inputs=1,
        num_arrays=2,  # 1 read (x), 1 write (y)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.ADD: HBMKernelSpec(
        name=HBMKernelOp.ADD.value,
        kernel_fn=lambda args, scalar: args[0] + args[1],
        num_inputs=2,
        num_arrays=3,  # 2 reads (x, y), 1 write (z)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.TRIAD: HBMKernelSpec(
        name=HBMKernelOp.TRIAD.value,
        kernel_fn=lambda args, scalar: args[0] + (args[1] * scalar),
        num_inputs=2,
        num_arrays=3,  # 2 reads (x, y), 1 write (z)
        num_flops_per_element=2.0,
    ),
    HBMKernelOp.READ_ONLY: HBMKernelSpec(
        name=HBMKernelOp.READ_ONLY.value,
        kernel_fn=lambda args, scalar: jnp.any(args[0]),
        num_inputs=1,
        num_arrays=1,  # 1 read (x), 0 writes (scalar boolean return)
        num_flops_per_element=1.0,
    ),
    HBMKernelOp.WRITE_ONLY: HBMKernelSpec(
        name=HBMKernelOp.WRITE_ONLY.value,
        kernel_fn=lambda args, scalar, shape, dtype: jnp.full(
            shape, scalar, dtype=dtype
        ),
        num_inputs=0,
        num_arrays=1,  # 0 reads, 1 write (y)
        num_flops_per_element=0.0,
    ),
}
HBM_KERNELS[HBMKernelOp.READ] = HBM_KERNELS[HBMKernelOp.READ_ONLY]
HBM_KERNELS[HBMKernelOp.WRITE] = HBM_KERNELS[HBMKernelOp.WRITE_ONLY]


@dataclasses.dataclass
class HBMBandwidthParams(base.SingleDtypeBenchmarkParams):
  op_type: HBMKernelOp = dataclasses.field(
      default=HBMKernelOp.COPY,
      metadata={"help": "HBM kernel operation type."},
  )
  num_elements: int = dataclasses.field(
      default=134217728,
      metadata={"min": 1, "help": "Number of elements for HBM input arrays."},
  )
  device_id: int = dataclasses.field(
      default=0,
      metadata={"min": 0, "help": "Target local accelerator device ID."},
  )


@registry.benchmark_registry.register("hbm", aliases=["hbm_bandwidth"])
class HBMBandwidthBenchmark(base.BaseBenchmark[HBMBandwidthParams]):
  """HBM bandwidth microbenchmark supporting standard memory kernels."""

  Config = HBMBandwidthParams
  derive_chip_bandwidth: bool = False
  roofline_mode: constants.RooflineMode = constants.RooflineMode.MEMORY_HBM
  REPORT_SCHEMA: Sequence[tuple[str, Callable[[Any], str]]] = (
      ("dtype", report.format_str),
      ("op_type", report.format_str),
      ("process_index", report.format_str),
      ("global_device_id", report.format_str),
      ("device_id", report.format_str),
      ("num_elements", report.format_str),
      ("data_size_mib", report.format_2f),
      ("wall_clock_p50_ms", report.format_4f),
      ("wall_clock_bandwidth_per_device_gb_s", report.format_2f),
      ("wall_clock_bw_vs_median_pct", report.format_2f),
      ("xprof_p50_ms", report.format_4f),
      ("xprof_bandwidth_per_device_gb_s", report.format_2f),
      ("xprof_bw_vs_median_pct", report.format_2f),
  )

  _SYNC_METRIC_KEYS: tuple[str, ...] = (
      f"{constants.TimingDomain.WALL_CLOCK}_avg_ms",
      f"{constants.TimingDomain.WALL_CLOCK}_p50_ms",
      f"{constants.TimingDomain.WALL_CLOCK}_p90_ms",
      f"{constants.TimingDomain.WALL_CLOCK}_std_ms",
      f"{constants.TimingDomain.WALL_CLOCK}_bandwidth_per_device_gb_s",
      f"{constants.TimingDomain.WALL_CLOCK}_memory_roofline_efficiency_pct",
      f"{constants.TimingDomain.XPROF}_avg_ms",
      f"{constants.TimingDomain.XPROF}_p50_ms",
      f"{constants.TimingDomain.XPROF}_p90_ms",
      f"{constants.TimingDomain.XPROF}_std_ms",
      f"{constants.TimingDomain.XPROF}_bandwidth_per_device_gb_s",
      f"{constants.TimingDomain.XPROF}_memory_roofline_efficiency_pct",
      "total_duration_s",
      "actual_runs",
  )

  def __init__(
      self,
      config: HBMBandwidthParams,
      hardware_spec: system.HardwareSpec,
      mesh: jax.sharding.Mesh | None = None,
      xprof_config: base.XprofConfig | None = None,
  ):
    super().__init__(
        config=config,
        hardware_spec=hardware_spec,
        mesh=mesh,
        xprof_config=xprof_config,
    )
    self.spec: HBMKernelSpec | None = None
    self.scalar: Any | None = None

  def setup(self):
    num_devices = len(jax.local_devices())
    if not (0 <= self.config.device_id < num_devices):
      raise ValueError(
          f"Invalid device_id: {self.config.device_id}. Must be in range"
          f" [0, {num_devices - 1}] (found {num_devices} local device(s))."
      )

    op_type = self.config.op_type
    if op_type is None:
      raise ValueError("op_type must be specified.")

    if op_type in HBM_KERNELS:
      spec = HBM_KERNELS[op_type]
    else:
      raise ValueError(
          f"Unsupported op_type: '{op_type}'. Supported:"
          f" {HBMKernelOp.supported_options_str()}."
      )
    self.spec = spec

    dtype = utils.parse_dtype(self.config.dtype)
    self.scalar = jnp.array(random.uniform(1.1, 10.0), dtype=dtype)
    scalar = self.scalar

    def hbm_op(*args):
      with jax.named_scope(constants.MARKER):
        if spec.num_inputs == 0:
          return spec.kernel_fn(
              args, scalar, (self.config.num_elements,), dtype
          )
        return spec.kernel_fn(args, scalar)

    sharding = jax.sharding.SingleDeviceSharding(self.get_device_to_measure())
    self._jit_fn = jax.jit(hbm_op, out_shardings=sharding)

  def get_run_identifier(self) -> str:
    return (
        f"{self.config.op_type}_dim_{self.config.num_elements}"
        f"_dev_{self.config.device_id}"
    )

  def get_device_to_measure(self) -> jax.Device:
    return jax.local_devices()[self.config.device_id]

  def generate_inputs(self) -> tuple[jnp.ndarray, ...]:
    assert self.spec is not None
    # 'num_elements' being the number of elements
    num_elements = self.config.num_elements
    dtype = utils.parse_dtype(self.config.dtype)

    # Force execution on target local device for single-device benchmark
    device_to_measure = self.get_device_to_measure()
    sharding = jax.sharding.SingleDeviceSharding(device_to_measure)

    # Use jit with out_shardings to generate on device to avoid host OOM
    generate_data = jax.jit(
        lambda k: jax.random.normal(k, (num_elements,), dtype=dtype),
        out_shardings=sharding,
    )

    inputs = []
    for i in range(self.spec.num_inputs):
      key = jax.random.PRNGKey(i)
      inputs.append(generate_data(key))
    return tuple(inputs)

  def run_op(self, *args, **kwargs) -> jnp.ndarray:
    if self._jit_fn is None:
      raise ValueError("JIT function not initialized.")
    return self._jit_fn(*args, **kwargs)

  def get_total_bytes(self) -> float:
    assert self.spec is not None
    num_elements = self.config.num_elements
    dtype = utils.parse_dtype(self.config.dtype)
    itemsize = jnp.dtype(dtype).itemsize

    return float(num_elements * itemsize * self.spec.num_arrays)

  def get_arithmetic_intensity(self) -> float:
    assert self.spec is not None
    num_elements = self.config.num_elements
    flops = float(self.spec.num_flops_per_element * num_elements)
    bytes_moved = self.get_total_bytes()
    return flops / bytes_moved

  def get_workload_metadata(self) -> dict[str, Any]:
    assert self.spec is not None
    total_bytes = self.get_total_bytes()
    measured_device = self.get_device_to_measure()
    return {
        "data_size_mib": total_bytes / (1024 * 1024),
        "op_type": self.spec.name,
        "process_index": int(
            getattr(measured_device, "process_index", jax.process_index())
        ),
        "global_device_id": int(
            getattr(measured_device, "id", self.config.device_id)
        ),
    }

  def calculate_throughput_metrics(
      self, latency_ms: float, prefix: constants.TimingDomain
  ) -> dict[str, Any]:
    total_bytes = self.get_total_bytes()

    latency_s = latency_ms / 1000.0
    if latency_s == 0:
      bandwidth_gb_s = float("inf")
      bw_vs_median_pct = None
    else:
      bandwidth_gb_s = (total_bytes / latency_s) / 1e9
      bw_vs_median_pct = 0.0

    return {
        f"{prefix}_bandwidth_per_device_gb_s": bandwidth_gb_s,
        f"{prefix}_bw_vs_median_pct": bw_vs_median_pct,
    }

  @classmethod
  def _allgather_results_across_hosts(
       cls, results: Sequence[base.BenchmarkResult]
  ) -> list[base.BenchmarkResult]:
    """All-gathers HBM BenchmarkResult entries across all JAX processes."""
    if not results or jax.process_count() <= 1:
      return list(results)

    rows = []
    for res in results:
      params = res.metadata.params or {}
      metrics = res.metrics or {}
      proc_idx = metrics.get(
          "process_index", res.metadata.platform_info.process_index
      )
      device_id = params.get("device_id", 0)
      global_dev_id = metrics.get("global_device_id", device_id)
      row = [float(proc_idx), float(global_dev_id), float(device_id)]
      for key in cls._SYNC_METRIC_KEYS:
        val = metrics.get(key)
        row.append(
            float(val)
            if val is not None and not isinstance(val, bool)
            else float("nan")
        )
      rows.append(row)

    local_tensor = np.asarray(rows, dtype=np.float32)
    try:
      gathered = np.asarray(
          multihost_utils.process_allgather(local_tensor, tiled=False)
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      print(f"Warning: HBM multihost all-gather failed ({e}); using local results.")
      return list(results)

    local_proc_idx = jax.process_index()
    merged: list[base.BenchmarkResult] = []
    for proc_slot in range(gathered.shape[0]):
      for case_idx, template in enumerate(results):
        vec = gathered[proc_slot, case_idx]
        proc_idx = int(round(float(vec[0])))
        global_dev_id = int(round(float(vec[1])))
        device_id = int(round(float(vec[2])))

        if proc_idx == local_proc_idx and proc_slot == local_proc_idx:
          local_copy = copy.deepcopy(template)
          local_copy.metrics["process_index"] = proc_idx
          local_copy.metrics["global_device_id"] = global_dev_id
          merged.append(local_copy)
          continue

        new_params = dict(template.metadata.params or {})
        new_params["device_id"] = device_id
        new_platform_info = dataclasses.replace(
            template.metadata.platform_info,
            process_index=proc_idx,
        )
        new_metadata = dataclasses.replace(
            template.metadata,
            params=new_params,
            platform_info=new_platform_info,
        )
        new_metrics = dict(template.metrics or {})
        new_metrics["process_index"] = proc_idx
        new_metrics["global_device_id"] = global_dev_id

        for key_idx, key in enumerate(cls._SYNC_METRIC_KEYS):
          raw_val = float(vec[3 + key_idx])
          if math.isnan(raw_val):
            if key in new_metrics:
              new_metrics[key] = None
          elif key == "actual_runs":
            new_metrics[key] = int(round(raw_val))
          else:
            new_metrics[key] = raw_val

        merged.append(
            base.BenchmarkResult(
                metadata=new_metadata,
                metrics=new_metrics,
                raw_times_ms=list(template.raw_times_ms),
            )
        )
    return merged

  @classmethod
  def compute_median_bandwidth_comparison(
      cls, results: Sequence[base.BenchmarkResult]
  ) -> list[base.BenchmarkResult]:
    """Computes per-(dtype, op_type, num_elements) signed % delta vs median bandwidth."""
    if not results:
      return []

    enriched = [copy.deepcopy(r) for r in results]
    groups: dict[tuple[Any, Any, Any], list[base.BenchmarkResult]] = (
        collections.defaultdict(list)
    )
    for res in enriched:
      params = res.metadata.params or {}
      metrics = res.metrics or {}
      proc_idx = metrics.get(
          "process_index", res.metadata.platform_info.process_index
      )
      device_id = params.get("device_id", 0)
      res.metrics.setdefault("process_index", int(proc_idx))
      res.metrics.setdefault("global_device_id", int(device_id))

      group_key = (
          str(params.get("dtype", "")),
          str(metrics.get("op_type", params.get("op_type", ""))),
          params.get("num_elements"),
      )
      groups[group_key].append(res)

    for group_results in groups.values():
      for prefix in (
          constants.TimingDomain.WALL_CLOCK,
          constants.TimingDomain.XPROF,
      ):
        bw_key = f"{prefix}_bandwidth_per_device_gb_s"
        cmp_key = f"{prefix}_bw_vs_median_pct"

        valid_bws = [
            float(r.metrics[bw_key])
            for r in group_results
            if r.metrics.get(bw_key) is not None
            and not isinstance(r.metrics.get(bw_key), bool)
            and math.isfinite(float(r.metrics[bw_key]))
        ]
        median_bw = float(np.median(valid_bws)) if valid_bws else None

        for r in group_results:
          if (
              prefix == constants.TimingDomain.XPROF
              and bw_key not in r.metrics
              and cmp_key not in r.metrics
          ):
            continue
          bw_val = r.metrics.get(bw_key)
          if (
              bw_val is not None
              and not isinstance(bw_val, bool)
              and math.isfinite(float(bw_val))
              and median_bw is not None
              and median_bw > 0
          ):
            r.metrics[cmp_key] = (
                (float(bw_val) - median_bw) / median_bw
            ) * 100.0
          else:
            r.metrics[cmp_key] = None

    return enriched

  @classmethod
  def finalize_results(
      cls, results: Sequence[base.BenchmarkResult]
  ) -> list[base.BenchmarkResult]:
    """All-gathers HBM results across hosts and computes median bandwidth deltas."""
    gathered = cls._allgather_results_across_hosts(results)
    return cls.compute_median_bandwidth_comparison(gathered)
