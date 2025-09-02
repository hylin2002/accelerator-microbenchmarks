"""A script to run the microbenchmarks in Jax over DCN and ICI collectives."""

# pylint: disable=g-importing-member
from functools import partial
from typing import Any, Callable, Dict, Tuple

from benchmark_utils import simple_timeit, MetricsStatistics
import jax
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

# pylint: disable=g-importing-member


def create_mesh(dcn_size: int, ici_size: int) -> tuple[Mesh, list[int], list[int]]:
    """Creates a hybrid mesh with the given DCN and ICI sizes."""
    dcn_parallelism = [dcn_size, 1]
    ici_parallelism = [1, ici_size]

    total_devices = jax.device_count()
    if total_devices != (dcn_size * ici_size):
        raise ValueError(
            f"Need {dcn_size * ici_size} devices, but found {total_devices}"
        )
    if dcn_size > 1:
        mesh_devices = mesh_utils.create_hybrid_device_mesh(
            ici_parallelism, dcn_parallelism, devices=jax.devices()
        )
        mesh = Mesh(mesh_devices, ("dcn", "ici"))
    else:
        mesh_devices = mesh_utils.create_device_mesh([ici_size], devices=jax.devices())
        mesh = Mesh(mesh_devices, "ici")
    return mesh, dcn_parallelism, ici_parallelism


def get_metrics_helper(
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Helper function to build the metrics and metadata for the benchmark."""
    exclude_keys = ["ici_average_time_ms_list", "dcn_average_time_ms_list"]
    metadata = {
        key: value
        for key, value in params
        if value is not None and key not in exclude_keys
    }
    metadata["dtype"] = metadata["dtype"].dtype.itemsize
    return metadata, {}


def _run_collective(
    mesh: Mesh,
    matrix: jnp.ndarray,
    in_spec: P,
    out_spec: P,
    collective_op: Callable[[Any], Any],
    num_runs: int,
    matrix_dim: int,
    task_name: str,
    trace_dir: str | None,
    check_rep: bool = True,
) -> list[float] | None:
    """Helper to run a single collective benchmark."""
    sharded_matrix = jax.device_put(matrix, jax.sharding.NamedSharding(mesh, in_spec))

    @partial(
        shard_map, mesh=mesh, in_specs=in_spec, out_specs=out_spec, check_rep=check_rep
    )
    def f(x):
        return collective_op(x)

    jitted_op = jax.jit(f)

    return simple_timeit(
        jitted_op,
        sharded_matrix,
        matrix_dim=matrix_dim,
        tries=num_runs,
        task=task_name,
        trace_dir=trace_dir,
    )


def _calculate_collective_metrics(
    time_ms_list: list[float] | None,
    matrix_size_gbyte: float,
    num_devices: int,
    bandwidth_formula: Callable[[float, float, int], float],
    network_type: str,
    benchmark_name: str,
    matrix_dim: int,
    dtype: jnp.dtype,
) -> Dict[str, Any]:
    """Helper to calculate metrics for a single collective benchmark."""
    if time_ms_list is None:
        return {}

    bandwidth_gbyte_s_list = [
        bandwidth_formula(matrix_size_gbyte, time_ms, num_devices)
        for time_ms in time_ms_list
    ]

    bandwidth_gbyte_s_statistics = MetricsStatistics(
        metrics_list=bandwidth_gbyte_s_list,
        metrics_name=f"{network_type}_bandwidth_gbyte_s",
    )
    print(
        f"{benchmark_name}_{network_type}: Matrix size: {matrix_dim}x{matrix_dim}, {dtype=}, "
        f"{matrix_size_gbyte=}, achieved_bandwidth_gbyte_s (median) = {bandwidth_gbyte_s_statistics.statistics['p50']}"
    )
    return bandwidth_gbyte_s_statistics.serialize_statistics()


def psum_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the psum collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
        benchmark is run.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run. The ICI and DCN

    Returns:
      The measured time for the DCN and ICI benchmarks.
    """
    mesh, _, _ = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    dcn_average_time_ms_list = ici_average_time_ms_list = None
    # DCN benchmark
    if dcn_size > 1:
        dcn_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P("dcn", None),
            P(None),
            partial(jax.lax.psum, axis_name="dcn"),
            num_runs,
            matrix_dim,
            "psum_dcn_op",
            trace_dir,
        )

    # ICI benchmark
    if ici_size > 1:
        ici_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P(None, None),
            P(None, None),
            partial(jax.lax.psum, axis_name="ici"),
            num_runs,
            matrix_dim,
            "psum_ici_op",
            trace_dir,
        )
    return {
        "dcn_average_time_ms_list": dcn_average_time_ms_list,
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def psum_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms_list: list[float],
    dcn_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata, metrics = get_metrics_helper(params)
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                dcn_average_time_ms_list,
                matrix_size_gbyte,
                dcn_size,
                # bandwidth is claculated as psum can be done via reduce_scatter +
                # all_gather so bandwidth is the sum of the two (formulas below)
                lambda size, time, n: size * (n - 1) * 2 / n / n / (time / 1e3),
                "dcn",
                "psum",
                matrix_dim,
                dtype,
            )
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                ici_average_time_ms_list,
                matrix_size_gbyte,
                ici_size,
                lambda size, time, n: size * (n - 1) * 2 / n / (time / 1e3),
                "ici",
                "psum",
                matrix_dim,
                dtype,
            )
        )
    return metadata, metrics


def psum_scatter_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the psum_scatter collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
        benchmark is run.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run. The ICI and DCN

    Returns:
      The measured time for the DCN and ICI benchmarks.
    """
    mesh, _, _ = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    dcn_average_time_ms_list = ici_average_time_ms_list = None
    # DCN benchmark
    if dcn_size > 1:
        dcn_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P("dcn", None),
            P("dcn", None),
            partial(jax.lax.psum_scatter, axis_name="dcn", tiled=True),
            num_runs,
            matrix_dim,
            "psum_scatter_dcn_op",
            trace_dir,
        )

    # ICI benchmark
    if ici_size > 1:
        ici_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P(None, None),
            P(None, "ici"),
            partial(jax.lax.psum_scatter, axis_name="ici", tiled=True),
            num_runs,
            matrix_dim,
            "psum_scatter_ici_op",
            trace_dir,
        )
    return {
        "dcn_average_time_ms_list": dcn_average_time_ms_list,
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def psum_scatter_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms_list: list[float],
    dcn_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the psum_scatter benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata, metrics = get_metrics_helper(params)
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                dcn_average_time_ms_list,
                matrix_size_gbyte,
                dcn_size,
                # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
                # to use (dcn_size - 1) steps in a ring algorithm
                lambda size, time, n: size * (n - 1) / n / n / (time / 1e3),
                "dcn",
                "psum_scatter",
                matrix_dim,
                dtype,
            )
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                ici_average_time_ms_list,
                matrix_size_gbyte,
                ici_size,
                lambda size, time, n: size * (n - 1) / n / (time / 1e3),
                "ici",
                "psum_scatter",
                matrix_dim,
                dtype,
            )
        )
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def all_gather_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the all_gather collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
        benchmark is run.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run. The ICI and DCN

    Returns:
      The measured time for the DCN and ICI benchmarks.
    """
    mesh, _, _ = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    dcn_average_time_ms_list = ici_average_time_ms_list = None

    # DCN benchmark
    if dcn_size > 1:
        dcn_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P("dcn", None),
            P(None, None),
            partial(jax.lax.all_gather, axis_name="dcn", tiled=True),
            num_runs,
            matrix_dim,
            "all_gather_dcn_op",
            trace_dir,
            check_rep=False,
        )

    # ICI benchmark
    if ici_size > 1:
        ici_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P("ici", None),
            P(None, None),
            partial(jax.lax.all_gather, axis_name="ici", tiled=True),
            num_runs,
            matrix_dim,
            "all_gather_ici_op",
            trace_dir,
            check_rep=False,
        )

    return {
        "dcn_average_time_ms_list": dcn_average_time_ms_list,
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def all_gather_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms_list: list[float],
    dcn_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the all_gather benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata, metrics = get_metrics_helper(params)
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                dcn_average_time_ms_list,
                matrix_size_gbyte,
                dcn_size,
                # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
                # to use (dcn_size - 1) steps in a ring algorithm
                lambda size, time, n: size * (n - 1) / n / (time / 1e3),
                "dcn",
                "all_gather",
                matrix_dim,
                dtype,
            )
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                ici_average_time_ms_list,
                matrix_size_gbyte,
                ici_size,
                lambda size, time, n: size * (n - 1) / n / (time / 1e3),
                "ici",
                "all_gather",
                matrix_dim,
                dtype,
            )
        )
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics


def ppermute_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the ppermute collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
        benchmark is run.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run. The ICI and DCN

    Returns:
      The measured time for the DCN and ICI benchmarks.
    """
    mesh, _, _ = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    dcn_average_time_ms_list = ici_average_time_ms_list = None

    # DCN benchmark
    if dcn_size > 1:
        perm = [(i, (i + 1) % dcn_size) for i in range(dcn_size)]
        dcn_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P("dcn", None),
            P("dcn", None),
            partial(jax.lax.ppermute, axis_name="dcn", perm=perm),
            num_runs,
            matrix_dim,
            "ppermute_dcn_op",
            trace_dir,
        )

    # ICI benchmark
    if ici_size > 1:
        perm = [(i, (i + 1) % ici_size) for i in range(ici_size)]
        ici_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P(None, None),
            P(None, "ici"),
            partial(jax.lax.ppermute, axis_name="ici", perm=perm),
            num_runs,
            matrix_dim,
            "ppermute_ici_op",
            trace_dir,
        )
    return {
        "dcn_average_time_ms_list": dcn_average_time_ms_list,
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def ppermute_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms_list: list[float],
    dcn_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the ppermute benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata, metrics = get_metrics_helper(params)
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                dcn_average_time_ms_list,
                matrix_size_gbyte,
                dcn_size,
                # each sharded matrix size is matrix_size_gbyte / dcn_size and then it needs
                # to use 1 step
                lambda size, time, n: size / n / (time / 1e3),
                "dcn",
                "ppermute",
                matrix_dim,
                dtype,
            )
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                ici_average_time_ms_list,
                matrix_size_gbyte,
                ici_size,
                lambda size, time, n: size / (time / 1e3),
                "ici",
                "ppermute",
                matrix_dim,
                dtype,
            )
        )
    return metadata, metrics


def all_to_all_benchmark(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    num_runs: int = 1,
    trace_dir: str = None,
) -> Dict[str, Any]:
    """Benchmarks the all_to_all collective operation.

    Args:
      matrix_dim: The benchmark is run on a matrix with shape (matrix_dim,
        matrix_dim).
      dtype: The data type of the matrix.
      dcn_size: The number of DCN nodes, or number of slices. If 1, then no DCN
        benchmark is run.
      ici_size: The number of chips in a single slice. If 1, then no ICI benchmark
        is run. The ICI and DCN

    Returns:
      The measured time for the DCN and ICI benchmarks.
    """
    mesh, _, _ = create_mesh(dcn_size, ici_size)
    matrix = jnp.ones((matrix_dim, matrix_dim), dtype=dtype)
    dcn_average_time_ms_list = ici_average_time_ms_list = None

    # DCN benchmark
    if dcn_size > 1:
        dcn_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P("dcn", None),
            P("dcn", None),
            partial(
                jax.lax.all_to_all,
                axis_name="dcn",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            ),
            num_runs,
            matrix_dim,
            "all_to_all_dcn_op",
            trace_dir,
        )

    # ICI benchmark
    if ici_size > 1:
        ici_average_time_ms_list = _run_collective(
            mesh,
            matrix,
            P(None, None),
            P(None, None),
            partial(
                jax.lax.all_to_all,
                axis_name="ici",
                split_axis=0,
                concat_axis=0,
                tiled=True,
            ),
            num_runs,
            matrix_dim,
            "all_to_all_ici_op",
            trace_dir,
            check_rep=False,
        )

    return {
        "dcn_average_time_ms_list": dcn_average_time_ms_list,
        "ici_average_time_ms_list": ici_average_time_ms_list,
    }


def all_to_all_benchmark_calculate_metrics(
    matrix_dim: int,
    dtype: jnp.dtype,
    dcn_size: int,
    ici_size: int,
    ici_average_time_ms_list: list[float],
    dcn_average_time_ms_list: list[float],
) -> Dict[str, Any]:
    """Calculates the metrics for the all_to_all benchmark."""
    # Build dictionary of all the parameters in the function
    params = locals().items()
    metadata, metrics = get_metrics_helper(params)
    matrix_size_gbyte = matrix_dim * matrix_dim * dtype.dtype.itemsize / 1e9
    # Calculate metrics for DCN benchmark
    if dcn_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                dcn_average_time_ms_list,
                matrix_size_gbyte,
                dcn_size,
                lambda size, time, n: size * (n - 1) / n / n / (time / 1e3),
                "dcn",
                "all_to_all",
                matrix_dim,
                dtype,
            )
        )

    # Calculate metrics for ICI benchmark
    if ici_size > 1:
        metrics.update(
            _calculate_collective_metrics(
                ici_average_time_ms_list,
                matrix_size_gbyte,
                ici_size,
                lambda size, time, n: size * (n - 1) / n / (time / 1e3),
                "ici",
                "all_to_all",
                matrix_dim,
                dtype,
            )
        )
    metrics = {key: value for key, value in metrics.items() if value is not None}
    return metadata, metrics
