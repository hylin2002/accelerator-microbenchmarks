# HBM Microbnechmarks on tpu7x-8

This guide provides instructions for running High Bandwidth Memory (HBM) microbenchmarks on tpu7x-8 Google Kubernetes Engine (GKE) clusters. It covers creating a node pool, running the benchmarks, and viewing the expected output.

## Create Node Pools

Below is the command to create a TPU node-pool containing 4 ironwood chips:
```bash
# The default GKE version may be lower than `1.34.0-gke.2201000`. Check if `node-version` should be specified before applying this command.
gcloud container node-pools create ${SINGLE_SLICE_NODE_POOL_NAME} \
    --cluster=${CLUSTER_NAME} \
    --machine-type=tpu7x-standard-4t \
    --location=${LOCATION} \
    --node-locations=${LOCATION} \
    --project=${PROJECT_ID} \
    --num-nodes=1 \
    --reservation=${RESERVATION_NAME} \
    --reservation-affinity=specific
```

## Run HBM Microbenchmarks

To run the HBM microbenchmarks, apply the following Kubernetes configuration:
```bash
kubectl apply -f tpu7x-2x2x1-hbm-microbenchmark.yaml
```

To extract the log of HBM microbenchmark, use `kubectl log`:
```bash
kubectl log tpu7x-single-host-microbenchmark
```

To retrieve the complete results, including the trace and TSV output files, you must keep the pod running after the benchmark completes. To do this, add a `sleep` command to the `tpu7x-2x2x1-hbm-microbenchmark.yaml` file. You can then use `kubectl cp` to copy the output from the pod.

```bash
kubectl cp tpu7x-single-host-microbenchmark:/microbenchmarks/hbm hbm
```

## Expected bandwidth for different matrix size


| Matrix Size (Bytes) | Bandwidth (GB/s/core) | Bandwidth (GB/s/chip) |
|---------------------|-----------------------|-----------------------|
|             2097152 |           1379.335021 |           2758.670041 |
|             4194304 |           2249.746091 |           4499.492181 |
|             8388608 |           2246.129937 |           4492.259875 |
|            16777216 |           2757.308985 |            5514.61797 |
|            33554432 |            3009.83593 |            6019.67186 |
|            67108864 |           3097.217778 |           6194.435556 |
|           134217728 |            3176.50274 |           6353.005481 |
|           268435456 |           3167.144485 |           6334.288969 |
|           536870912 |           3199.020504 |           6398.041009 |
|          1073741824 |           3198.414211 |           6396.828421 |
|          2147483648 |           3203.486119 |           6406.972238 |
|          4294967296 |           3197.879607 |           6395.759214 |
|          8589934592 |           3210.480912 |           6420.961823 |