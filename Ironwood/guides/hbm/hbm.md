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

## Expected output

