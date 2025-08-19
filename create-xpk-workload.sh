#!/usr/bin/env bash

export CLUSTER_NAME="hongyi-xpk-v6e-8"
export ZONE="us-central1-b"
export PROJECT_ID="tpu-prod-env-one-vm"
export TPU_TYPE="v6e-8"
export NUM_SLICES=1
export WORKLOAD_NAME=${1}

xpk workload create \
    --cluster=${CLUSTER_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --device-type=${TPU_TYPE} \
    --command="git clone https://github.com/hylin2002/accelerator-microbenchmarks.git && cd accelerator-microbenchmarks && git checkout fix-all-to-all && pip install -r requirements.txt && python src/run_benchmark.py --config=configs/xlml_v6e_8.yaml && gsutil cp -r /tmp/microbenchmarks/outputs gs://hongyi-xpk-bucket/ && gsutil cp -r /tmp/microbenchmarks/trace gs://hongyi-xpk-bucket" \
    --num-slices=${NUM_SLICES} \
    --docker-image=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1 \
    --workload=${WORKLOAD_NAME}