docker run --gpus all \
  --name trt2023 \
  -d \
  --ipc=host \
  --ulimit memlock=-1 \
  --restart=always \
  --ulimit stack=67108864 \
  -v /root/workspace/trt2023:/root/workspace/trt2023 \
  registry.cn-hangzhou.aliyuncs.com/trt-hackathon/trt-hackathon:final_v1 sleep 8640000

docker exec -it trt2023 /bin/bash