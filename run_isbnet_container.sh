docker run \
    --runtime=nvidia --gpus all \
    -it --rm \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --shm-size=6g \
    --mount type=bind,source="$(pwd)/workspace",target=/root/workspace \
    --name isbnet \
    --network=host \
   isbnet:v2
