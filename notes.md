## check for existing runs
ps aux | grep mlx_lm


source ~/mlx-env/bin/activate
nohup mlx_lm.server \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --host 0.0.0.0 \
    --port 8080 \
    > ~/mlx_server.log 2>&1 &
disow