python main.py \
    --use_gpu \
    --create_plot \
    --filepath '.pt' | tee log.txt

python main.py \
    --use_gpu \
    --create_plot \
    --batch_size 32 \
    --lr 1e-3 \
    --filepath '.pt' | tee batch32-1e3-log.txt

python main.py \
    --use_gpu \
    --create_plot \
    --freeze_layers 15 \
    --filepath '.pt' | tee layer15-log.txt

python main.py \
    --use_gpu \
    --create_plot \
    --freeze_layers 0 \
    --filepath '.pt' | tee scratch-log.txt

python main.py \
    --use_gpu \
    --create_plot \
    --freeze_layers 3 \
    --filepath '.pt' | tee layer3-log.txt

python main.py \
    --model_name 'vgg16' \
    --freeze_layers 15 \
    --use_gpu \
    --create_plot \
    --batch_size 128 \
    --lr 5e-4 \
    --filepath '.pt' | tee vgg-log.txt