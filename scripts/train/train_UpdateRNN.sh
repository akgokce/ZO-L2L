python main_attack.py \
    --exp_name ZO_attack_mnist_fp32_test_20 \
    --train_task ZOL2L-Attack \
    --gpu_num 0 \
    --train optimizer_attack \
    --precision full \
    --max_test_during_training 20 \
    --convex_model_dim 100\
    --random_scaling 3
