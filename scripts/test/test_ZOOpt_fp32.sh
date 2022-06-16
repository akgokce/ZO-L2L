python main_attack.py \
    --exp_name ZO_attack_mnist_fp32_test20 \
    --train_task ZOL2L-Attack \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path ckpt_best \
    --save_loss \
    --save_fig \
    --precision full