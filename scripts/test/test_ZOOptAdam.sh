python main_attack.py \
    --exp_name ZO_attack_mnist_Adam_test20 \
    --train_task ZOL2LAdam-Attack \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path output/ZO_attack_mnist_Adam_test20/ckpt_best \
    --save_loss \
    --save_fig