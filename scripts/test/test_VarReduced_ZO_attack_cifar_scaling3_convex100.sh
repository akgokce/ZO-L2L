python main_attack.py \
    --exp_name VarReduced_ZO_attack_cifar_scaling3_convex100 \
    --train_task VarReducedZOL2L-Attack-CIFAR10 \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path output/VarReduced_ZO_attack_mnist_test20_scaling3_convex100/ckpt_best \
    --save_loss \
    --save_fig