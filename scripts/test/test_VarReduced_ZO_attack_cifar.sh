python main_attack.py \
    --exp_name VarReduced_ZO_attack_cifar10 \
    --train_task VarReducedZOL2L-Attack-CIFAR10 \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path /scratch/mete/ZO-L2L/optimizers/VarReduced_ZO_attack_mnist_finite_diff/ckpt_best \
    --save_loss \
    --save_fig