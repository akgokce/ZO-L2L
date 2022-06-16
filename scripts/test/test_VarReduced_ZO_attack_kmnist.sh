python main_attack.py \
    --exp_name VarReduced_ZO_attack_kmnist \
    --train_task VarReducedZOL2L-Attack-KMNIST \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path output/VarReduced_ZO_attack_mnist_finite_diff/ckpt_best \
    --save_loss \
    --save_fig