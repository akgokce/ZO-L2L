python main_attack.py \
    --exp_name VarReduced_ZO_attack_fmnist_scaling3 \
    --train_task VarReducedZOL2L-Attack-FMNIST \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path output/VarReduced_ZO_attack_mnist_test20_scaling3/ckpt_best \
    --save_loss \
    --save_fig