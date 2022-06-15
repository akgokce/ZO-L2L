python main_attack.py \
    --exp_name VarReduced_ZO_attack_fmnist \
    --train_task VarReducedZOL2L-Attack-FMNIST \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path /scratch/mete/ZO-L2L/VarReduced_ZO_attack_mnist_finite_diff/ckpt_best \
    --save_loss \
    --save_fig