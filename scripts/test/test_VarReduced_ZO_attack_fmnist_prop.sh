python ../../main_attack.py \
    --exp_name VarReduced_ZO_attack_fmnist_prop \
    --train_task VarReducedZOL2L-Attack-FMNIST-Prop \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path /scratch/mete/ZO-L2L/optimizers/VarReduced_ZO_attack_mnist_Prop_test20/ckpt_best \
    --save_loss \
    --save_fig