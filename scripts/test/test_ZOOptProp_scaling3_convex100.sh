python main_attack.py \
    --exp_name ZO_attack_mnist_Prop_test20_scaling3_convex100 \
    --train_task ZOL2LProp-Attack \
    --gpu_num 0 \
    --train optimizer_train_optimizee_attack \
    --ckpt_path output/ZO_attack_mnist_Prop_test20_scaling3_convex100/ckpt_best \
    --save_loss \
    --save_fig