python main_attack.py \
    --exp_name ZO_attack_mnist_Prop_test20_scaling3_convex100 \
    --train_task ZOL2LProp-Attack \
    --gpu_num 0 \
    --train optimizer_attack \
    --precision double \
    --max_test_during_training 20 \
    --convex_model_dim 100\
    --random_scaling 3
