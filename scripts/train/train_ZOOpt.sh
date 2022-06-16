python main_attack.py \
    --exp_name test \
    --train_task ZOL2LProp-Attack \
    --gpu_num 0 \
    --train optimizer_attack \
    --precision full \
    --max_test_during_training 20 \
    --convex_model_dim 100\
    --random_scaling 3
