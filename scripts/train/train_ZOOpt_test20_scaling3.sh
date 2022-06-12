cd /om2/user/akgokce/projects/zol2l
source /om2/user/akgokce/anaconda/etc/profile.d/conda.sh
conda activate zol2l

python main_attack.py \
    --exp_name ZO_attack_mnist_test20_scaling3 \
    --train_task ZOL2L-Attack \
    --gpu_num 0 \
    --train optimizer_attack \
    --precision double \
    --max_test_during_training 20 \
    --random_scaling 3
