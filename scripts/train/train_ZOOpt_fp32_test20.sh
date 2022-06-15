cd /om2/user/akgokce/projects/zol2l
source /om2/user/akgokce/anaconda/etc/profile.d/conda.sh
conda activate zol2l

python main_attack.py \
    --exp_name ZO_attack_mnist_fp32_test20 \
    --train_task ZOL2L-Attack \
    --gpu_num 0 \
    --train optimizer_attack \
    --precision full \
    --max_test_during_training 20 
