cd /om2/user/akgokce/projects/zol2l
source /om2/user/akgokce/anaconda/etc/profile.d/conda.sh
conda activate zol2l

python main_attack.py \
    --exp_name VarReduced_ZO_attack_mnist_finite_diff \
    --train_task VarReducedZOL2L-Attack \
    --gpu_num 0 \
    --train optimizer_attack \
    --warm_start_ckpt ./output/ZO_attack_mnist_finite_diff/ckpt_best \
    --use_finite_diff
