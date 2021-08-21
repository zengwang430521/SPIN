export MASTER_PORT=29502
srun -p 3dv-share -w SH-IDC1-10-198-6-130 \
    --ntasks 1 --job-name=spin \
    --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=4 --kill-on-bad-exit=1 \
    python3 eval.py --checkpoint=logs/hmr_opt/checkpoints/2021_08_21-20_49_18.pt --dataset=h36m-p2 --log_freq=20

    python3 train.py --name hmr_opt --run_smplify --num_smplify_iters=50 \
    --lr=3e-5 --batch_size=64 \
    --pretrained_checkpoint=../pvt_pose/logs/hmr_all/checkpoints/checkpoint_latest.pth


