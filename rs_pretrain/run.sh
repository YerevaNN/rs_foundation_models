#!/bin/bash


python -m torch.distributed.run --nproc_per_node 8 --rdzv-endpoint=localhost:29551 main_ibot_it.py  \
        --act_in_head gelu --arch channelvit_base --batch_size_per_gpu 16 --accumulation_steps 4 --num_workers 20 --seed 0 \
        --lr 0.00025 --min_lr 2e-06 --optimizer adamw --clip_grad 0.5 --drop_path 0.1 \
        --sample_iters 400e6 --warmup_sample_iters 6e6 --freeze_last_layer 6e6 \
        --global_crops_number 2 --global_crops_scale 0.32 1.0 --local_crops_number 10 --local_crops_scale 0.05 0.32 \
        --momentum_teacher 0.996  --norm_last_layer true --out_dim 8192 --shared_head false --shared_head_teacher true \
        --patch_size 16 --pred_ratio 0 0.7 --pred_ratio_var 0 0.05 --pred_shape rand \
        --warmup_teacher_temp 0.04 --warmup_teacher_patch_temp 0.04 --teacher_temp 0.06 --teacher_patch_temp 0.06 \
        --use_fp16 true --use_masked_im_modeling true \
        --warmup_teacher_temp_sample_iters 9e6 --weight_decay 0.04 --lambda2 1 --lambda3 1 \
        --saving_freq 2e6 --saving_sample_iters 9e6 18e6 27e6 36e6 45e6 90e6 135e6 180e6 225e6 270e6 315e6 360e6 \
        --compile_decoder false --compile_loss true --decoder_compile_mode default --use_overlap false --only_decay false \
        --output_dir /nfs/dgx/raid/rs/rs/channel_logs --sampling_subset true \
        --add_ch_embed true --shared_proj true --sync_channels false 