#!/bin/bash

# Set environment variables for Open MPI
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Change to the directory containing the scripts
cd consistency_models/scripts

# Set some parameters
model_directory="/home/consistency_models/models"
log_directory="/home/consistency_models/logs"
num_samples=5

#########################################################################
# Sampling from EDM models on class-conditional ImageNet-64, and LSUN 256
#########################################################################

# ImageNet-64
mpiexec -n 1 python image_sample.py --training_mode edm --batch_size 64 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path ${model_directory}/edm_imagenet64_ema.pt --attention_resolutions 32,16,8  --class_cond True --dropout 0.1 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --use_scale_shift_norm True --weight_schedule karras --log_dir ${log_directory}/edm_imagenet64

# LSUN-256 (bedroom)
mpiexec -n 1 python image_sample.py --training_mode edm --generator determ-indiv --batch_size 8 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path ${model_directory}/edm_bedroom256_ema.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras --log_dir ${log_directory}/edm_bedroom256

# LSUN-256 (cat)
mpiexec -n 1 python image_sample.py --training_mode edm --generator determ-indiv --batch_size 8 --sigma_max 80 --sigma_min 0.002 --s_churn 0 --steps 40 --sampler heun --model_path ${model_directory}/edm_cat256_ema.pt --attention_resolutions 32,16,8  --class_cond False --dropout 0.1 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --use_scale_shift_norm False --weight_schedule karras --log_dir ${log_directory}/edm_cat256

#################################################################################
# Sampling from consistency models on class-conditional ImageNet-64, and LSUN 256
#################################################################################

# ImageNet-64
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler onestep --model_path ${model_directory}/cd_imagenet64_lpips.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_imagenet64_lpips_1

mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler onestep --model_path ${model_directory}/cd_imagenet64_l2.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_imagenet64_l2_1

mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_training --sampler onestep --model_path ${model_directory}/ct_imagenet64.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/ct_imagenet64_1

# LSUN-256 (bedroom)
mpiexec -n 1 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_distillation --sampler onestep --model_path ${model_directory}/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_bedroom256_lpips_1

mpiexec -n 1 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_distillation --sampler onestep --model_path ${model_directory}/cd_bedroom256_l2.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_bedroom256_l2_1

mpiexec -n 1 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_training --sampler onestep --model_path ${model_directory}/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/ct_bedroom256_1

# LSUN-256 (cat)
mpiexec -n 1 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_distillation --sampler onestep --model_path ${model_directory}/cd_cat256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_cat256_lpips_1

mpiexec -n 1 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_distillation --sampler onestep --model_path ${model_directory}/cd_cat256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_cat256_l2_1

mpiexec -n 1 python image_sample.py --batch_size 32 --generator determ-indiv --training_mode consistency_training --sampler onestep --model_path ${model_directory}/ct_cat256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/ct_cat256_1

###################################################################
# Multistep sampling on class-conditional ImageNet-64, and LSUN 256
###################################################################

## Two-step sampling for CD (LPIPS) on ImageNet-64
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,22,39 --steps 40 --model_path ${model_directory}/cd_imagenet64_lpips.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_imagenet64_lpips_2

## Two-step sampling for CD (L2) on ImageNet-64
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,22,39 --steps 40 --model_path ${model_directory}/cd_imagenet64_l2.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_imagenet64_l2_2

## Two-step sampling for CT on ImageNet-64
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_training --sampler multistep --ts 0,106,200 --steps 201 --model_path ${model_directory}/ct_imagenet64.pt --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/ct_imagenet64_2

## Two-step sampling for CD (LPIPS) on LSUN-256 (bedroom)
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,17,39 --steps 40 --model_path ${model_directory}/cd_bedroom256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_bedroom256_lpips_2

## Two-step sampling for CD (l2) on LSUN-256 (bedroom)
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,18,39 --steps 40 --model_path ${model_directory}/cd_bedroom256_l2.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_bedroom256_l2_2

## Two-step sampling for CT on LSUN-256 (bedroom)
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_training --sampler multistep --ts 0,67,150 --steps 151 --model_path ${model_directory}/ct_bedroom256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/ct_bedroom256_2

## Two-step sampling for CD (LPIPS) on LSUN-256 (cat)
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,17,39 --steps 40 --model_path ${model_directory}/cd_cat256_lpips.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_cat256_lpips_2

## Two-step sampling for CD (l2) on LSUN-256 (cat)
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_distillation --sampler multistep --ts 0,18,39 --steps 40 --model_path ${model_directory}/cd_cat256_l2.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/cd_cat256_l2_2

## Two-step sampling for CT on LSUN-256 (cat)
mpiexec -n 1 python image_sample.py --batch_size 32 --training_mode consistency_training --sampler multistep --ts 0,67,150 --steps 151 --model_path ${model_directory}/ct_cat256.pt --attention_resolutions 32,16,8 --class_cond False --use_scale_shift_norm False --dropout 0.0 --image_size 256 --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --num_samples ${num_samples} --resblock_updown True --use_fp16 True --weight_schedule uniform --log_dir ${log_directory}/ct_cat256_2
