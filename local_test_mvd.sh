#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test_mvd.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# datasets folder
DATASET_DIR="datasets"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directories.
MVD_FOLDER="mvd"
##############################################################################################
EXP_FOLDER="exp/train_on_train_set_pascal_voc_2012_iter_20000_bs_16_lr_5e-3_lr_decay_step_200_weight_decay_1e-5"
##############################################################################################
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${MVD_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MVD_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MVD_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${MVD_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${MVD_FOLDER}/${EXP_FOLDER}/export"
mkdir -p "${INIT_FOLDER}"
mkdir -p "${TRAIN_LOGDIR}"
mkdir -p "${EVAL_LOGDIR}"
mkdir -p "${VIS_LOGDIR}"
mkdir -p "${EXPORT_DIR}"

# # Copy locally the trained checkpoint as the initial checkpoint.
# TF_INIT_ROOT="http://download.tensorflow.org/models"
# TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"
# cd "${INIT_FOLDER}"
# wget -nd -c "${TF_INIT_ROOT}/${TF_INIT_CKPT}"
# tar -xf "${TF_INIT_CKPT}"
cd "${CURRENT_DIR}"

MVD_DATASET="${WORK_DIR}/${DATASET_DIR}/${MVD_FOLDER}/tfrecord"

#### Train 10 iterations.
# NUM_ITERATIONS=20000
# python "${WORK_DIR}"/train.py \
#   --logtostderr \
#   --num_clones=4 \
#   --train_split="train" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --train_crop_size=513 \
#   --train_crop_size=513 \
#   --train_batch_size=16 \
#   --base_learning_rate=0.005 \
#   --learning_rate_decay_step=200 \
#   --weight_decay=0.00001 \
#   --training_number_of_steps="${NUM_ITERATIONS}" \
#   --log_steps=1 \
#   --save_summaries_secs=60 \
#   --fine_tune_batch_norm=true \
#   --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \
#   --initialize_last_layer=false \
#   --train_logdir="${TRAIN_LOGDIR}" \
#   --dataset_dir="${MVD_DATASET}"

# --tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_cityscapes_train/model.ckpt" \
# —-tf_initial_checkpoint="${INIT_FOLDER}/deeplabv3_pascal_train_aug/model.ckpt" \

#### Run evaluation. This performs eval over the full val split (33 images) and
#### will take a while.
#### Using the provided checkpoint, one should expect mIOU=82.20%.
python "${WORK_DIR}"/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size=2449 \
  --eval_crop_size=3265 \
  --checkpoint_dir="${TRAIN_LOGDIR}" \
  --eval_logdir="${EVAL_LOGDIR}" \
  --dataset_dir="${MVD_DATASET}" \
  --max_number_of_evaluations=1

# Visualize the results.
# python "${WORK_DIR}"/vis.py \
#   --logtostderr \
#   --vis_split="val" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --vis_crop_size=2449 \
#   --vis_crop_size=3265 \
#   --checkpoint_dir="${TRAIN_LOGDIR}" \
#   --vis_logdir="${VIS_LOGDIR}" \
#   --dataset_dir="${MVD_DATASET}" \
#   --max_number_of_iterations=1

# Export the trained checkpoint.
# CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
# EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

# python "${WORK_DIR}"/export_model.py \
#   --logtostderr \
#   --checkpoint_path="${CKPT_PATH}" \
#   --export_path="${EXPORT_PATH}" \
#   --model_variant="xception_65" \
#   --atrous_rates=6 \
#   --atrous_rates=12 \
#   --atrous_rates=18 \
#   --output_stride=16 \
#   --decoder_output_stride=4 \
#   --num_classes=65 \
#   --crop_size=2449 \
#   --crop_size=3265 \
#   --inference_scales=1.0

# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
