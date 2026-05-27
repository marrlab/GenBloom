#!/bin/bash

#SBATCH -J ctr_catifold_eval
#SBATCH -o logs/eval_catifold_%A_%a.out
#SBATCH -e logs/eval_catifold_%A_%a.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --array=0-4

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${REPO_ROOT}/checkpoints/genbloom_g/lam1.0/transformer_finetune_aug_band_level_supcon}"
CONDA_ENV="${CONDA_ENV:-dinoBloom2}"

export GENHEMA_DATA_ROOT="${GENHEMA_DATA_ROOT:-${REPO_ROOT}/data}"
export GENHEMA_SPLITS_ROOT="${GENHEMA_SPLITS_ROOT:-${REPO_ROOT}/splits}"

# Feature bag directories — set these to the directories containing your .h5 feature files.
# Each directory must contain one .h5 file per patient, named <patient_name>.h5,
# with a 'features' dataset of shape (N_patches, feature_dim).
export AML_HEHR_DATA_DIR="${AML_HEHR_DATA_DIR:-/path/to/AML_Hehr_features/dinobloom-b}"
export APL_AML_DATA_DIR="${APL_AML_DATA_DIR:-/path/to/APL_AML_features}"
export CATIOMORPH_DATA_DIR="${CATIOMORPH_DATA_DIR:-/path/to/catiomorph_features/dinobloom-b}"

FOLD="${SLURM_ARRAY_TASK_ID:-0}"

CKPT_PATH="${CKPT_PATH:-${CHECKPOINT_ROOT}/fold_${FOLD}/best_model.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/classification/fold_${FOLD}}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Checkpoint not found: ${CKPT_PATH}. Skipping."
  exit 0
fi

mkdir -p logs "${OUTPUT_DIR}"

set +u
source "${HOME}/.bashrc"
set -u
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python dinov2/eval/multi_dataset_eval.py \
    --genbloom-g-checkpoint "${CKPT_PATH}" \
    --fold "${FOLD}" \
    --output-dir "${OUTPUT_DIR}"
