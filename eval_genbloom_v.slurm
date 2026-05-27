#!/bin/bash

#SBATCH -J genbloom_v_eval
#SBATCH -o logs/eval_genbloom_v_%j.out
#SBATCH -e logs/eval_genbloom_v_%j.err
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=40G

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
CONDA_ENV="${CONDA_ENV:-dinoBloom2}"

export GENHEMA_DATA_ROOT="${GENHEMA_DATA_ROOT:-${REPO_ROOT}/data}"
export GENHEMA_SPLITS_ROOT="${GENHEMA_SPLITS_ROOT:-${REPO_ROOT}/splits}"

# Feature bag directories — set these to the directories containing your .h5 feature files.
# Each directory must contain one .h5 file per patient, named <patient_name>.h5,
# with a 'features' dataset of shape (N_patches, feature_dim).
export AML_HEHR_DATA_DIR="${AML_HEHR_DATA_DIR:-/path/to/AML_Hehr_features/dinobloom-b}"
export APL_AML_DATA_DIR="${APL_AML_DATA_DIR:-/path/to/APL_AML_features}"
export CATIOMORPH_DATA_DIR="${CATIOMORPH_DATA_DIR:-/path/to/catiomorph_features/dinobloom-b}"

CKPT_PATH="${CKPT_PATH:-${REPO_ROOT}/checkpoints/genbloom_v/teacher_checkpoint.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/classification/genbloom_v}"

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Checkpoint not found: ${CKPT_PATH}. Exiting."
  exit 1
fi

mkdir -p logs "${OUTPUT_DIR}"

set +u
source "${HOME}/.bashrc"
set -u
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

python dinov2/eval/multi_dataset_eval.py \
    --genbloom-v-checkpoint "${CKPT_PATH}" \
    --output-dir "${OUTPUT_DIR}"
