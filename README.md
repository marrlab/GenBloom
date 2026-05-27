# GenBloom

Inference and reproducibility code for **GenBloom**, a genetically-aligned foundation model for peripheral blood smears.

- Model weights: [`MarrLab/GenBloom`](https://huggingface.co/MarrLab/GenBloom)
- Patient embeddings: [`MarrLab/DinoBloom_hemato_embeddings`](https://huggingface.co/datasets/MarrLab/DinoBloom_hemato_embeddings)

## Setup

```bash
conda create -n genbloom python=3.9 -y
conda activate genbloom
pip install -e .
```

## Demo

[`inference_genbloom.ipynb`](inference_genbloom.ipynb) — a minimal notebook that downloads the GenBloom-V and GenBloom-G checkpoints plus one example patient from HuggingFace, and runs inference end-to-end. Start here.

## Reproducing the paper

Recreates the WSI classification numbers (AML-Hehr, APL-AML, cAItomorph binary fold).

### 1. Get the checkpoints

```python
from huggingface_hub import snapshot_download
snapshot_download("MarrLab/GenBloom", local_dir="checkpoints")
```

Layout:

```
checkpoints/genbloom_v/genbloom_v.pth
checkpoints/genbloom_g/genbloom_g_fold{0..4}.pth
```

### 2. Point to your feature bags

Each dataset directory must contain one `<patient>.h5` per patient with a `features` dataset of shape `(N_cells, 768)`. If you don't have them locally, the same embeddings are released at [`MarrLab/DinoBloom_hemato_embeddings`](https://huggingface.co/datasets/MarrLab/DinoBloom_hemato_embeddings).

```bash
export AML_HEHR_DATA_DIR=/path/to/aml_hehr
export APL_AML_DATA_DIR=/path/to/apl_aml
export CATIOMORPH_DATA_DIR=/path/to/catiomorph
```

### 3. Run evaluation

**GenBloom-V** (vision encoder only):

```bash
python dinov2/eval/multi_dataset_eval.py \
    --genbloom-v-checkpoint checkpoints/genbloom_v/genbloom_v.pth \
    --output-dir outputs/classification/genbloom_v
```

**GenBloom-G** — single fold:

```bash
python dinov2/eval/multi_dataset_eval.py \
    --genbloom-g-checkpoint checkpoints/genbloom_g/genbloom_g_fold0.pth \
    --fold 0 \
    --output-dir outputs/classification/fold_0
```

**GenBloom-G** — all 5 folds on SLURM:

```bash
sbatch eval_genbloom_g.slurm    # 5-fold array job
sbatch eval_genbloom_v.slurm    # GenBloom-V baseline
```

Results land in `outputs/classification/.../all_metrics.csv`.

### 4. Plot

```bash
python plot_barplots.py --output-dir figures
```

## License

Apache 2.0 — see `LICENSE`. Derived from Meta AI's DINOv2.
