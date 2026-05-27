"""
Evaluate GenBloom-G on AML-Hehr, APL-AML, and cAItomorph (binary fold).

Runs logistic regression and KNN (k=1,5,10) across 5-fold CV for each dataset.
Results are written to --output-dir/all_metrics.csv.
"""

import sys
import os
from pathlib import Path

dinov2_root = str(Path(__file__).parent.parent.parent)
if dinov2_root not in sys.path:
    sys.path.insert(0, dinov2_root)

import argparse
import json
import logging
from typing import Dict, List
import pandas as pd
import numpy as np

from dinov2.eval.eval_linear_probe import WSILinearProbeEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("GENHEMA_DATA_ROOT", REPO_ROOT / "data"))
SPLITS_ROOT = Path(os.environ.get("GENHEMA_SPLITS_ROOT", REPO_ROOT / "splits"))


def _env_path(env_name: str, default: Path) -> str:
    return os.environ.get(env_name, str(default))


class MultiDatasetEvaluator:

    DATASETS = {
        'aml_hehr': {
            'name': 'AML_Hehr',
            'data_dir': _env_path('AML_HEHR_DATA_DIR', DATA_ROOT / 'AML_Hehr_features_extracted' / 'dinobloom-b'),
            'folds_dir': _env_path('AML_HEHR_FOLDS_DIR', SPLITS_ROOT / 'aml_hehr' / 'folds'),
            'num_folds': 5,
            'num_classes': 5,
        },
        'apl_aml': {
            'name': 'APL_AML',
            'data_dir': _env_path('APL_AML_DATA_DIR', DATA_ROOT / 'APL_AML_all'),
            'folds_dir': _env_path('APL_AML_FOLDS_DIR', SPLITS_ROOT / 'apl_aml' / 'folds'),
            'num_folds': 5,
            'num_classes': 2,
        },
        'catiomorph_binary_fold': {
            'name': 'Catiomorph Binary Fold (AML vs Normal, 5-Fold CV)',
            'data_dir': _env_path('CATIOMORPH_DATA_DIR', DATA_ROOT / 'beluga_features_extracted' / 'dinobloom-b'),
            'folds_dir': _env_path('CATIOMORPH_BINARY_FOLD_DIR', SPLITS_ROOT / 'catiomorph' / 'binary_fold'),
            'num_folds': 5,
            'num_classes': 2,
        },
    }

    def __init__(
        self,
        output_base_dir: str,
        model,
        embed_dim: int,
        datasets: List[str] = None,
        device: str = 'cuda',
        contrastive_fold: int = None,
    ):
        self.output_base_dir = output_base_dir
        self.model = model
        self.embed_dim = embed_dim
        self.device = device
        self.contrastive_fold = contrastive_fold
        self.all_metrics_rows = []

        self.datasets_to_eval = list(self.DATASETS.keys()) if datasets is None else datasets
        os.makedirs(output_base_dir, exist_ok=True)

    def _evaluate_dataset(self, dataset_key: str) -> Dict:
        config = self.DATASETS[dataset_key]
        logger.info("\n" + "=" * 80)
        logger.info(f"EVALUATING {config['name']} ({config['num_folds']}-Fold CV)")
        logger.info("=" * 80)

        # AML-Hehr: restrict to matching contrastive fold to prevent data leakage
        if self.contrastive_fold is not None and dataset_key == 'aml_hehr':
            fold_range = [self.contrastive_fold]
            logger.info(f"Restricting to fold {self.contrastive_fold} (contrastive leakage prevention)")
        else:
            fold_range = range(config['num_folds'])

        folds_results = {}
        for fold_idx in fold_range:
            logger.info(f"\n--- Fold {fold_idx + 1}/{config['num_folds']} ---")

            fold_dir = os.path.join(config['folds_dir'], f'data_fold_{fold_idx}')
            output_dir = os.path.join(self.output_base_dir, f'{dataset_key}_fold_{fold_idx}')
            os.makedirs(output_dir, exist_ok=True)

            evaluator = WSILinearProbeEvaluator(
                checkpoint_path="unused",
                train_csv=os.path.join(fold_dir, 'train.csv'),
                val_csv=os.path.join(fold_dir, 'val.csv'),
                test_csv=os.path.join(fold_dir, 'test.csv'),
                data_dir=config['data_dir'],
                output_dir=output_dir,
                device=self.device,
            )
            evaluator.model = self.model
            evaluator.embed_dim = self.embed_dim

            result = evaluator.run_full_evaluation(iteration=fold_idx, quick_eval=True)
            result['dataset'] = dataset_key
            result['fold'] = fold_idx
            result['num_classes'] = config['num_classes']
            folds_results[f'fold_{fold_idx}'] = result

            logger.info(f"Fold {fold_idx} completed")

        return self._aggregate_folds(folds_results, dataset_key, config['num_classes'])

    def _aggregate_folds(self, folds_results: Dict, dataset_name: str, num_classes: int) -> Dict:
        logger.info(f"\n=== Aggregating {dataset_name} ===")

        aggregated = {
            'dataset': dataset_name,
            'num_classes': num_classes,
            'folds': folds_results,
            'fold_statistics': {}
        }

        for method in ['logistic_regression', 'knn']:
            aggregated['fold_statistics'][method] = {
                'val': self._fold_stats(folds_results, method, 'val'),
                'test': self._fold_stats(folds_results, method, 'test'),
            }
            self._collect_metrics(folds_results, dataset_name, method)

        return aggregated

    def _fold_stats(self, folds_results: Dict, method: str, split: str) -> Dict:
        metrics = ['balanced_accuracy', 'accuracy', 'f1_macro']

        if method == 'knn':
            k_keys = set()
            for fold_result in folds_results.values():
                k_keys.update(fold_result.get(method, {}).get(split, {}).keys())
            stats = {}
            for k_key in sorted(k_keys):
                stats[k_key] = {}
                for metric in metrics:
                    values = [
                        fold_result.get(method, {}).get(split, {}).get(k_key, {}).get(metric)
                        for fold_result in folds_results.values()
                    ]
                    values = [v for v in values if v is not None]
                    if values:
                        stats[k_key][metric] = {
                            'mean': float(np.mean(values)),
                            'std': float(np.std(values)),
                            'values': [float(v) for v in values],
                        }
            return stats
        else:
            stats = {}
            for metric in metrics:
                values = [
                    fold_result.get(method, {}).get(split, {}).get(metric)
                    for fold_result in folds_results.values()
                ]
                values = [v for v in values if v is not None]
                if values:
                    stats[metric] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'values': [float(v) for v in values],
                    }
            return stats

    def _collect_metrics(self, folds_results: Dict, dataset_name: str, method: str):
        for fold_result in folds_results.values():
            fold_num = fold_result.get('fold')
            num_classes = fold_result.get('num_classes', 'N/A')
            for split in ['val', 'test']:
                if method == 'knn':
                    for k_key, k_metrics in fold_result.get(method, {}).get(split, {}).items():
                        if isinstance(k_metrics, dict) and 'balanced_accuracy' in k_metrics:
                            self.all_metrics_rows.append({
                                'dataset': dataset_name,
                                'fold': fold_num,
                                'method': f'{method}_{k_key}',
                                'split': split,
                                'balanced_accuracy': k_metrics.get('balanced_accuracy', 'N/A'),
                                'accuracy': k_metrics.get('accuracy', 'N/A'),
                                'f1_macro': k_metrics.get('f1_macro', 'N/A'),
                                'num_classes': num_classes,
                            })
                else:
                    fold_metrics = fold_result.get(method, {}).get(split, {})
                    if fold_metrics and 'balanced_accuracy' in fold_metrics:
                        self.all_metrics_rows.append({
                            'dataset': dataset_name,
                            'fold': fold_num,
                            'method': method,
                            'split': split,
                            'balanced_accuracy': fold_metrics.get('balanced_accuracy', 'N/A'),
                            'accuracy': fold_metrics.get('accuracy', 'N/A'),
                            'f1_macro': fold_metrics.get('f1_macro', 'N/A'),
                            'num_classes': num_classes,
                        })

    def run(self) -> Dict:
        logger.info("\n" + "=" * 100)
        logger.info("STARTING EVALUATION")
        logger.info("=" * 100)

        all_results = {'datasets': {}}
        self.all_metrics_rows = []

        for ds_key in self.datasets_to_eval:
            if ds_key not in self.DATASETS:
                logger.warning(f"Unknown dataset: {ds_key}")
                continue
            try:
                all_results['datasets'][ds_key] = self._evaluate_dataset(ds_key)
            except Exception as e:
                import traceback
                logger.error(f"Error evaluating {ds_key}: {e}")
                traceback.print_exc()
                all_results['datasets'][ds_key] = {'error': str(e)}

        # Save all_metrics.csv — this is the primary output used by plot_barplots.py
        if self.all_metrics_rows:
            metrics_csv_path = os.path.join(self.output_base_dir, 'all_metrics.csv')
            pd.DataFrame(self.all_metrics_rows).to_csv(metrics_csv_path, index=False)
            logger.info(f"Metrics saved to {metrics_csv_path}")

        self._print_summary(all_results)
        return all_results

    def _print_summary(self, all_results: Dict):
        logger.info("\n" + "=" * 100)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 100)

        for dataset_name, dataset_result in all_results['datasets'].items():
            logger.info(f"\n{dataset_name.upper()}")
            logger.info("-" * 60)
            if 'error' in dataset_result:
                logger.error(f"  Error: {dataset_result['error']}")
                continue
            for method, method_stats in dataset_result.get('fold_statistics', {}).items():
                for split, split_stats in method_stats.items():
                    if method == 'knn':
                        for k_key, k_stats in split_stats.items():
                            if 'balanced_accuracy' in k_stats:
                                bacc = k_stats['balanced_accuracy']
                                logger.info(f"  {split} {method} {k_key}: {bacc['mean']:.4f} ± {bacc['std']:.4f}")
                    else:
                        if 'balanced_accuracy' in split_stats:
                            bacc = split_stats['balanced_accuracy']
                            logger.info(f"  {split} {method}: {bacc['mean']:.4f} ± {bacc['std']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate GenBloom on AML-Hehr, APL-AML, and cAItomorph'
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        '--genbloom-g-checkpoint', type=str,
        help='Path to GenBloom-G best_model.pth (contrastive aligner + fine-tuned vision encoder)',
    )
    model_group.add_argument(
        '--genbloom-v-checkpoint', type=str,
        help='Path to GenBloom-V teacher_checkpoint.pth (vision encoder only, no contrastive head)',
    )

    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to write all_metrics.csv and per-fold results')
    parser.add_argument('--fold', type=int, default=None,
                        help='Restrict AML-Hehr to this fold (prevents contrastive data leakage)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])

    args = parser.parse_args()

    from genbloom_g import load_contrastive_model, _build_vision_model

    if args.genbloom_g_checkpoint:
        # GenBloom-G: contrastive aligner with (optionally fine-tuned) vision encoder.
        # Uses unprojected 768-d CLS token — projection head is bypassed.
        dinov2_ckpt = str(REPO_ROOT / 'checkpoints' / 'genbloom_v' / 'teacher_checkpoint.pth')
        model, _ = load_contrastive_model(
            checkpoint_path=args.genbloom_g_checkpoint,
            dinov2_checkpoint_path=dinov2_ckpt,
            embed_dim=768, feature_dim=768, depth=6, num_heads=12,
            device=args.device,
        )
        model.return_unprojected = True
        embed_dim = model.unprojected_dim
        logger.info(f"GenBloom-G loaded (embed_dim={embed_dim})")
    else:
        # GenBloom-V: vision encoder alone, no contrastive head.
        # Aggregates patch-level feature bags into a 768-d CLS token directly.
        model = _build_vision_model(
            checkpoint_path=args.genbloom_v_checkpoint,
            embed_dim=768, feature_dim=768, depth=6, num_heads=12,
            patch_size=1, device=args.device,
        )
        embed_dim = 768
        logger.info(f"GenBloom-V loaded (embed_dim={embed_dim})")

    os.makedirs(args.output_dir, exist_ok=True)

    MultiDatasetEvaluator(
        output_base_dir=args.output_dir,
        model=model,
        embed_dim=embed_dim,
        device=args.device,
        contrastive_fold=args.fold,
    ).run()

    logger.info("\nDone. Results in: " + args.output_dir)


if __name__ == '__main__':
    main()
