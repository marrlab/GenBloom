"""
WSI bag-level evaluation: logistic regression and KNN (k=1,5,10).

Called by multi_dataset_eval.py. Not intended to be run directly.
"""

import csv
import h5py
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WSILinearProbeEvaluator:

    def __init__(
        self,
        checkpoint_path: str,
        train_csv: str,
        val_csv: str,
        data_dir: str,
        output_dir: str,
        test_csv: str = None,
        embed_dim: int = 1024,
        feature_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        patch_size: int = 16,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.checkpoint_path = checkpoint_path
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.embed_dim = embed_dim
        self.feature_dim = feature_dim
        self.depth = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.device = device
        self.model = None
        os.makedirs(output_dir, exist_ok=True)

    def load_model(self):
        """Load DinoVisionTransformerNoPos from a teacher checkpoint."""
        from dinov2.models import vision_transformer_no_pos
        from dinov2.layers import FeatureEmbed

        logger.info(f"Loading model from: {self.checkpoint_path}")

        self.model = vision_transformer_no_pos.DinoVisionTransformerNoPos(
            img_size=[224, 224],
            patch_size=self.patch_size,
            in_chans=self.feature_dim,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=4,
            qkv_bias=True,
            ffn_bias=True,
            proj_bias=True,
            drop_path_rate=0.0,
            init_values=1e-5,
            embed_layer=FeatureEmbed,
        )

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        if 'teacher' not in checkpoint:
            raise KeyError(f"Missing 'teacher' key. Found: {list(checkpoint.keys())}")
        state_dict = checkpoint['teacher']

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k.replace('backbone.', '', 1)
            elif not k.startswith('dino_head.') and not k.startswith('ibot_head.'):
                new_key = k
            else:
                continue
            new_key = re.sub(r'^blocks\.(\d+)\.', r'blocks.0.\1.', new_key)
            new_state_dict[new_key] = v

        self.model.load_state_dict(new_state_dict, strict=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded.")

    def extract_bag_embedding(self, h5_path: str) -> np.ndarray:
        with h5py.File(h5_path, 'r') as f:
            features = f['features'][:]
        features = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model.forward_features(features)
        return output['x_norm_clstoken'].squeeze(0).cpu().numpy()

    def extract_all_embeddings(self, csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Extracting embeddings from: {csv_path}")
        df = pd.read_csv(csv_path)

        if 'patient_files' in df.columns:
            patient_col = 'patient_files'
        elif 'patient_name' in df.columns:
            patient_col = 'patient_name'
        else:
            patient_col = [col for col in df.columns if col != 'labels'][0]

        embeddings, labels = [], []
        for _, row in df.iterrows():
            h5_path = os.path.join(self.data_dir, f"{row[patient_col]}.h5")
            if not os.path.exists(h5_path):
                logger.warning(f"Missing: {h5_path} — skipping")
                continue
            try:
                embeddings.append(self.extract_bag_embedding(h5_path))
                labels.append(int(row['labels']))
            except Exception as e:
                logger.error(f"Error processing {h5_path}: {e}")

        logger.info(f"Extracted {len(embeddings)} embeddings")
        return np.stack(embeddings), np.array(labels)

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
        }

    def _logistic(self, X_train, y_train, X_val, y_val):
        clf = LogisticRegression(
            max_iter=10000, class_weight='balanced',
            random_state=42, multi_class='multinomial', solver='lbfgs',
        )
        clf.fit(X_train, y_train)
        metrics = self._metrics(y_val, clf.predict(X_val))
        logger.info(f"LR val — bAcc: {metrics['balanced_accuracy']:.4f}")
        return metrics, clf

    def _knn(self, X_train, y_train, X_val, y_val, k: int):
        clf = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='euclidean')
        clf.fit(X_train, y_train)
        metrics = self._metrics(y_val, clf.predict(X_val))
        logger.info(f"KNN k={k} val — bAcc: {metrics['balanced_accuracy']:.4f}")
        return metrics, clf

    def _save_confusion_matrix(self, method, split, y_true, y_pred, class_labels):
        cm = confusion_matrix(y_true, y_pred, labels=class_labels.tolist())
        np.save(os.path.join(self.output_dir, f"confusion_matrix_{method}_{split}.npy"), cm)

        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{method} {split} (N={cm.sum()})")
        tick_marks = np.arange(len(class_labels))
        ax.set_xticks(tick_marks); ax.set_xticklabels(class_labels, rotation=45, ha='right')
        ax.set_yticks(tick_marks); ax.set_yticklabels(class_labels)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"confusion_matrix_{method}_{split}.png"), dpi=200)
        plt.close()

    def run_full_evaluation(self, iteration: int = None, **_) -> Dict:
        if self.model is None:
            self.load_model()

        logger.info("=== Extracting embeddings ===")
        X_train, y_train = self.extract_all_embeddings(self.train_csv)
        X_val, y_val = self.extract_all_embeddings(self.val_csv)
        X_test, y_test = None, None
        if self.test_csv and os.path.exists(self.test_csv):
            X_test, y_test = self.extract_all_embeddings(self.test_csv)

        class_labels = np.unique(
            np.concatenate([y_train, y_val] + ([y_test] if y_test is not None else []))
        )

        results = {
            'logistic_regression': {'val': {}, 'test': {}},
            'knn': {'val': {}, 'test': {}},
            'metadata': {
                'iteration': iteration,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test) if X_test is not None else 0,
                'num_classes': len(class_labels),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
        }

        # Logistic regression
        logger.info("\n=== Logistic Regression ===")
        lr_val, lr_clf = self._logistic(X_train, y_train, X_val, y_val)
        results['logistic_regression']['val'] = lr_val
        self._save_confusion_matrix('linear', 'val', y_val, lr_clf.predict(X_val), class_labels)
        if X_test is not None:
            lr_test = self._metrics(y_test, lr_clf.predict(X_test))
            results['logistic_regression']['test'] = lr_test
            self._save_confusion_matrix('linear', 'test', y_test, lr_clf.predict(X_test), class_labels)
            logger.info(f"LR test — bAcc: {lr_test['balanced_accuracy']:.4f}")

        # KNN
        logger.info("\n=== KNN ===")
        for k in [1, 5, 10]:
            knn_val, knn_clf = self._knn(X_train, y_train, X_val, y_val, k)
            results['knn']['val'][f'k_{k}'] = knn_val
            self._save_confusion_matrix(f'knn_k{k}', 'val', y_val, knn_clf.predict(X_val), class_labels)
            if X_test is not None:
                knn_test = self._metrics(y_test, knn_clf.predict(X_test))
                results['knn']['test'][f'k_{k}'] = knn_test
                self._save_confusion_matrix(f'knn_k{k}', 'test', y_test, knn_clf.predict(X_test), class_labels)
                logger.info(f"KNN k={k} test — bAcc: {knn_test['balanced_accuracy']:.4f}")

        self._save_csv(results, iteration)
        self._save_json(results)
        return results

    def _save_csv(self, results: Dict, iteration):
        csv_path = os.path.join(self.output_dir, 'evaluation_results.csv')
        rows = []
        ts = results['metadata']['timestamp']

        for split in ['val', 'test']:
            m = results['logistic_regression'][split]
            if m:
                rows.append({'method': 'logistic_regression', 'split': split,
                             'iteration': iteration or '', **{k: f"{v:.6f}" for k, v in m.items()},
                             'timestamp': ts})
        for split in ['val', 'test']:
            for k_str, m in results['knn'][split].items():
                rows.append({'method': f'knn_{k_str}', 'split': split,
                             'iteration': iteration or '',
                             'balanced_accuracy': f"{m['balanced_accuracy']:.6f}",
                             'accuracy': f"{m['accuracy']:.6f}",
                             'f1_macro': f"{m['f1_macro']:.6f}",
                             'timestamp': ts})

        fieldnames = ['method', 'split', 'iteration', 'balanced_accuracy', 'accuracy', 'f1_macro', 'timestamp']
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Results saved to {csv_path}")

    def _save_json(self, results: Dict):
        json_path = os.path.join(self.output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
