import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import cv2
from pathlib import Path

class STEMBalancer:
    def __init__(self, target_ratio=0.7, k_neighbors=5, max_iterations=30, min_improvement=5, image_size=(256, 256)):
        self.target_ratio = target_ratio
        self.k_neighbors = k_neighbors
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.image_size = image_size

    def load_data_from_folders(self, base_path):
        """Load and preprocess images from class folders."""
        X, y = [], []
        base_path = Path(base_path)
        print("Loading data from folders...")

        for class_idx, class_folder in enumerate(base_path.iterdir()):

            if class_folder.is_dir():
                print(f"Processing class '{class_folder.name}' (index {class_idx})")
                folder_count = 0
                for file_path in class_folder.glob("*"):
                    if file_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
                        try:
                            img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img_resized = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
                                img_flat = img_resized.flatten().astype(np.float32) / 255.0
                                X.append(img_flat)
                                y.append(class_folder.name)
                                folder_count += 1
                            else:
                                print(f"Warning: Failed to load image {file_path}")
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")

                print(f"Loaded {folder_count} images from '{class_folder.name}'")

        if not X:
            raise ValueError("No valid images found in the specified directory.")

        X = np.array(X)
        y = np.array(y)

        print(f"\nDataset loaded: {X.shape[0]} samples, {len(np.unique(y))} classes.")
        return X, y

    def calculate_imbalance_ratio(self, y):
        """Calculate minority-to-majority class ratio."""
        class_counts = Counter(y)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        return min_count / max_count if max_count > 0 else 0

    def apply_smote_enn(self, X, y):
        """Apply iterative SMOTE followed by ENN cleaning."""
        X_current, y_current = X.copy(), y.copy()
        class_counts = Counter(y_current)
        max_class_count = max(class_counts.values())

        iteration = 0
        prev_total_samples = len(X_current)

        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")
            print(f"Current class distribution: {Counter(y_current)}")
            print(f"Current imbalance ratio: {self.calculate_imbalance_ratio(y_current):.3f}")

            X_new, y_new = [], []
            samples_added = 0

            for class_label in np.unique(y_current):
                class_samples = X_current[y_current == class_label]
                if len(class_samples) < 2:
                    continue

                n_samples_needed = max_class_count - len(class_samples)
                nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(class_samples)))
                nn.fit(class_samples)

                for _ in range(n_samples_needed):
                    idx = np.random.randint(0, len(class_samples))
                    sample = class_samples[idx]
                    distances, indices = nn.kneighbors([sample])
                    neighbor_idx = np.random.choice(indices[0][1:])
                    neighbor = class_samples[neighbor_idx]

                    delta = np.random.rand()
                    synthetic = sample + delta * (neighbor - sample)

                    X_new.append(synthetic)
                    y_new.append(class_label)
                    samples_added += 1

            if samples_added == 0:
                print("No new samples added. Ending SMOTE iterations.")
                break

            X_current = np.vstack([X_current] + X_new)
            y_current = np.hstack([y_current, y_new])

            # ENN step
            nn = NearestNeighbors(n_neighbors=4)
            nn.fit(X_current)
            _, indices = nn.kneighbors(X_current)

            keep_indices = []
            for idx in range(len(X_current)):
                neighbors = y_current[indices[idx][1:]]
                if np.sum(neighbors == y_current[idx]) >= 2:
                    keep_indices.append(idx)

            X_current = X_current[keep_indices]
            y_current = y_current[keep_indices]

            current_total_samples = len(X_current)
            improvement = current_total_samples - prev_total_samples

            print(f"Samples after ENN: {current_total_samples} (Improvement: {improvement})")

            if improvement < self.min_improvement:
                print(f"Improvement {improvement} < minimum {self.min_improvement}. Stopping.")
                break

            prev_total_samples = current_total_samples

            if self.calculate_imbalance_ratio(y_current) >= self.target_ratio:
                print(f"Target imbalance ratio {self.target_ratio} reached.")
                break

        return X_current, y_current

    def apply_mixup(self, X, y):
        """Apply Mixup augmentation technique."""
        print("\nApplying Mixup augmentation...")
        X_mixed, y_mixed = X.tolist(), y.tolist()

        for class_label in np.unique(y):
            indices = np.where(y == class_label)[0]
            n_samples = len(indices)

            for _ in range(n_samples // 2):
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
                lam = np.random.beta(0.4, 0.4)
                mixed_sample = lam * X[idx1] + (1 - lam) * X[idx2]

                X_mixed.append(mixed_sample)
                y_mixed.append(class_label)

        X_mixed = np.array(X_mixed)
        y_mixed = np.array(y_mixed)

        print(f"Mixup completed. New dataset size: {len(X_mixed)} samples.")
        return X_mixed, y_mixed

    def save_balanced_data(self, X, y, output_path, stage="final"):
        """Save balanced images into directories."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        for class_label in np.unique(y):
            class_folder = output_path / class_label
            class_folder.mkdir(parents=True, exist_ok=True)

            indices = np.where(y == class_label)[0]
            for idx, sample_idx in enumerate(indices):
                img = (X[sample_idx].reshape(self.image_size) * 255).astype(np.uint8)
                img_path = class_folder / f"{stage}_sample_{idx}.png"
                cv2.imwrite(str(img_path), img)

        print(f"Saved {stage} dataset at: {output_path}")

    def fit_resample(self, base_path, output_base_folder="balanced_output"):
        """Complete STEM resampling pipeline."""
        X, y = self.load_data_from_folders(base_path)
        print("\nInitial class distribution:", Counter(y))
        print(f"Initial imbalance ratio: {self.calculate_imbalance_ratio(y):.3f}")

        base_path = Path(base_path)
        output_base_path = base_path / output_base_folder
        smote_enn_path = output_base_path / "after_smote_enn"
        final_path = output_base_path / "final_balanced"

        # Step 1: SMOTE-ENN
        X_smote_enn, y_smote_enn = self.apply_smote_enn(X, y)
        self.save_balanced_data(X_smote_enn, y_smote_enn, smote_enn_path, stage="smote_enn")

        # Step 2: Mixup
        X_final, y_final = self.apply_mixup(X_smote_enn, y_smote_enn)
        self.save_balanced_data(X_final, y_final, final_path, stage="final")

        # Step 3: Save Summary
        summary_path = output_base_path / "balance_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=== STEM Balancing Summary ===\n\n")
            f.write(f"Initial distribution: {Counter(y)}\n")
            f.write(f"After SMOTE-ENN: {Counter(y_smote_enn)}\n")
            f.write(f"Final distribution: {Counter(y_final)}\n\n")
            f.write(f"Initial imbalance ratio: {self.calculate_imbalance_ratio(y):.3f}\n")
            f.write(f"Final imbalance ratio: {self.calculate_imbalance_ratio(y_final):.3f}\n")

        print("\nBalancing completed successfully!")
        return X_final, y_final

if __name__ == "__main__":
    balancer = STEMBalancer(
        target_ratio=0.7,
        k_neighbors=4,
        max_iterations=30,
        min_improvement=5,
        image_size=(256, 256)
    )

    data_folder = r"/mnt/d/Harsh/Major_Project/Database/DU_original"

    try:
        X_balanced, y_balanced = balancer.fit_resample(data_folder)
    except Exception as e:
        print(f"An error occurred during balancing: {e}")
