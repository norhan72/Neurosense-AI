# from __future__ import annotations

# import json
# import math
# import os
# from glob import glob
# from typing import Dict, Iterable, List, Optional, Tuple
# import matplotlib.pyplot as plt
# import cv2
# import numpy as np


# def list_images(path: str, recursive: bool = False) -> List[str]:
#     if os.path.isfile(path):
#         return [path]
#     exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
#     imgs: List[str] = []
#     if recursive:
#         for ext in exts:
#             imgs.extend(glob(os.path.join(path, "**", ext), recursive=True))
#     else:
#         for ext in exts:
#             imgs.extend(glob(os.path.join(path, ext)))
#     imgs.sort()
#     return imgs


# def load_image(path: str, as_gray: bool = False) -> np.ndarray:
#     flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
#     img = cv2.imread(path, flag)
#     if img is None:
#         raise FileNotFoundError(f"Could not read image: {path}")
#     if not as_gray:
#         # convert BGR to RGB
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img


# def save_image_with_metadata(img: np.ndarray, out_path: str, difficulty: float) -> str:
#     # ensure RGB->BGR for OpenCV write
#     to_write = img.copy()
#     if to_write.ndim == 3 and to_write.shape[2] == 3:
#         to_write = cv2.cvtColor(to_write, cv2.COLOR_RGB2BGR)

#     base, ext = os.path.splitext(out_path)
#     out_path = f"{base}_diff_{difficulty:.2f}{ext or '.png'}"

#     ok = cv2.imwrite(out_path, to_write)
#     if not ok:
#         raise OSError(f"Failed saving image to {out_path}")

#     return out_path


# def apply_blur(img: np.ndarray, ksize: int = 11, sigma: float = 0) -> np.ndarray:
#     """Apply Gaussian blur. ksize must be odd.
#     If sigma=0 OpenCV computes from ksize.
#     """
#     if ksize % 2 == 0:
#         ksize += 1
#     blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma)
#     return blurred


# def apply_shift(img: np.ndarray, dx: int = 10) -> np.ndarray:
#     shifted = np.roll(img, dx, axis=1)
#     double_img = cv2.addWeighted(img, 0.5, shifted, 0.5, 0)
#     return double_img


# def apply_foggy(img: np.ndarray, severity: float = 0.5) -> np.ndarray:
#     """Create a foggy / hazy effect by blending the image with a bright noisy layer.

#     severity: 0..1 where 0=no change, 1=heavy fog.
#     """
#     severity = float(max(0.0, min(1.0, severity)))
#     h, w = img.shape[:2]
#     # create white-ish noise layer
#     noise = np.random.normal(loc=255 * 0.9, scale=25, size=(h, w, 1)).astype(np.float32)
#     if img.ndim == 3 and img.shape[2] == 3:
#         noise = np.repeat(noise, 3, axis=2)
#     fog_layer = noise
#     # slight blur the fog layer to make it smooth
#     fog_layer = cv2.GaussianBlur(fog_layer, (101, 101), 0)
#     fog_layer = np.clip(fog_layer, 0, 255)
#     out = (1.0 - severity) * img.astype(np.float32) + severity * fog_layer
#     out = np.clip(out, 0, 255).astype(np.uint8)
#     return out


# def compute_difficulty(img: np.ndarray) -> Tuple[float, Dict[str, float]]:
#     """Compute a difficulty metric for the image.

#     Returns (score, components) where score is between 0 and 1 (1 = most
#     difficult). The components dict contains intermediate component scores in
#     0..1 where higher means more difficult for that factor.

#     Components used:
#     - blur: high when image is blurred
#     - contrast: high when low contrast
#     - edges: high when few edges
#     - brightness: high when extreme brightness (over/under exposure)
#     """
#     # convert to grayscale
#     gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # 1) Blur measure (variance of Laplacian). Higher var -> sharper.
#     lap = cv2.Laplacian(gray, cv2.CV_64F)
#     var_lap = float(lap.var())
#     # smoother mapping: use 1 - var/(var + k) so values map smoothly to (0,1)
#     k_blur = 150.0
#     blur_comp = 1.0 - (var_lap / (var_lap + k_blur))

#     # 2) contrast (stddev). lower std -> harder. Smooth mapping with k.
#     std = float(np.std(gray))
#     k_contrast = 30.0
#     contrast_comp = 1.0 - (std / (std + k_contrast))

#     # 3) edges density
#     edges = cv2.Canny(gray, 100, 200)
#     edge_density = edges.sum() / 255.0 / gray.size
#     # smooth invert mapping with small k (edge densities are small)
#     k_edge = 0.03
#     edge_comp = 1.0 - (edge_density / (edge_density + k_edge))

#     # 4) brightness extremes: distance from mid-tone normalized
#     mean = float(np.mean(gray))
#     brightness_comp = abs(mean - 127.0) / 127.0
#     # compress a bit so mild deviations don't dominate
#     brightness_comp = brightness_comp**0.9

#     # ensure components are clamped to [0,1]
#     blur_comp = float(max(0.0, min(1.0, blur_comp)))
#     contrast_comp = float(max(0.0, min(1.0, contrast_comp)))
#     edge_comp = float(max(0.0, min(1.0, edge_comp)))
#     brightness_comp = float(max(0.0, min(1.0, brightness_comp)))

#     # weights (kept similar but can be tuned)
#     w_blur, w_edge, w_contrast, w_brightness = 0.4, 0.3, 0.2, 0.1

#     score = (
#         w_blur * blur_comp
#         + w_edge * edge_comp
#         + w_contrast * contrast_comp
#         + w_brightness * brightness_comp
#     )
#     # final clamp + small epsilon to avoid exact 0 for reasonable images
#     score = float(max(0.0, min(1.0, score)))

#     components = {
#         "var_laplacian": var_lap,
#         "blur_comp": blur_comp,
#         "stddev": std,
#         "contrast_comp": contrast_comp,
#         "edge_density": float(edge_density),
#         "edge_comp": edge_comp,
#         "mean_brightness": mean,
#         "brightness_comp": brightness_comp,
#     }

#     return score, components


# def process_and_save_images(
#     input_path: str,
#     output_dir: str,
#     edits: Iterable[str] = ("blur", "fog", "shift"),
#     recursive: bool = False,
#     blur_params: Optional[Dict] = None,
#     fog_params: Optional[Dict] = None,
#     shift_params: Optional[Dict] = None,
# ) -> List[Dict[str, str]]:
#     """Process all images found at input_path and save edited versions.

#     Returns a list of dicts with keys: original, edit, image_path, difficulty.
#     """
#     blur_params = blur_params or {"ksize": 11, "sigma": 0}
#     fog_params = fog_params or {"severity": 0.5}
#     shift_params = shift_params or {"dx": 10}

#     os.makedirs(output_dir, exist_ok=True)
#     imgs = list_images(input_path, recursive=recursive)
#     results: List[Dict[str, str]] = []

#     for p in imgs:
#         try:
#             img = load_image(p)
#         except FileNotFoundError:
#             continue
#         base = os.path.splitext(os.path.basename(p))[0]

#         variants = []
#         for e in edits:
#             if e == "blur":
#                 out = apply_blur(img, **blur_params)
#             elif e == "fog":
#                 out = apply_foggy(img, **fog_params)
#             elif e == "shift":
#                 out = apply_shift(img, **shift_params)
#             elif e == "original":
#                 out = img
#             else:
#                 # unknown edit -> skip
#                 continue

#             score, comps = compute_difficulty(out)
#             out_name = f"{base}_{e}.png"
#             out_path = os.path.join(output_dir, out_name)
#             img_path = save_image_with_metadata(out, out_path, difficulty=score)
#             results.append(
#                 {
#                     "original": p,
#                     "edit": e,
#                     "image_path": img_path,
#                     "difficulty": f"{score:.2f}",
#                 }
#             )

#     return results


# def evaluate_classifier(model, X_test, y_test, name="model"):
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     print(f"{name} Accuracy:", acc)
#     print(f"{name} Classification Report:\n", report)


# def plot_training_data(blur_errors, double_errors, has_blur, has_double):
#     plots_dir = "plots"
#     if not os.path.exists(plots_dir):
#         os.makedirs(plots_dir)

#     # Plot for blur detection
#     plt.figure(figsize=(10, 6))
#     plt.scatter(
#         blur_errors,
#         double_errors,
#         c=has_blur,
#         cmap="viridis",
#         label="Has Blur (1=Yes, 0=No)",
#     )
#     plt.xlabel("Blur Errors")
#     plt.ylabel("Double Errors")
#     plt.title("Training Data Visualization (Blur Detection)")
#     plt.legend()
#     plt.grid(True)
#     blur_plot_path = os.path.join(plots_dir, "training_blur.png")
#     plt.savefig(blur_plot_path)
#     plt.close()

#     # Plot for double vision
#     plt.figure(figsize=(10, 6))
#     plt.scatter(
#         blur_errors,
#         double_errors,
#         c=has_double,
#         cmap="plasma",
#         label="Has Double Vision (1=Yes, 0=No)",
#     )
#     plt.xlabel("Blur Errors")
#     plt.ylabel("Double Errors")
#     plt.title("Training Data Visualization (Double Vision)")
#     plt.legend()
#     plt.grid(True)
#     double_plot_path = os.path.join(plots_dir, "training_double.png")
#     plt.savefig(double_plot_path)
#     plt.close()

#     print(f"Saved training plots to: {blur_plot_path}, {double_plot_path}")


# def plot_datasets_visualization(blur_df, double_df, plots_dir="plots"):
#     """Create a set of diagnostic plots for blur and double datasets.

#     Saved files:
#       - plots/blur_hist.png
#       - plots/blur_counts.png
#       - plots/double_scatter.png
#       - plots/double_histograms.png

#     The function is defensive: it checks for expected columns and skips plots
#     if data is missing.
#     """
#     if not os.path.exists(plots_dir):
#         os.makedirs(plots_dir)

#     # ---- Blur dataset visualizations ----
#     try:
#         if blur_df is not None and len(blur_df) > 0:
#             # ensure column exists
#             if "blur_errors" in blur_df.columns:
#                 plt.figure(figsize=(8, 5))
#                 for label in sorted(
#                     blur_df.get("has_blur", pd.Series([-1] * len(blur_df))).unique()
#                 ):
#                     if label not in (0, 1):
#                         continue
#                     subset = blur_df[blur_df.get("has_blur") == label]
#                     plt.hist(
#                         subset["blur_errors"],
#                         alpha=0.6,
#                         bins=range(0, 7),
#                         label=f"has_blur={label}",
#                     )
#                 plt.xlabel("blur_errors")
#                 plt.ylabel("count")
#                 plt.title("Blur errors distribution by class")
#                 plt.legend()
#                 blur_hist_path = os.path.join(plots_dir, "blur_hist.png")
#                 plt.savefig(blur_hist_path)
#                 plt.close()

#             # class counts
#             if "has_blur" in blur_df.columns:
#                 counts = blur_df["has_blur"].value_counts().sort_index()
#                 plt.figure(figsize=(5, 4))
#                 counts.plot(kind="bar", color=["#2ca02c", "#d62728"])
#                 plt.xlabel("has_blur")
#                 plt.ylabel("count")
#                 plt.title("Blur dataset class counts")
#                 blur_counts_path = os.path.join(plots_dir, "blur_counts.png")
#                 plt.tight_layout()
#                 plt.savefig(blur_counts_path)
#                 plt.close()
#     except Exception as e:
#         logging.warning("Failed to create blur plots: %s", e)

#     # ---- Double dataset visualizations ----
#     try:
#         if double_df is not None and len(double_df) > 0:
#             # scatter: blur_errors vs double_errors colored by has_double
#             if (
#                 "blur_errors" in double_df.columns
#                 and "double_errors" in double_df.columns
#             ):
#                 plt.figure(figsize=(8, 6))
#                 colors = double_df.get("has_double", 0)
#                 plt.scatter(
#                     double_df["blur_errors"],
#                     double_df["double_errors"],
#                     c=colors,
#                     cmap="bwr",
#                     alpha=0.6,
#                 )
#                 plt.colorbar(label="has_double")
#                 plt.xlabel("blur_errors")
#                 plt.ylabel("double_errors")
#                 plt.title("Blur vs Double errors (colored by has_double)")
#                 double_scatter_path = os.path.join(plots_dir, "double_scatter.png")
#                 plt.savefig(double_scatter_path)
#                 plt.close()

#             # histograms per class for both features
#             cols = []
#             if "blur_errors" in double_df.columns:
#                 cols.append("blur_errors")
#             if "double_errors" in double_df.columns:
#                 cols.append("double_errors")

#             if cols:
#                 fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4 * len(cols)))
#                 if len(cols) == 1:
#                     axes = [axes]
#                 for ax, col in zip(axes, cols):
#                     for label in sorted(
#                         double_df.get(
#                             "has_double", pd.Series([-1] * len(double_df))
#                         ).unique()
#                     ):
#                         if label not in (0, 1):
#                             continue
#                         subset = double_df[double_df.get("has_double") == label]
#                         ax.hist(
#                             subset[col],
#                             alpha=0.6,
#                             bins=range(0, 7),
#                             label=f"has_double={label}",
#                         )
#                     ax.set_xlabel(col)
#                     ax.set_ylabel("count")
#                     ax.set_title(f"Distribution of {col} by has_double")
#                     ax.legend()
#                 double_hist_path = os.path.join(plots_dir, "double_histograms.png")
#                 plt.tight_layout()
#                 plt.savefig(double_hist_path)
#                 plt.close()
#     except Exception as e:
#         logging.warning("Failed to create double plots: %s", e)

#     print(f"Saved dataset visualizations to folder: {plots_dir}")


# if __name__ == "__main__":
#     # quick demo when executed directly (adjust paths as needed)
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Batch process images and save difficulty metadata."
#     )
#     parser.add_argument("input", help="Input file or directory")
#     parser.add_argument("output", help="Output directory")
#     args = parser.parse_args()

#     res = process_and_save_images(
#         args.input, args.output, edits=("blur", "fog", "shift")
#     )
