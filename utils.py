from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
import logging
import joblib
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report


def get_training_data_fallback():
    # small hard-coded dataset used when no images are available
    data = {
        "blur_errors": [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 1, 2, 4, 5, 3, 2, 1, 0],
        "double_errors": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0],
        "has_blur": [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        "has_double": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
    }
    df = pd.DataFrame(data)
    X = df[["blur_errors", "double_errors"]]
    y_blur = df["has_blur"]
    y_double = df["has_double"]
    return X, y_blur, y_double, df


def build_blur_dataset(blur_dir="images/train/blur"):
    """Build dataset for blur model from files in blur_dir.

    Returns: X (DataFrame with 'blur_errors'), y (Series), df (DataFrame)
    """
    records = []
    if not os.path.exists(blur_dir) or not os.path.isdir(blur_dir):
        X, yb, yd, df = get_training_data_fallback()
        return X[["blur_errors"]], yb, df

    var_list = []
    paths = []
    labels = []
    for fname in os.listdir(blur_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(blur_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        has_blur = 0 if "clear" in fname.lower() else 1
        var_lap = cv2.Laplacian(img, cv2.CV_64F).var()
        var_list.append(var_lap)
        paths.append(path)
        labels.append(has_blur)

    if len(var_list) == 0:
        X, yb, yd, df = get_training_data_fallback()
        return X[["blur_errors"]], yb, df

    vmin = min(var_list)
    vmax = max(var_list)
    vrange = vmax - vmin if vmax > vmin else 1.0

    for path, var_lap, has_blur in zip(paths, var_list, labels):
        norm = (var_lap - vmin) / vrange
        blur_errors = int(round(5 * (1 - norm)))
        records.append({"blur_errors": blur_errors, "path": path, "has_blur": has_blur})

    df = pd.DataFrame(records)
    X = df[["blur_errors"]]
    y = df["has_blur"]
    return X, y, df


def build_double_dataset(double_dir="images/train/double"):
    """Build dataset for double-vision model from files in double_dir.

    Expects files named with '_double' or 'double' for augmented images.
    Returns: X (DataFrame with 'double_errors'), y (Series), df, groups
    """

    records = []
    # If the double directory doesn't exist, fall back to the toy dataset
    # but avoid leaking the label via the `double_errors` column. Return only
    # the blur-based feature to prevent trivial perfect accuracy.
    if not os.path.exists(double_dir) or not os.path.isdir(double_dir):
        X, yb, yd, df = get_training_data_fallback()
        return X[["blur_errors"]], yd, df

    for fname in os.listdir(double_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(double_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # determine label based on filename marker (existing ground truth)
        is_double = (
            1 if ("_double" in fname.lower() or "double" in fname.lower()) else 0
        )

        # compute a simple 'doubleness' metric from image content: mean absolute
        # difference between the image and a horizontally-shifted copy. This is a
        # cheap heuristic; higher values often indicate overlay / ghosting.
        try:
            shift = max(1, min(img.shape[1] // 20, 30))  # shift by ~5% width, cap 30px
            shifted = np.roll(img, shift, axis=1)
            diff = cv2.absdiff(img, shifted)
            double_metric = float(np.mean(diff)) / 255.0  # normalized 0..1
        except Exception:
            double_metric = 0.0

        # derive a group id from filename so original and its _double variant
        # are assigned the same group for group-aware splitting later.
        base = os.path.splitext(fname)[0]
        group = re.sub(r"(_double$|_double$)", "", base, flags=re.I)
        group = re.sub(r"(_double)", "", group, flags=re.I).strip()

        records.append(
            {
                "path": path,
                "double_metric": double_metric,
                "has_double": is_double,
                "group": group,
            }
        )

    if len(records) == 0:
        X, yb, yd, df = get_training_data_fallback()
        return X[["double_errors"]], yd, df

    df = pd.DataFrame(records)

    # normalize double metric to 0..5 as an integer feature
    dmin = df["double_metric"].min()
    dmax = df["double_metric"].max()
    drange = dmax - dmin if dmax > dmin else 1.0
    df["double_errors"] = (
        (((df["double_metric"] - dmin) / drange) * 5).round().astype(int)
    )

    # Features now include both blur_errors (handcrafted) and a simple
    # image-derived double_errors metric (not filename-derived)
    X = df[["double_errors"]]
    y = df["has_double"]
    groups = df["group"]
    return X, y, df, groups


def train_classifier(
    X,
    y,
    test_size=0.2,
    random_state=42,
    n_estimators=10,
    persist_path=None,
    verbose=False,
    groups=None,
):
    if verbose:
        logging.info(
            "Training classifier: test_size=%s, random_state=%s",
            test_size,
            random_state,
        )
    # If groups are provided, use a group-aware split to avoid placing
    # near-duplicate / paired images (e.g., original + _double) across
    # train and test folds which would leak information.
    if groups is not None:
        try:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state
            )
            train_idx, test_idx = next(gss.split(X, y, groups))
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        except Exception:
            # fallback to regular split if grouping fails
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    if persist_path:
        try:
            os.makedirs(os.path.dirname(persist_path), exist_ok=True)
            joblib.dump(model, persist_path)
            if verbose:
                logging.info("Saved classifier to %s", persist_path)
        except Exception as e:
            logging.warning("Failed to persist classifier: %s", e)
    return model, X_test, y_test


def evaluate_classifier(model, X_test, y_test, name="model"):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print(f"{name} Accuracy:", acc)
    print(f"{name} Classification Report:\n", report)


def plot_training_data(blur_errors, double_errors, has_blur, has_double):
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Plot for blur detection
    plt.figure(figsize=(10, 6))
    plt.scatter(
        blur_errors,
        double_errors,
        c=has_blur,
        cmap="viridis",
        label="Has Blur (1=Yes, 0=No)",
    )
    plt.xlabel("Blur Errors")
    plt.ylabel("Double Errors")
    plt.title("Training Data Visualization (Blur Detection)")
    plt.legend()
    plt.grid(True)
    blur_plot_path = os.path.join(plots_dir, "training_blur.png")
    plt.savefig(blur_plot_path)
    plt.close()

    # Plot for double vision
    plt.figure(figsize=(10, 6))
    plt.scatter(
        blur_errors,
        double_errors,
        c=has_double,
        cmap="plasma",
        label="Has Double Vision (1=Yes, 0=No)",
    )
    plt.xlabel("Blur Errors")
    plt.ylabel("Double Errors")
    plt.title("Training Data Visualization (Double Vision)")
    plt.legend()
    plt.grid(True)
    double_plot_path = os.path.join(plots_dir, "training_double.png")
    plt.savefig(double_plot_path)
    plt.close()

    print(f"Saved training plots to: {blur_plot_path}, {double_plot_path}")


def plot_datasets_visualization(blur_df, double_df, plots_dir="plots"):
    """Create a set of diagnostic plots for blur and double datasets.

    Saved files:
      - plots/blur_hist.png
      - plots/blur_counts.png
      - plots/double_scatter.png
      - plots/double_histograms.png

    The function is defensive: it checks for expected columns and skips plots
    if data is missing.
    """
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # ---- Blur dataset visualizations ----
    try:
        if blur_df is not None and len(blur_df) > 0:
            # ensure column exists
            if "blur_errors" in blur_df.columns:
                plt.figure(figsize=(8, 5))
                for label in sorted(
                    blur_df.get("has_blur", pd.Series([-1] * len(blur_df))).unique()
                ):
                    if label not in (0, 1):
                        continue
                    subset = blur_df[blur_df.get("has_blur") == label]
                    plt.hist(
                        subset["blur_errors"],
                        alpha=0.6,
                        bins=range(0, 7),
                        label=f"has_blur={label}",
                    )
                plt.xlabel("blur_errors")
                plt.ylabel("count")
                plt.title("Blur errors distribution by class")
                plt.legend()
                blur_hist_path = os.path.join(plots_dir, "blur_hist.png")
                plt.savefig(blur_hist_path)
                plt.close()

            # class counts
            if "has_blur" in blur_df.columns:
                counts = blur_df["has_blur"].value_counts().sort_index()
                plt.figure(figsize=(5, 4))
                counts.plot(kind="bar", color=["#2ca02c", "#d62728"])
                plt.xlabel("has_blur")
                plt.ylabel("count")
                plt.title("Blur dataset class counts")
                blur_counts_path = os.path.join(plots_dir, "blur_counts.png")
                plt.tight_layout()
                plt.savefig(blur_counts_path)
                plt.close()
    except Exception as e:
        logging.warning("Failed to create blur plots: %s", e)

    # ---- Double dataset visualizations ----
    try:
        if double_df is not None and len(double_df) > 0:
            # scatter: blur_errors vs double_errors colored by has_double
            if (
                "blur_errors" in double_df.columns
                and "double_errors" in double_df.columns
            ):
                plt.figure(figsize=(8, 6))
                colors = double_df.get("has_double", 0)
                plt.scatter(
                    double_df["blur_errors"],
                    double_df["double_errors"],
                    c=colors,
                    cmap="bwr",
                    alpha=0.6,
                )
                plt.colorbar(label="has_double")
                plt.xlabel("blur_errors")
                plt.ylabel("double_errors")
                plt.title("Blur vs Double errors (colored by has_double)")
                double_scatter_path = os.path.join(plots_dir, "double_scatter.png")
                plt.savefig(double_scatter_path)
                plt.close()

            # histograms per class for both features
            cols = []
            if "blur_errors" in double_df.columns:
                cols.append("blur_errors")
            if "double_errors" in double_df.columns:
                cols.append("double_errors")

            if cols:
                fig, axes = plt.subplots(len(cols), 1, figsize=(8, 4 * len(cols)))
                if len(cols) == 1:
                    axes = [axes]
                for ax, col in zip(axes, cols):
                    for label in sorted(
                        double_df.get(
                            "has_double", pd.Series([-1] * len(double_df))
                        ).unique()
                    ):
                        if label not in (0, 1):
                            continue
                        subset = double_df[double_df.get("has_double") == label]
                        ax.hist(
                            subset[col],
                            alpha=0.6,
                            bins=range(0, 7),
                            label=f"has_double={label}",
                        )
                    ax.set_xlabel(col)
                    ax.set_ylabel("count")
                    ax.set_title(f"Distribution of {col} by has_double")
                    ax.legend()
                double_hist_path = os.path.join(plots_dir, "double_histograms.png")
                plt.tight_layout()
                plt.savefig(double_hist_path)
                plt.close()
    except Exception as e:
        logging.warning("Failed to create double plots: %s", e)

    print(f"Saved dataset visualizations to folder: {plots_dir}")


def load_models(model_dir):
    model_blur_path = os.path.join(model_dir, "model_blur.joblib")
    model_double_path = os.path.join(model_dir, "model_double.joblib")
    model_blur = (
        joblib.load(model_blur_path) if os.path.exists(model_blur_path) else None
    )
    model_double = (
        joblib.load(model_double_path) if os.path.exists(model_double_path) else None
    )
    return model_blur, model_double


############ UNUSED FUNCTIONS ############
# def train_model(
#     X,
#     y_blur,
#     y_double,
#     test_size=0.2,
#     random_state=42,
#     n_estimators=10,
#     persist_dir=None,
#     verbose=False,
# ):
#     if verbose:
#         logging.info(
#             "Splitting data: test_size=%s, random_state=%s", test_size, random_state
#         )

#     (
#         X_train,
#         X_test,
#         y_train_blur,
#         y_test_blur,
#         y_train_double,
#         y_test_double,
#     ) = train_test_split(
#         X, y_blur, y_double, test_size=test_size, random_state=random_state
#     )

#     model_blur = RandomForestClassifier(
#         n_estimators=n_estimators, random_state=random_state
#     )
#     model_blur.fit(X_train, y_train_blur)

#     model_double = RandomForestClassifier(
#         n_estimators=n_estimators, random_state=random_state
#     )
#     model_double.fit(X_train, y_train_double)

#     if persist_dir:
#         try:
#             os.makedirs(persist_dir, exist_ok=True)
#             blur_path = os.path.join(persist_dir, "model_blur.joblib")
#             double_path = os.path.join(persist_dir, "model_double.joblib")
#             joblib.dump(model_blur, blur_path)
#             joblib.dump(model_double, double_path)
#             if verbose:
#                 logging.info("Saved models to: %s, %s", blur_path, double_path)
#         except Exception as e:
#             logging.warning("Failed to persist models: %s", e)

#     return model_blur, model_double, X_test, y_test_blur, y_test_double


# def get_training_data(
#     use_cnn: bool = False, cnn_batch_size: int = 32, verbose: bool = False
# ):
#     """Build training data from images folder.

#     Behavior:
#     - Walks the `images/blur` subfolder and treats files whose name contains
#       "clear" as non-blurred (has_blur=0) and others (e.g. containing "fog")
#       as blurred (has_blur=1).
#     - Computes a simple blur metric (variance of Laplacian) for each image and
#       scales it into a discrete `blur_errors` value in range [0,5].
#     - Optionally creates an augmented "double vision" sample for each image
#       (using `generate_double_vision_image`) and labels that sample as
#       has_double=1 while the original is has_double=0. The `double_errors`
#       column is set to 0 for originals and to a fixed value for augmented
#       double samples to provide a clear signal for the classifier.

#     Falls back to the small hard-coded toy dataset if no images are found.
#     Returns: X (DataFrame of features), y_blur (Series), y_double (Series), df (full DataFrame)
#     """

#     images_base = "images"
#     blur_dir = os.path.join(images_base, "blur")

#     records = []

#     if os.path.exists(blur_dir) and os.path.isdir(blur_dir):
#         # collect blur metric for normalization
#         var_list = []
#         paths = []
#         labels_blur = []
#         for fname in os.listdir(blur_dir):
#             if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 continue
#             path = os.path.join(blur_dir, fname)
#             img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#             if img is None:
#                 continue
#             # ground-truth label: assume filenames containing 'clear' are non-blurred
#             has_blur = 0 if "clear" in fname.lower() else 1
#             # variance of Laplacian as blur metric (higher -> sharper)
#             var_lap = cv2.Laplacian(img, cv2.CV_64F).var()
#             var_list.append(var_lap)
#             paths.append(path)
#             labels_blur.append(has_blur)

#         if len(var_list) == 0:
#             # fallback to toy data
#             return get_training_data_fallback()

#         vmin = min(var_list)
#         vmax = max(var_list)
#         vrange = vmax - vmin if vmax > vmin else 1.0

#         # create original samples (has_double = 0)
#         for path, var_lap, has_blur in zip(paths, var_list, labels_blur):
#             # scale var_lap into 0..5 but invert so that more blur => higher errors
#             norm = (var_lap - vmin) / vrange
#             blur_errors = int(round(5 * (1 - norm)))
#             records.append(
#                 {
#                     "blur_errors": blur_errors,
#                     "double_errors": 0,  # original images treated as not-double
#                     "has_blur": has_blur,
#                     "has_double": 0,
#                     "path": path,
#                 }
#             )
#             # create an augmented double-vision sample for this image
#             # double_img = generate_double_vision_image(path)
#             # if double_img is not None:
#             #     # use a fixed double error signal for augmented doubles
#             #     records.append(
#             #         {
#             #             "blur_errors": blur_errors,
#             #             "double_errors": 4,
#             #             "has_blur": has_blur,
#             #             "has_double": 1,
#             #             "path": path + "::double",
#             #         }
#             #     )
#             # save the augmented image
#             # double_img_path = path.replace(".jpg", "_double.jpg")
#             # cv2.imwrite(double_img_path, double_img)

#         df = pd.DataFrame(records)

#         # If requested, try to extract CNN embeddings as features (optional).
#         # if use_cnn:
#         #     try:
#         #         if verbose:
#         #             logging.info(
#         #                 "Attempting to extract CNN features using TensorFlow/Keras"
#         #             )
#         #         from tensorflow.keras.applications import MobileNetV2
#         #         from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#         #         from tensorflow.keras.preprocessing import image as kimage

#         #         base = MobileNetV2(weights="imagenet", include_top=False, pooling="avg")
#         #         paths = []
#         #         for p in df["path"]:
#         #             if "::double" in str(p):
#         #                 paths.append(p.split("::double")[0])
#         #             else:
#         #                 paths.append(p)

#         #         features = []
#         #         for i in range(0, len(paths), cnn_batch_size):
#         #             batch_paths = paths[i : i + cnn_batch_size]
#         #             imgs = []
#         #             for p in batch_paths:
#         #                 img = kimage.load_img(p, target_size=(224, 224))
#         #                 arr = kimage.img_to_array(img)
#         #                 imgs.append(arr)
#         #             x = np.array(imgs)
#         #             x = preprocess_input(x)
#         #             feats = base.predict(x, verbose=0)
#         #             features.append(feats)
#         #         features = np.vstack(features)

#         #         feat_cols = [f"f_{i}" for i in range(features.shape[1])]
#         #         X = pd.DataFrame(features, columns=feat_cols)
#         #         y_blur = df["has_blur"].reset_index(drop=True)
#         #         y_double = df["has_double"].reset_index(drop=True)
#         #         return X, y_blur, y_double, df
#         #     except Exception as e:
#         #         logging.warning(
#         #             "Could not extract CNN features (TensorFlow may be missing). Falling back to handcrafted features. Error: %s",
#         #             e,
#         #         )

#         X = df[["blur_errors", "double_errors"]]
#         y_blur = df["has_blur"]
#         y_double = df["has_double"]

#         return X, y_blur, y_double, df
#     else:
#         # fallback to the original toy dataset
#         return get_training_data_fallback()


# def generate_blurred_image(image_path, blur_level=5):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return None  # Return None or a placeholder if image loading fails
#     # Ensure blur_level is odd and greater than 0 for GaussianBlur kernel size
#     ksize = (blur_level, blur_level)
#     if ksize[0] % 2 == 0 or ksize[1] % 2 == 0 or ksize[0] <= 0 or ksize[1] <= 0:
#         print(f"Warning: Invalid blur level {blur_level}. Using default blur level 5.")
#         ksize = (5, 5)
#     blurred = cv2.GaussianBlur(img, ksize, 0)
#     return blurred


# def generate_double_vision_image(image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Error: Unable to load image from {image_path}")
#         return None
#     # إنشاء تأثير ازدواجية بسيط (نسخ الصورة وإزاحتها قليلاً)
#     shifted = np.roll(img, 10, axis=1)  # إزاحة أفقية
#     double_img = cv2.addWeighted(img, 0.5, shifted, 0.5, 0)
#     return double_img


# def evaluate_models(model_blur, model_double, X_test, y_test_blur, y_test_double):
#     # Evaluate models
#     y_pred_blur = model_blur.predict(X_test)
#     y_pred_double = model_double.predict(X_test)

#     accuracy_blur = accuracy_score(y_test_blur, y_pred_blur)
#     accuracy_double = accuracy_score(y_test_double, y_pred_double)

#     report_blur = classification_report(y_test_blur, y_pred_blur)
#     report_double = classification_report(y_test_double, y_pred_double)

#     print("Blur Detection Model Accuracy:", accuracy_blur)
#     print("Blur Detection Model Classification Report:\n", report_blur)

#     print("Double Vision Model Accuracy:", accuracy_double)
#     print("Double Vision Model Classification Report:\n", report_double)


# def create_dummy_image(folder_name, width=100, height=100, color=(255, 255, 255)):
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)

#     # Create dummy image files inside the 'images' folder with relevant names
#     dummy_image_path1 = os.path.join(folder_name, "road_clear.jpg")
#     dummy_image_path2 = os.path.join(folder_name, "road_blurred.jpg")
#     dummy_image_path3 = os.path.join(folder_name, "road_double.png")

#     # Create dummy images (e.g., white images)
#     # dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
#     dummy_image = np.ones((height, width, 3), dtype=np.uint8) * np.array(
#         color, dtype=np.uint8
#     )

#     cv2.imwrite(dummy_image_path1, dummy_image)
#     cv2.imwrite(dummy_image_path2, dummy_image)
#     cv2.imwrite(dummy_image_path3, dummy_image)

#     print(
#         f"Created '{folder_name}' folder and dummy image files: {dummy_image_path1}, {dummy_image_path2}, {dummy_image_path3}"
#     )


# def get_blur_test_image_paths(images_folder):
#     test_images, blur_levels = [], []
#     for filename in os.listdir(images_folder):
#         test_images.append(os.path.join(images_folder, filename))
#         blur_levels.append(1 if "clear" in filename else 5)
#     return test_images, blur_levels


# def get_double_test_image_paths(images_folder):
#     test_images = []
#     return test_images


# def blury_vision_test(st, images, blur_levels):
#     st.header("اختبار 1: الرؤية الضبابية")
#     st.write("اختر الصور الواضحة من المشوشة.")

#     user_choices_blur = []
#     for i, img_path in enumerate(images):
#         img = generate_blurred_image(img_path, blur_levels[i])
#         if img is not None:  # Check if image was loaded successfully
#             st.image(img, caption=f"صورة {i+1}", width=200)
#             choice = st.radio(
#                 f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
#             )
#             user_choices_blur.append(
#                 1 if choice == "لا" and blur_levels[i] > 0 else 0
#             )  # 1 إذا أخطأ
#         else:
#             st.write(f"Could not load image: {img_path}")

#     return sum(user_choices_blur)


# def double_vision_test(st, images):
#     st.header("اختبار 2: الازدواجية في الرؤية")
#     st.write("حدد إذا كانت الصورة تبدو مزدوجة.")

#     user_choices_double = []
#     for i, img_path in enumerate(images):
#         img = (
#             generate_double_vision_image(img_path)
#             if i % 2 == 1
#             else cv2.imread(img_path)
#         )  # كل صورة ثانية مزدوجة
#         if img is not None:
#             st.image(img, caption=f"صورة {i+1}", width=200)
#             choice = st.radio(
#                 f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})",
#                 ["نعم", "لا"],
#                 key=f"double_{i}",
#             )
#             user_choices_double.append(
#                 1 if choice == "نعم" and i % 2 == 1 else 0
#             )  # 1 إذا أخطأ
#         else:
#             st.write(f"Could not load image: {img_path}")

#     return sum(user_choices_double)


# def single_image_test(st, uploaded_file, model_blur, model_double):
#     if uploaded_file is not None:
#         # Read the uploaded image
#         file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#         img = cv2.imdecode(file_bytes, 1)

#         # Convert BGR image to RGB for Streamlit display
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         st.image(img_rgb, caption="الصورة المحملة", width=200)

#         st.write("Analyzing the uploaded image...")

#         # *** IMPORTANT: Image analysis functions would go here in a real application ***
#         # For this example, we'll use dummy error counts based on a simple check
#         # Replace this with actual image processing to determine blur and double vision levels
#         # Dummy analysis: check if image looks mostly uniform (like the dummy ones)
#         # This is a very basic check and not a real image analysis for blur/double vision

#         if np.mean(img) > 200:  # Assuming a white or very light dummy image
#             image_blur_errors = 0
#             image_double_errors = 0
#         else:  # Assume a more complex image might have errors (for demo purposes)
#             image_blur_errors = 2  # Placeholder value
#             image_double_errors = 3  # Placeholder value

#         input_data_single = pd.DataFrame(
#             [[image_blur_errors, image_double_errors]],
#             columns=["blur_errors", "double_errors"],
#         )
#         pred_blur_single = model_blur.predict(input_data_single)[0]
#         pred_double_single = model_double.predict(input_data_single)[0]

#         st.subheader("نتائج التوقع للصورة:")
#         if pred_blur_single == 1:
#             st.write("احتمال وجود رؤية ضبابية في هذه الصورة (بناءً على analysis مبسط).")
#         else:
#             st.write(
#                 "لا يوجد دليل على رؤية ضبابية في هذه الصورة (بناءً على analysis مبسط)."
#             )

#         if pred_double_single == 1:
#             st.write(
#                 "احتمال وجود ازدواجية في الرؤية في هذه الصورة (بناءً على analysis مبسط)."
#             )
#         else:
#             st.write(
#                 "لا يوجد دليل على ازدواجية في الرؤية في هذه الصورة (بناءً على analysis مبسط)."
#             )


# def analyze_results(
#     st,
#     model_blur,
#     model_double,
#     blur_errors,
#     double_errors,
# ):
#     if st.button("احسب النتائج"):
#         # Ensure input to predict is a 2D array/DataFrame
#         input_data = pd.DataFrame(
#             [[blur_errors, double_errors]], columns=["blur_errors", "double_errors"]
#         )
#         pred_blur = model_blur.predict(input_data)[0]
#         pred_double = model_double.predict(input_data)[0]

#         st.subheader("النتائج من النموذج AI:")
#         if pred_blur == 1:
#             st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
#         else:
#             st.write("لا يوجد دليل على رؤية ضبابية.")

#         if pred_double == 1:
#             st.write(
#                 "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
#             )
#         else:
#             st.write("لا يوجد دليل على ازدواجية في الرؤية.")

#         st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
#         st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")
