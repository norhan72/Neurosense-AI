import os
import pandas as pd
from utils import (
    evaluate_classifier,
    build_blur_dataset,
    build_double_dataset,
    plot_datasets_visualization,
    load_models,
    plot_training_data,
    train_classifier,
)


def run_training(
    test_size=0.2, random_state=42, n_estimators=10, persist_dir="models", verbose=True
):
    if verbose:
        print("Building dataset from images...")

    # Build datasets separately
    if verbose:
        print("Building blur dataset...")
    Xb, yb, dfb = build_blur_dataset()

    # Visualize blur dataset
    try:
        plot_datasets_visualization(dfb, None)
    except Exception:
        pass

    if verbose:
        print(f"Blur dataset: X.shape={Xb.shape}, samples={len(dfb)}")

    if verbose:
        print("Training blur classifier...")
    blur_model, Xb_test, yb_test = train_classifier(
        Xb,
        yb,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        persist_path=(
            os.path.join(persist_dir, "model_blur.joblib") if persist_dir else None
        ),
        verbose=verbose,
    )

    if verbose:
        print("Evaluating blur classifier...")
    evaluate_classifier(blur_model, Xb_test, yb_test, name="Blur Model")

    # Double dataset and classifier
    if verbose:
        print("Building double dataset...")
    Xd, yd, dfd, gd = build_double_dataset()

    # Visualize both datasets together (double_df may include blur_errors)
    try:
        plot_datasets_visualization(dfb, dfd)
    except Exception:
        pass

    if verbose:
        print(f"Double dataset: X.shape={Xd.shape}, samples={len(dfd)}")

    if verbose:
        print("Training double classifier...")
    double_model, Xd_test, yd_test = train_classifier(
        Xd,
        yd,
        test_size=test_size,
        random_state=random_state,
        n_estimators=n_estimators,
        persist_path=(
            os.path.join(persist_dir, "model_double.joblib") if persist_dir else None
        ),
        verbose=verbose,
        groups=gd,
    )

    if verbose:
        print("Evaluating double classifier...")
    evaluate_classifier(double_model, Xd_test, yd_test, name="Double Model")

    # # Create combined df for plotting (fill missing columns)
    # try:
    #     df_combined = pd.concat(
    #         [
    #             dfb.reindex(
    #                 columns=["blur_errors", "double_errors", "has_blur", "has_double"]
    #             ).fillna({"double_errors": 0, "has_double": 0}),
    #             dfd.reindex(
    #                 columns=["blur_errors", "double_errors", "has_blur", "has_double"]
    #             ).fillna({"blur_errors": 0, "has_blur": 0}),
    #         ],
    #         ignore_index=True,
    #     )
    #     plot_training_data(
    #         df_combined["blur_errors"],
    #         df_combined["double_errors"],
    #         df_combined.get("has_blur", pd.Series([0] * len(df_combined))),
    #         df_combined.get("has_double", pd.Series([0] * len(df_combined))),
    #     )
    # except Exception:
    #     print("Exception in plotting training data:", exc_info=True)

    return blur_model, double_model, dfb, dfd


# If run as a script (not imported), run training once with defaults
if __name__ == "__main__":
    model_blur, model_double = None, None
    print("Loading trained models...")
    model_blur, model_double = load_models("models")
    if not model_blur or not model_double:
        print("Failed to load models, re-training...")
        model_blur, model_double, dfb, dfd = run_training()
    print("Models are ready.")
    print("Blur Model:", model_blur, "Double Model:", model_double)

    # Even if models are loaded, create dataset visualizations so we can inspect
    # the training data without forcing a retrain.
    try:
        Xb, yb, dfb = build_blur_dataset()
        Xd, yd, dfd, gd = build_double_dataset()
        plot_datasets_visualization(dfb, dfd)
    except Exception:
        pass


# 2. Load blur test images and run the test (not needed)
# images_folder = "images"
# blur_images, blur_levels = get_blur_test_image_paths(images_folder + "/blur")
# st.title("اختبار الاستدراك البصري - AI Model")
# blur_errors = blury_vision_test(st, blur_images, blur_levels)

# 3. Load double vision test images and run the test (not needed)
# double_images = get_double_test_image_paths(images_folder + "/double")
# double_errors = double_vision_test(st, double_images)

# # 4. Analyze results using trained models
# analyze_results(st, model_blur, model_double, blur_errors, double_errors)

# # 5. Run individual image prediction
# st.header("توقع حالة صورة فردية")
# uploaded_file = st.file_uploader("اختر صورة لتحليلها:", type=["jpg", "png"])
# single_image_test(st, uploaded_file, model_blur, model_double)
