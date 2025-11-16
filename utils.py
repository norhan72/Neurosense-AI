from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def generate_blurred_image(image_path, blur_level=5):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return None  # Return None or a placeholder if image loading fails
    # Ensure blur_level is odd and greater than 0 for GaussianBlur kernel size
    ksize = (blur_level, blur_level)
    if ksize[0] % 2 == 0 or ksize[1] % 2 == 0 or ksize[0] <= 0 or ksize[1] <= 0:
        print(f"Warning: Invalid blur level {blur_level}. Using default blur level 5.")
        ksize = (5, 5)
    blurred = cv2.GaussianBlur(img, ksize, 0)
    return blurred


def generate_double_vision_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return None
    # إنشاء تأثير ازدواجية بسيط (نسخ الصورة وإزاحتها قليلاً)
    shifted = np.roll(img, 10, axis=1)  # إزاحة أفقية
    double_img = cv2.addWeighted(img, 0.5, shifted, 0.5, 0)
    return double_img


def create_dummy_image(folder_name, width=100, height=100, color=(255, 255, 255)):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # Create dummy image files inside the 'images' folder with relevant names
    dummy_image_path1 = os.path.join(folder_name, "road_clear.jpg")
    dummy_image_path2 = os.path.join(folder_name, "road_blurred.jpg")
    dummy_image_path3 = os.path.join(folder_name, "road_double.png")

    # Create dummy images (e.g., white images)
    # dummy_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    dummy_image = np.ones((height, width, 3), dtype=np.uint8) * np.array(
        color, dtype=np.uint8
    )

    cv2.imwrite(dummy_image_path1, dummy_image)
    cv2.imwrite(dummy_image_path2, dummy_image)
    cv2.imwrite(dummy_image_path3, dummy_image)

    print(
        f"Created '{folder_name}' folder and dummy image files: {dummy_image_path1}, {dummy_image_path2}, {dummy_image_path3}"
    )


def get_training_data():
    # بيانات تدريب افتراضية للنموذج (مثال: نتائج اختبارات سابقة)
    # X: عدد الأخطاء في الاختبارات، y: وجود مشكلة (0=لا، 1=نعم)
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


def train_model(X, y_blur, y_double):
    # تدريب النماذج
    X_train, X_test, y_train_blur, y_test_blur = train_test_split(
        X, y_blur, test_size=0.2, random_state=42
    )
    model_blur = RandomForestClassifier(n_estimators=10, random_state=42)
    model_blur.fit(X_train, y_train_blur)

    X_train, X_test, y_train_double, y_test_double = train_test_split(
        X, y_double, test_size=0.2, random_state=42
    )
    model_double = RandomForestClassifier(n_estimators=10, random_state=42)
    model_double.fit(X_train, y_train_double)
    return model_blur, model_double, X_test, y_test_blur, y_test_double


def evaluate_models(model_blur, model_double, X_test, y_test_blur, y_test_double):
    # Evaluate models
    y_pred_blur = model_blur.predict(X_test)
    y_pred_double = model_double.predict(X_test)

    accuracy_blur = accuracy_score(y_test_blur, y_pred_blur)
    accuracy_double = accuracy_score(y_test_double, y_pred_double)

    report_blur = classification_report(y_test_blur, y_pred_blur)
    report_double = classification_report(y_test_double, y_pred_double)

    print("Blur Detection Model Accuracy:", accuracy_blur)
    print("Blur Detection Model Classification Report:\n", report_blur)

    print("Double Vision Model Accuracy:", accuracy_double)
    print("Double Vision Model Classification Report:\n", report_double)


def plot_training_data(blur_errors, double_errors, has_blur, has_double):
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
    plt.show()
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
    plt.show()


def get_test_image_paths(images_folder):
    test_images = []
    for filename in os.listdir(images_folder):
        if "road" in filename and (
            filename.endswith(".jpg") or filename.endswith(".png")
        ):
            test_images.append(os.path.join(images_folder, filename))
    blur_levels = [1 if "clear" in img else 5 for img in test_images]
    return test_images, blur_levels


def blury_vision_test(st, images, blur_levels):
    st.header("اختبار 1: الرؤية الضبابية")
    st.write("اختر الصور الواضحة من المشوشة.")

    user_choices_blur = []
    for i, img_path in enumerate(images):
        img = generate_blurred_image(img_path, blur_levels[i])
        if img is not None:  # Check if image was loaded successfully
            st.image(img, caption=f"صورة {i+1}", width=200)
            choice = st.radio(
                f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
            )
            user_choices_blur.append(
                1 if choice == "لا" and blur_levels[i] > 0 else 0
            )  # 1 إذا أخطأ
        else:
            st.write(f"Could not load image: {img_path}")

    return sum(user_choices_blur)


def double_vision_test(st, images):
    st.header("اختبار 2: الازدواجية في الرؤية")
    st.write("حدد إذا كانت الصورة تبدو مزدوجة.")

    user_choices_double = []
    for i, img_path in enumerate(images):
        img = (
            generate_double_vision_image(img_path)
            if i % 2 == 1
            else cv2.imread(img_path)
        )  # كل صورة ثانية مزدوجة
        if img is not None:
            st.image(img, caption=f"صورة {i+1}", width=200)
            choice = st.radio(
                f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})",
                ["نعم", "لا"],
                key=f"double_{i}",
            )
            user_choices_double.append(
                1 if choice == "نعم" and i % 2 == 1 else 0
            )  # 1 إذا أخطأ
        else:
            st.write(f"Could not load image: {img_path}")

    return sum(user_choices_double)


def single_image_test(st, uploaded_file, model_blur, model_double):
    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        # Convert BGR image to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="الصورة المحملة", width=200)

        st.write("Analyzing the uploaded image...")

        # *** IMPORTANT: Image analysis functions would go here in a real application ***
        # For this example, we'll use dummy error counts based on a simple check
        # Replace this with actual image processing to determine blur and double vision levels
        # Dummy analysis: check if image looks mostly uniform (like the dummy ones)
        # This is a very basic check and not a real image analysis for blur/double vision

        if np.mean(img) > 200:  # Assuming a white or very light dummy image
            image_blur_errors = 0
            image_double_errors = 0
        else:  # Assume a more complex image might have errors (for demo purposes)
            image_blur_errors = 2  # Placeholder value
            image_double_errors = 3  # Placeholder value

        input_data_single = pd.DataFrame(
            [[image_blur_errors, image_double_errors]],
            columns=["blur_errors", "double_errors"],
        )
        pred_blur_single = model_blur.predict(input_data_single)[0]
        pred_double_single = model_double.predict(input_data_single)[0]

        st.subheader("نتائج التوقع للصورة:")
        if pred_blur_single == 1:
            st.write("احتمال وجود رؤية ضبابية في هذه الصورة (بناءً على analysis مبسط).")
        else:
            st.write(
                "لا يوجد دليل على رؤية ضبابية في هذه الصورة (بناءً على analysis مبسط)."
            )

        if pred_double_single == 1:
            st.write(
                "احتمال وجود ازدواجية في الرؤية في هذه الصورة (بناءً على analysis مبسط)."
            )
        else:
            st.write(
                "لا يوجد دليل على ازدواجية في الرؤية في هذه الصورة (بناءً على analysis مبسط)."
            )


def analyze_results(
    st,
    model_blur,
    model_double,
    blur_errors,
    double_errors,
):
    if st.button("احسب النتائج"):
        # Ensure input to predict is a 2D array/DataFrame
        input_data = pd.DataFrame(
            [[blur_errors, double_errors]], columns=["blur_errors", "double_errors"]
        )
        pred_blur = model_blur.predict(input_data)[0]
        pred_double = model_double.predict(input_data)[0]

        st.subheader("النتائج من النموذج AI:")
        if pred_blur == 1:
            st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
        else:
            st.write("لا يوجد دليل على رؤية ضبابية.")

        if pred_double == 1:
            st.write(
                "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
            )
        else:
            st.write("لا يوجد دليل على ازدواجية في الرؤية.")

        st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
        st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")
