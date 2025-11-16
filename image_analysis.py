import streamlit as st
from utils import (
    analyze_results,
    blury_vision_test,
    double_vision_test,
    evaluate_models,
    get_test_image_paths,
    get_training_data,
    plot_training_data,
    single_image_test,
    train_model,
)

images_folder = "images"

# 1. Train models, evaluate them, and plot the training data before running the tests
X, y_blur, y_double, df = get_training_data()
model_blur, model_double, X_test, y_test_blur, y_test_double = train_model(
    X, y_blur, y_double
)
evaluate_models(model_blur, model_double, X_test, y_test_blur, y_test_double)
plot_training_data(df["blur_errors"], df["double_errors"], y_blur, y_double)

# 2. Load test images
images, blur_levels = get_test_image_paths(images_folder)

# 3. Run the two vision tests
st.title("اختبار الاستدراك البصري - AI Model")
blur_errors = blury_vision_test(st, images, blur_levels)
double_errors = double_vision_test(st, images)

# 4. Analyze results using trained models
analyze_results(st, model_blur, model_double, blur_errors, double_errors)

# 5. Run individual image prediction
st.header("توقع حالة صورة فردية")
uploaded_file = st.file_uploader("اختر صورة لتحليلها:", type=["jpg", "png"])
single_image_test(st, uploaded_file, model_blur, model_double)


# 5. Create dummy image files (for the Streamlit app)
# create_dummy_image(images_folder, 100, 100, (255, 255, 255))

# 6. Write the complete Streamlit UI
# st.title("اختبار الاستدراك البصري - AI Model")

# اختبار الضبابية
# st.header("اختبار 1: الرؤية الضبابية")
# st.write("اختر الصور الواضحة من المشوشة.")

# عرض صور (using dummy images created in the notebook)
# images, blur_levels = get_test_image_paths(images_folder)
# user_choices_blur = []

# for i, img_path in enumerate(images):
#     img = generate_blurred_image(img_path, blur_levels[i])
#     if img is not None:  # Check if image was loaded successfully
#         # Convert BGR image to RGB for Streamlit display
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         st.image(img_rgb, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
#         )
#         # 1 if the user chose "لا" (No) and the image was intentionally blurred (>1 blur level)
#         user_choices_blur.append(1 if choice == "لا" and blur_levels[i] > 1 else 0)
#     else:
#         st.write(f"Could not load image: {img_path}")


# blur_errors = sum(user_choices_blur)

# # اختبار الازدواجية
# st.header("اختبار 2: الازدواجية في الرؤية")
# st.write("حدد إذا كانت الصورة تبدو مزدوجة.")

# user_choices_double = []
# for i, img_path in enumerate(images):
#     # Generate double vision for images containing 'double' in filename or based on index
#     is_double_image = (
#         "double" in img_path.lower() or i % 2 == 1
#     )  # Condition for double image
#     img = (
#         generate_double_vision_image(img_path)
#         if is_double_image
#         else cv2.imread(img_path)
#     )

#     if img is not None:
#         # Convert BGR image to RGB for Streamlit display
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         st.image(img_rgb, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})", ["نعم", "لا"], key=f"double_{i}"
#         )
#         # 1 if the user chose "نعم" (Yes) and it was a double image
#         user_choices_double.append(1 if choice == "نعم" and is_double_image else 0)
#     else:
#         st.write(f"Could not load image: {img_path}")

# double_errors = sum(user_choices_double)

# # تحليل باستخدام AI
# if st.button("احسب النتائج"):
#     # Ensure input to predict is a 2D array/DataFrame
#     input_data = pd.DataFrame(
#         [[blur_errors, double_errors]], columns=["blur_errors", "double_errors"]
#     )
#     pred_blur = model_blur.predict(input_data)[0]
#     pred_double = model_double.predict(input_data)[0]

#     st.subheader("النتائج من النموذج AI:")
#     if pred_blur == 1:
#         st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
#     else:
#         st.write("لا يوجد دليل على رؤية ضبابية.")

#     if pred_double == 1:
#         st.write(
#             "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
#         )
#     else:
#         st.write("لا يوجد دليل على ازدواجية في الرؤية.")

#     st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
#     st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")


# # بيانات تدريب افتراضية للنموذج (مثال: نتائج اختبارات سابقة)
# # X: عدد الأخطاء في الاختبارات، y: وجود مشكلة (0=لا، 1=نعم)
# data = {
#     "blur_errors": [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 1, 2, 4, 5, 3, 2, 1, 0],
#     "double_errors": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0],
#     "has_blur": [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
#     "has_double": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
# }
# df = pd.DataFrame(data)
# X = df[["blur_errors", "double_errors"]]
# y_blur = df["has_blur"]
# y_double = df["has_double"]

# # تدريب النماذج
# X_train, X_test, y_train_blur, y_test_blur = train_test_split(
#     X, y_blur, test_size=0.2, random_state=42
# )
# model_blur = RandomForestClassifier(n_estimators=10, random_state=42)
# model_blur.fit(X_train, y_train_blur)

# X_train, X_test, y_train_double, y_test_double = train_test_split(
#     X, y_double, test_size=0.2, random_state=42
# )
# model_double = RandomForestClassifier(n_estimators=10, random_state=42)
# model_double.fit(X_train, y_train_double)

# # واجهة Streamlit
# st.title("اختبار الاستدراك البصري - AI Model")

# # اختبار الضبابية
# st.header("اختبار 1: الرؤية الضبابية")
# st.write("اختر الصور الواضحة من المشوشة.")

# # عرض صور (افتراضية: استخدم صورًا من مجلد 'images'، مثل cat.jpg)
# images = ["cat.jpg", "dog.jpg", "bird.jpg"]  # استبدل بمسارات صور حقيقية
# blur_levels = [1, 5, 3]  # Updated to use odd numbers > 0

# user_choices_blur = []
# for i, img_path in enumerate(images):
#     img = generate_blurred_image(img_path, blur_levels[i])
#     if img is not None:  # Check if image was loaded successfully
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
#         )
#         user_choices_blur.append(
#             1 if choice == "لا" and blur_levels[i] > 0 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")


# blur_errors = sum(user_choices_blur)

# # اختبار الازدواجية
# st.header("اختيار 2: الازدواجية في الرؤية")
# st.write("حدد إذا كانت الصورة تبدو مزدوجة.")

# user_choices_double = []
# for i, img_path in enumerate(images):
#     img = (
#         generate_double_vision_image(img_path) if i % 2 == 1 else cv2.imread(img_path)
#     )  # كل صورة ثانية مزدوجة
#     if img is not None:
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})", ["نعم", "لا"], key=f"double_{i}"
#         )
#         user_choices_double.append(
#             1 if choice == "نعم" and i % 2 == 1 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")

# double_errors = sum(user_choices_double)

# # تحليل باستخدام AI
# if st.button("احسب النتائج"):
#     # Ensure input to predict is a 2D array/DataFrame
#     input_data = pd.DataFrame(
#         [[blur_errors, double_errors]], columns=["blur_errors", "double_errors"]
#     )
#     pred_blur = model_blur.predict(input_data)[0]
#     pred_double = model_double.predict(input_data)[0]

#     st.subheader("النتائج من النموذج AI:")
#     if pred_blur == 1:
#         st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
#     else:
#         st.write("لا يوجد دليل على رؤية ضبابية.")

#     if pred_double == 1:
#         st.write(
#             "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
#         )
#     else:
#         st.write("لا يوجد دليل على ازدواجية في الرؤية.")

#     st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
#     st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")


# # بيانات تدريب افتراضية للنموذج (مثال: نتائج اختبارات سابقة)
# # X: عدد الأخطاء في الاختبارات، y: وجود مشكلة (0=لا، 1=نعم)
# data = {
#     "blur_errors": [0, 1, 2, 3, 4, 5],
#     "double_errors": [0, 1, 2, 3, 4, 5],
#     "has_blur": [0, 0, 1, 1, 1, 1],
#     "has_double": [0, 0, 1, 1, 1, 1],
# }
# df = pd.DataFrame(data)
# X = df[["blur_errors", "double_errors"]]
# y_blur = df["has_blur"]
# y_double = df["has_double"]

# # تدريب النماذج
# X_train, X_test, y_train_blur, y_test_blur = train_test_split(
#     X, y_blur, test_size=0.2, random_state=42
# )
# model_blur = RandomForestClassifier(n_estimators=10, random_state=42)
# model_blur.fit(X_train, y_train_blur)

# X_train, X_test, y_train_double, y_test_double = train_test_split(
#     X, y_double, test_size=0.2, random_state=42
# )
# model_double = RandomForestClassifier(n_estimators=10, random_state=42)
# model_double.fit(X_train, y_train_double)

# # واجهة Streamlit
# st.title("اختبار الاستدراك البصري - AI Model")

# # اختبار الضبابية
# st.header("اختبار 1: الرؤية الضبابية")
# st.write("اختر الصور الواضحة من المشوشة.")

# # عرض صور (افتراضية: استخدم صورًا من مجلد 'images'، مثل cat.jpg)
# images = ["cat.jpg", "dog.jpg", "bird.jpg"]  # استبدل بمسارات صور حقيقية
# blur_levels = [1, 5, 3]  # Updated to use odd numbers > 0

# user_choices_blur = []
# for i, img_path in enumerate(images):
#     img = generate_blurred_image(img_path, blur_levels[i])
#     if img is not None:  # Check if image was loaded successfully
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
#         )
#         user_choices_blur.append(
#             1 if choice == "لا" and blur_levels[i] > 0 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")


# blur_errors = sum(user_choices_blur)

# # اختبار الازدواجية
# st.header("اختيار 2: الازدواجية في الرؤية")
# st.write("حدد إذا كانت الصورة تبدو مزدوجة.")

# user_choices_double = []
# for i, img_path in enumerate(images):
#     img = (
#         generate_double_vision_image(img_path) if i % 2 == 1 else cv2.imread(img_path)
#     )  # كل صورة ثانية مزدوجة
#     if img is not None:
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})", ["نعم", "لا"], key=f"double_{i}"
#         )
#         user_choices_double.append(
#             1 if choice == "نعم" and i % 2 == 1 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")

# double_errors = sum(user_choices_double)

# # تحليل باستخدام AI
# if st.button("احسب النتائج"):
#     # Ensure input to predict is a 2D array/DataFrame
#     input_data = pd.DataFrame(
#         [[blur_errors, double_errors]], columns=["blur_errors", "double_errors"]
#     )
#     pred_blur = model_blur.predict(input_data)[0]
#     pred_double = model_double.predict(input_data)[0]

#     st.subheader("النتائج من النموذج AI:")
#     if pred_blur == 1:
#         st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
#     else:
#         st.write("لا يوجد دليل على رؤية ضبابية.")

#     if pred_double == 1:
#         st.write(
#             "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
#         )
#     else:
#         st.write("لا يوجد دليل على ازدواجية في الرؤية.")

#     st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
#     st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")


# # Create a scatter plot of the training data
# plt.figure(figsize=(10, 6))

# # Plot data points for 'has_blur'
# plt.scatter(
#     df["blur_errors"],
#     df["double_errors"],
#     c=df["has_blur"],
#     cmap="viridis",
#     label="Has Blur (1=Yes, 0=No)",
# )

# # Add labels and title
# plt.xlabel("Blur Errors")
# plt.ylabel("Double Errors")
# plt.title("Training Data Visualization")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(10, 6))
# # Plot data points for 'has_double'
# plt.scatter(
#     df["blur_errors"],
#     df["double_errors"],
#     c=df["has_double"],
#     cmap="plasma",
#     label="Has Double Vision (1=Yes, 0=No)",
# )

# # Add labels and title
# plt.xlabel("Blur Errors")
# plt.ylabel("Double Errors")
# plt.title("Training Data Visualization (Double Vision)")
# plt.legend()
# plt.grid(True)
# plt.show()

# # بيانات تدريب افتراضية للنموذج (مثال: نتائج اختبارات سابقة)
# # X: عدد الأخطاء في الاختبارات، y: وجود مشكلة (0=لا، 1=نعم)
# data = {
#     "blur_errors": [0, 1, 2, 3, 4, 5],
#     "double_errors": [0, 1, 2, 3, 4, 5],
#     "has_blur": [0, 0, 1, 1, 1, 1],
#     "has_double": [0, 0, 1, 1, 1, 1],
# }
# df = pd.DataFrame(data)
# X = df[["blur_errors", "double_errors"]]
# y_blur = df["has_blur"]
# y_double = df["has_double"]


# # تدريب النماذج
# X_train, X_test, y_train_blur, y_test_blur = train_test_split(
#     X, y_blur, test_size=0.2, random_state=42
# )
# model_blur = RandomForestClassifier(n_estimators=10, random_state=42)
# model_blur.fit(X_train, y_train_blur)

# X_train, X_test, y_train_double, y_test_double = train_test_split(
#     X, y_double, test_size=0.2, random_state=42
# )
# model_double = RandomForestClassifier(n_estimators=10, random_state=42)
# model_double.fit(X_train, y_train_double)


# # واجهة Streamlit
# st.title("اختبار الاستدراك البصري - AI Model")


# # اختبار الضبابية
# st.header("اختبار 1: الرؤية الضبابية")
# st.write("اختر الصور الواضحة من المشوشة.")

# # عرض صور (افتراضية: استخدم صورًا من مجلد 'images'، مثل cat.jpg)
# images = ["cat.jpg", "dog.jpg", "bird.jpg"]  # استبدل بمسارات صور حقيقية
# blur_levels = [1, 5, 3]  # 0=واضح، 5=مشوش

# user_choices_blur = []
# for i, img_path in enumerate(images):
#     img = generate_blurred_image(img_path, blur_levels[i])
#     if img is not None:  # Check if image was loaded successfully
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
#         )
#         user_choices_blur.append(
#             1 if choice == "لا" and blur_levels[i] > 0 else 0
#         )  # 1 إذا أخطأ
#     else:
#         print(f"Skipping image {img_path} due to loading error.")


# blur_errors = sum(user_choices_blur)

# # بيانات تدريب افتراضية للنموذج (مثال: نتائج اختبارات سابقة)
# # X: عدد الأخطاء في الاختبارات، y: وجود مشكلة (0=لا، 1=نعم)
# data = {
#     "blur_errors": [0, 1, 2, 3, 4, 5],
#     "double_errors": [0, 1, 2, 3, 4, 5],
#     "has_blur": [0, 0, 1, 1, 1, 1],
#     "has_double": [0, 0, 1, 1, 1, 1],
# }
# df = pd.DataFrame(data)
# X = df[["blur_errors", "double_errors"]]
# y_blur = df["has_blur"]
# y_double = df["has_double"]

# # تدريب النماذج
# X_train, X_test, y_train_blur, y_test_blur = train_test_split(
#     X, y_blur, test_size=0.2, random_state=42
# )
# model_blur = RandomForestClassifier(n_estimators=10, random_state=42)
# model_blur.fit(X_train, y_train_blur)

# X_train, X_test, y_train_double, y_test_double = train_test_split(
#     X, y_double, test_size=0.2, random_state=42
# )
# model_double = RandomForestClassifier(n_estimators=10, random_state=42)
# model_double.fit(X_train, y_train_double)

# # واجهة Streamlit
# st.title("اختبار الاستدراك البصري - AI Model")

# # اختبار الضبابية
# st.header("اختبار 1: الرؤية الضبابية")
# st.write("اختر الصور الواضحة من المشوشة.")

# # عرض صور (افتراضية: استخدم صورًا من مجلد 'images'، مثل cat.jpg)
# images = ["cat.jpg", "dog.jpg", "bird.jpg"]  # استبدل بمسارات صور حقيقية
# blur_levels = [1, 5, 3]  # Updated to use odd numbers > 0

# user_choices_blur = []
# for i, img_path in enumerate(images):
#     img = generate_blurred_image(img_path, blur_levels[i])
#     if img is not None:  # Check if image was loaded successfully
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
#         )
#         user_choices_blur.append(
#             1 if choice == "لا" and blur_levels[i] > 0 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")


# blur_errors = sum(user_choices_blur)

# # اختبار الازدواجية
# st.header("اختيار 2: الازدواجية في الرؤية")
# st.write("حدد إذا كانت الصورة تبدو مزدوجة.")

# user_choices_double = []
# for i, img_path in enumerate(images):
#     img = (
#         generate_double_vision_image(img_path) if i % 2 == 1 else cv2.imread(img_path)
#     )  # كل صورة ثانية مزدوجة
#     if img is not None:
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})", ["نعم", "لا"], key=f"double_{i}"
#         )
#         user_choices_double.append(
#             1 if choice == "نعم" and i % 2 == 1 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")

# double_errors = sum(user_choices_double)

# # تحليل باستخدام AI
# if st.button("احسب النتائج"):
#     # Ensure input to predict is a 2D array/DataFrame
#     input_data = pd.DataFrame(
#         [[blur_errors, double_errors]], columns=["blur_errors", "double_errors"]
#     )
#     pred_blur = model_blur.predict(input_data)[0]
#     pred_double = model_double.predict(input_data)[0]

#     st.subheader("النتائج من النموذج AI:")
#     if pred_blur == 1:
#         st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
#     else:
#         st.write("لا يوجد دليل على رؤية ضبابية.")

#     if pred_double == 1:
#         st.write(
#             "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
#         )
#     else:
#         st.write("لا يوجد دليل على ازدواجية في الرؤية.")

#     st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
#     st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")


# # بيانات تدريب افتراضية للنموذج (مثال: نتائج اختبارات سابقة)
# # X: عدد الأخطاء في الاختبارات، y: وجود مشكلة (0=لا، 1=نعم)
# data = {
#     "blur_errors": [0, 1, 2, 3, 4, 5, 1, 2, 3, 0, 1, 2, 4, 5, 3, 2, 1, 0],
#     "double_errors": [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0],
#     "has_blur": [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
#     "has_double": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
# }
# df = pd.DataFrame(data)
# X = df[["blur_errors", "double_errors"]]
# y_blur = df["has_blur"]
# y_double = df["has_double"]

# # تدريب النماذج
# X_train, X_test, y_train_blur, y_test_blur = train_test_split(
#     X, y_blur, test_size=0.2, random_state=42
# )
# model_blur = RandomForestClassifier(n_estimators=10, random_state=42)
# model_blur.fit(X_train, y_train_blur)

# X_train, X_test, y_train_double, y_test_double = train_test_split(
#     X, y_double, test_size=0.2, random_state=42
# )
# model_double = RandomForestClassifier(n_estimators=10, random_state=42)
# model_double.fit(X_train, y_train_double)

# # واجهة Streamlit
# st.title("اختبار الاستدراك البصري - AI Model")

# # اختبار الضبابية
# st.header("اختبار 1: الرؤية الضبابية")
# st.write("اختر الصور الواضحة من المشوشة.")

# # عرض صور (افتراضية: استخدم صورًا من مجلد 'images'، مثل cat.jpg)
# images = ["cat.jpg", "dog.jpg", "bird.jpg"]  # استبدل بمسارات صور حقيقية
# blur_levels = [1, 5, 3]  # Updated to use odd numbers > 0

# user_choices_blur = []
# for i, img_path in enumerate(images):
#     img = generate_blurred_image(img_path, blur_levels[i])
#     if img is not None:  # Check if image was loaded successfully
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة واضحة؟ (صورة {i+1})", ["نعم", "لا"], key=f"blur_{i}"
#         )
#         user_choices_blur.append(
#             1 if choice == "لا" and blur_levels[i] > 0 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")


# blur_errors = sum(user_choices_blur)

# # اختبار الازدواجية
# st.header("اختيار 2: الازدواجية في الرؤية")
# st.write("حدد إذا كانت الصورة تبدو مزدوجة.")

# user_choices_double = []
# for i, img_path in enumerate(images):
#     img = (
#         generate_double_vision_image(img_path) if i % 2 == 1 else cv2.imread(img_path)
#     )  # كل صورة ثانية مزدوجة
#     if img is not None:
#         st.image(img, caption=f"صورة {i+1}", width=200)
#         choice = st.radio(
#             f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})", ["نعم", "لا"], key=f"double_{i}"
#         )
#         user_choices_double.append(
#             1 if choice == "نعم" and i % 2 == 1 else 0
#         )  # 1 إذا أخطأ
#     else:
#         st.write(f"Could not load image: {img_path}")

# double_errors = sum(user_choices_double)

# # تحليل باستخدام AI
# if st.button("احسب النتائج"):
#     # Ensure input to predict is a 2D array/DataFrame
#     input_data = pd.DataFrame(
#         [[blur_errors, double_errors]], columns=["blur_errors", "double_errors"]
#     )
#     pred_blur = model_blur.predict(input_data)[0]
#     pred_double = model_double.predict(input_data)[0]

#     st.subheader("النتائج من النموذج AI:")
#     if pred_blur == 1:
#         st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
#     else:
#         st.write("لا يوجد دليل على رؤية ضبابية.")

#     if pred_double == 1:
#         st.write(
#             "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
#         )
#     else:
#         st.write("لا يوجد دليل على ازدواجية في الرؤية.")

#     st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
#     st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")

# # New section for individual image prediction
# st.header("توقع حالة صورة فردية")
# uploaded_file = st.file_uploader("اختر صورة لتحليلها:", type=["jpg", "png"])

# if uploaded_file is not None:
#     # Read the uploaded image
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, 1)

#     st.image(img, caption="الصورة المحملة", width=200)

#     # Perform predictions (using dummy error counts for now)
#     # In a real application, you would analyze the image content to get these errors
#     # For demonstration, we'll use placeholder values
#     image_blur_errors = 2  # Replace with actual image analysis result
#     image_double_errors = 3  # Replace with actual image analysis result

#     input_data_single = pd.DataFrame(
#         [[image_blur_errors, image_double_errors]],
#         columns=["blur_errors", "double_errors"],
#     )
#     pred_blur_single = model_blur.predict(input_data_single)[0]
#     pred_double_single = model_double.predict(input_data_single)[0]

#     st.subheader("نتائج التوقع للصورة:")
#     if pred_blur_single == 1:
#         st.write("احتمال وجود رؤية ضبابية في هذه الصورة.")
#     else:
#         st.write("لا يوجد دليل على رؤية ضبابية في هذه الصورة.")

#     if pred_double_single == 1:
#         st.write("احتمال وجود ازدواجية في الرؤية في هذه الصورة.")
#     else:
#         st.write("لا يوجد دليل على ازدواجية في الرؤية في هذه الصورة.")


# # اختبار الازدواجية
# st.header("اختبار 2: الازدواجية في الرؤية")
# st.write("حدد إذا كانت الصورة تبدو مزدوجة.")


# user_choices_double = []
# for i, img_path in enumerate(images):
#     img = (
#         generate_double_vision_image(img_path) if i % 2 == 1 else cv2.imread(img_path)
#     )  # كل صورة ثانية مزدوجة
#     st.image(img, caption=f"صورة {i+1}", width=200)
#     choice = st.radio(
#         f"هل هذه الصورة تبدو مزدوجة؟ (صورة {i+1})", ["نعم", "لا"], key=f"double_{i}"
#     )
#     user_choices_double.append(1 if choice == "نعم" and i % 2 == 1 else 0)  # 1 إذا أخطأ

# double_errors = sum(user_choices_double)

# # تحليل باستخدام AI
# if st.button("احسب النتائج"):
#     pred_blur = model_blur.predict([[blur_errors, double_errors]])[0]
#     pred_double = model_double.predict([[blur_errors, double_errors]])[0]

#     st.subheader("النتائج من النموذج AI:")
#     if pred_blur == 1:
#         st.write("احتمال وجود رؤية ضبابية (بناءً على الأخطاء في الاختبار الأول).")
#     else:
#         st.write("لا يوجد دليل على رؤية ضبابية.")

#     if pred_double == 1:
#         st.write(
#             "احتمال وجود ازدواجية في الرؤية (بناءً على الأخطاء في الاختبار الثاني)."
#         )
#     else:
#         st.write("لا يوجد دليل على ازدواجية في الرؤية.")

#     st.write(f"عدد الأخطاء في اختبار الضبابية: {blur_errors}")
#     st.write(f"عدد الأخطاء في اختبار الازدواجية: {double_errors}")
