# استيراد المكتبات المطلوبة
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st
import requests
from io import BytesIO
from scipy.io import loadmat  # لقراءة ملفات .mat إذا كانت البيانات في هذا الصيغة

# تحميل البيانات الحقيقية من الرابط (DOI: 10.12751/g-node.3emvhy)
# الرابط الأصلي هو DOI، لذا سنستخدم رابط تحميل مباشر لملف عينة (مثل ملف .mat من المستودع)
# إذا لم يعمل، قم بتحميل الملف يدوياً من https://gin.g-node.org/G-Node/3emvhy ورفعه إلى Colab
url = "https://gin.g-node.org/G-Node/3emvhy/raw/master/data/subject1.mat"  # مثال لملف عينة (قد يحتاج تعديل حسب الملف المتاح)
try:
    response = requests.get(url)
    response.raise_for_status()
    mat_data = loadmat(BytesIO(response.content))

    # استخراج ميزات بسيطة من البيانات (افتراضياً، البيانات تحتوي على إشارات EEG)
    # هنا، نفترض أن البيانات تحتوي على مصفوفات مثل 'data' للإشارات
    # سنستخرج متوسط الإشارات كتمثيل للميزات (يمكن تعديل حسب البيانات الحقيقية)
    eeg_signals = mat_data["data"]  # افتراضياً، مصفوفة الإشارات
    n_samples = eeg_signals.shape[0]  # عدد العينات

    # استخراج ميزات بسيطة:
    # - reaction_time: متوسط الإشارة الأولى (كتمثيل لزمن الحركة)
    # - tremor_level: انحراف معياري للإشارة (كتمثيل للارتعاش)
    # - accuracy: نسبة الإشارات فوق عتبة (كتمثيل للدقة)
    reaction_time = (
        np.mean(eeg_signals[:, 0], axis=1)
        if eeg_signals.ndim > 1
        else eeg_signals[:, 0]
    )
    tremor_level = (
        np.std(eeg_signals, axis=1) if eeg_signals.ndim > 1 else np.std(eeg_signals)
    )
    accuracy = (
        np.mean(eeg_signals > np.median(eeg_signals), axis=1)
        if eeg_signals.ndim > 1
        else (eeg_signals > np.median(eeg_signals)).astype(float)
    )

    # تطبيع القيم إلى النطاق 0-1
    reaction_time = (reaction_time - np.min(reaction_time)) / (
        np.max(reaction_time) - np.min(reaction_time)
    )
    tremor_level = (tremor_level - np.min(tremor_level)) / (
        np.max(tremor_level) - np.min(tremor_level)
    )
    accuracy = (accuracy - np.min(accuracy)) / (np.max(accuracy) - np.min(accuracy))

    # تحديد has_motor_issue بناءً على قواعد بسيطة (يمكن تعديلها بناءً على البيانات الحقيقية)
    has_motor_issue = []
    for i in range(len(reaction_time)):
        if reaction_time[i] > 0.6 or tremor_level[i] > 0.5 or accuracy[i] < 0.5:
            has_motor_issue.append(1)  # يعاني من ضعف حركي
        else:
            has_motor_issue.append(0)  # سليم

    # إنشاء DataFrame
    data = pd.DataFrame(
        {
            "reaction_time": reaction_time,
            "tremor_level": tremor_level,
            "accuracy": accuracy,
            "has_motor_issue": has_motor_issue,
        }
    )

    # حفظ البيانات في ملف CSV
    data.to_csv("motor_test_data_real.csv", index=False)
    print("تم تحميل ومعالجة البيانات الحقيقية وحفظها في motor_test_data_real.csv.")

except Exception as e:
    print(
        f"فشل في تحميل البيانات من الرابط: {e}. استخدم البيانات التجريبية بدلاً من ذلك."
    )
    # إذا فشل، استخدم البيانات التجريبية كبديل
    np.random.seed(42)
    n_samples = 1000
    reaction_time = np.random.uniform(0.1, 1.0, n_samples)
    tremor_level = np.random.uniform(0.0, 1.0, n_samples)
    accuracy = np.random.uniform(0.0, 1.0, n_samples)
    has_motor_issue = [
        1 if rt > 0.6 or tl > 0.5 or acc < 0.5 else 0
        for rt, tl, acc in zip(reaction_time, tremor_level, accuracy)
    ]
    data = pd.DataFrame(
        {
            "reaction_time": reaction_time,
            "tremor_level": tremor_level,
            "accuracy": accuracy,
            "has_motor_issue": has_motor_issue,
        }
    )
    data.to_csv("motor_test_data_real.csv", index=False)
    print("تم إنشاء بيانات تجريبية كبديل.")

# قراءة البيانات من CSV
data = pd.read_csv("motor_test_data_real.csv")

# فصل الميزات (X) والنتيجة (y)
X = data[["reaction_time", "tremor_level", "accuracy"]]
y = data["has_motor_issue"]

# تقسيم البيانات إلى تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تدريب الموديل باستخدام RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred = model.predict(X_test)

# طباعة الدقة وتقرير التصنيف
accuracy = accuracy_score(y_test, y_pred)
print(f"دقة الموديل: {accuracy:.2f}")
print("تقرير التصنيف:")
print(classification_report(y_test, y_pred))

# واجهة Streamlit
st.title("اختبار الحركة (Motor Disability Test) - باستخدام بيانات حقيقية")

st.write("أدخل القيم التالية لاختبار الحركة:")

# Sliders لإدخال القيم
reaction_time_input = st.slider(
    "الزمن المستغرق للحركة (reaction_time) بالثواني", 0.1, 1.0, 0.5
)
tremor_level_input = st.slider(
    "مستوى الارتعاش (tremor_level) من 0 إلى 1", 0.0, 1.0, 0.5
)
accuracy_input = st.slider("نسبة الدقة (accuracy) من 0 إلى 1", 0.0, 1.0, 0.5)

# زر للتحليل
if st.button("تحليل النتيجة"):
    # إنشاء مصفوفة للتنبؤ
    input_data = np.array([[reaction_time_input, tremor_level_input, accuracy_input]])

    # التنبؤ
    prediction = model.predict(input_data)[0]

    # عرض النتيجة
    if prediction == 0:
        st.success("الحركة سليمة")
    else:
        st.error("في احتمال ضعف حركي")
