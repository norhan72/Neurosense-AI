import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from google.cloud import speech
import io
import os

# import speech_recognition as sr  # لتحويل الصوت إلى نص (اختياري)


# دالة لاستخراج ميزات من الصوت
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)  # تحميل الصوت
    # ميزات أساسية
    pitch = librosa.yin(y, fmin=50, fmax=300)  # نبرة الصوت
    energy = librosa.feature.rms(y=y)  # طاقة الصوت
    tempo = librosa.beat.tempo(y=y, sr=sr)  # إيقاع (لقياس التركيز)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # ميزات MFCC للكلام

    # حساب متوسطات (لتبسيط)
    features = {
        "avg_pitch": np.mean(pitch),
        "avg_energy": np.mean(energy),
        "tempo": tempo[0],
        "mfcc_mean": np.mean(mfccs, axis=1).tolist(),  # قائمة من 13 قيمة
    }
    return features


# دالة لقياس التلعثم والتأخر (بسيطة، بناءً على تحليل النص)
def analyze_speech_text(audio_path, expected_text=None):
    client = speech.SpeechClient()
    try:
        with io.open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Adjust encoding based on your audio file
            sample_rate_hertz=16000,  # Adjust sample rate based on your audio file
            language_code="ar-AE",  # Changed to Arabic - UAE
        )

        response = client.recognize(config=config, audio=audio)

        transcribed_text = ""
        for result in response.results:
            transcribed_text += result.alternatives[0].transcript + " "
        transcribed_text = transcribed_text.strip()

        # قياس التلعثم: عدد التكرارات أو الأخطاء (بسيط)
        stutter_score = (
            len(transcribed_text.split()) / len(expected_text.split())
            if expected_text and len(expected_text.split()) > 0
            else 1.0
        )

        # قياس التأخر: طول النص مقارنة بالوقت (بسيط)
        y, sr_librosa = librosa.load(audio_path, sr=16000)
        duration = librosa.get_duration(y=y, sr=sr_librosa)
        speech_rate = len(transcribed_text.split()) / duration if duration > 0 else 0.0

        return {
            "stutter_score": stutter_score,
            "speech_rate": speech_rate,
            "transcribed": transcribed_text,
        }
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return {"stutter_score": 0, "speech_rate": 0, "transcribed": "خطأ في التعرف"}


# دمج النتائج من الاختبارين
def combine_results(features1, features2, text_analysis1, text_analysis2):
    # دمج الميزات في قائمة واحدة
    combined = (
        list(features1.values())
        + list(features2.values())
        + [
            text_analysis1["stutter_score"],
            text_analysis1["speech_rate"],
            text_analysis2["stutter_score"],
            text_analysis2["speech_rate"],
        ]
    )
    return np.array(combined).reshape(1, -1)


# نموذج ML بسيط (تدريبه على بيانات وهمية - في الواقع، تحتاج بيانات طبية حقيقية)
# افترض بيانات تدريب: X = ميزات، y = 1 لـ MS، 0 لغيرها (هذا مثال فقط!)
X_sample = np.random.rand(100, 20)  # 20 ميزة (افتراضي)
y_sample = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# تشغيل على صوت المريض
# IMPORTANT: Replace 'path/to/audio1.wav' and 'path/to/audio2.wav' below with the actual paths to your audio files.
# Ensure your audio files are in a format compatible with Google Cloud Speech-to-Text (e.g., WAV with LINEAR16 encoding and a suitable sample rate).
audio1 = "path/to/audio1.wav"  # صوت التذكر
audio2 = "path/to/audio2.wav"  # صوت القراءة
expected_text2 = "النص المحدد اللي المريض هيقرأه"  # النص الثابت للاختبار الثاني

features1 = extract_audio_features(audio1)
features2 = extract_audio_features(audio2)
text1 = analyze_speech_text(audio1)  # بدون نص متوقع للتذكر
text2 = analyze_speech_text(audio2, expected_text2)

combined_features = combine_results(features1, features2, text1, text2)
prediction = model.predict(combined_features)[0]  # 1 = احتمال MS، 0 = لا
probability = model.predict_proba(combined_features)[0][1]  # نسبة الاحتمال

# Replace with the directory path where you uploaded your audio files in Google Drive
# Example: '/content/drive/My Drive/audio_files'
audio_directory_path = (
    "/content/drive/My Drive/audio_files"  # TODO: Replace with your directory path
)

# List files in the directory
if os.path.exists(audio_directory_path):
    print(f"Files in {audio_directory_path}:")
    for filename in os.listdir(audio_directory_path):
        print(filename)
else:
    print(f"Directory not found: {audio_directory_path}")

import os

# Replace with the full path to your audio file in Google Drive
# Example: '/content/drive/My Drive/audio_files/audio1.wav'
audio_file_path = "/content/drive/My Drive/audio_files/audio1.wav"  # TODO: Replace with your file path

# Check if the specific file exists
if os.path.exists(audio_file_path):
    print(f"File exists: {audio_file_path}")
else:
    print(f"File not found: {audio_file_path}")

# تشغيل على صوت المريض
# IMPORTANT: Replace 'path/to/audio1.wav' and 'path/to/audio2.wav' with the actual paths to your audio files.
# Ensure your audio files are in a format compatible with Google Cloud Speech-to-Text (e.g., WAV with LINEAR16 encoding and a suitable sample rate).
audio1 = "path/to/audio1.wav"  # صوت التذكر # TODO: Replace with actual audio file path
audio2 = "path/to/audio2.wav"  # صوت القراءة # TODO: Replace with actual audio file path
expected_text2 = "النص المحدد اللي المريض هيقرأه"  # النص الثابت للاختبار الثاني

features1 = extract_audio_features(audio1)
features2 = extract_audio_features(audio2)
text1 = analyze_speech_text(audio1)  # بدون نص متوقع للتذكر
text2 = analyze_speech_text(audio2, expected_text2)

combined_features = combine_results(features1, features2, text1, text2)
prediction = model.predict(combined_features)[0]  # 1 = احتمال MS، 0 = لا
probability = model.predict_proba(combined_features)[0][1]  # نسبة الاحتمال

# تقرير النتائج
print("نتيجة التشخيص الافتراضي:")
print(
    f"احتمال وجود أعراض MS: {'عالي' if prediction == 1 else 'منخفض'} (نسبة: {probability:.2f})"
)
print("مشاكل مكتشفة:")
print(f"- تلعثم في الاختبار الأول: {text1['stutter_score']:.2f}")
print(f"- سرعة الكلام في الاختبار الأول: {text1['speech_rate']:.2f} كلمة/ثانية")
print(f"- تلعثم في الاختبار الثاني: {text2['stutter_score']:.2f}")
print(f"- سرعة الكلام في الاختبار الثاني: {text2['speech_rate']:.2f} كلمة/ثانية")
print(
    f"- متوسط النبرة: {features1['avg_pitch']:.2f} (اختبار 1), {features2['avg_pitch']:.2f} (اختبار 2)"
)
# إضافة المزيد بناءً على الميزات...
