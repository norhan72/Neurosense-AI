import os
import numpy as np
import pandas as pd
import tempfile
import uuid
import soundfile as sf
import librosa
import parselmouth
from parselmouth.praat import call
from difflib import SequenceMatcher
from faster_whisper import WhisperModel
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from joblib import dump, load


class SpeechTest:
    def __init__(self, sr=16000):
        self.sr = sr
        self.BASE = os.getcwd()
        self.DATA_DIR = os.path.join(self.BASE, "data", "speech")
        self.HEALTHY_DIR = os.path.join(self.DATA_DIR, "healthy_audio")
        self.FEATURES_CSV = os.path.join(self.DATA_DIR, "healthy_features.csv")
        self.MODEL_PATH = os.path.join(self.DATA_DIR, "anomaly_model.joblib")
        self.SCALER_PATH = os.path.join(self.DATA_DIR, "scaler.joblib")
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.HEALTHY_DIR, exist_ok=True)
        self.whisper_model = None
        self.clf, self.scaler = self._load_model_and_scaler()

    def _transcribe_with_timestamps(self, audio_path):
        if self.whisper_model is None:
            self.whisper_model = WhisperModel("base", device="cpu")
        segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
        text = ""
        segs = []
        for seg in segments:
            seg_text = seg.text.strip()
            segs.append({"start": seg.start, "end": seg.end, "text": seg_text})
            text += " " + seg_text
        return text.strip(), segs

    def _calc_text_similarity(self, expected, actual):
        if not expected:
            return -1.0
        return SequenceMatcher(None, expected, actual).ratio()

    def _load_audio_bytes(self, data):
        """Load bytes (wav blob) into numpy array using librosa (expects WAV or common)."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            tmp_path = tmp.name
        y, _ = librosa.load(tmp_path, sr=self.sr)
        os.remove(tmp_path)
        return y

    def _safe_write_temp_audio(self, data_array):
        """Write numpy audio to a temp wav file and return path (safe on Windows). Expects 1D ndarray."""
        tmp_path = os.path.join(tempfile.gettempdir(), f"tmp_{uuid.uuid4().hex}.wav")
        sf.write(tmp_path, data_array, self.sr)
        return tmp_path

    def _extract_features_from_array(self, data_array):
        features = {}
        # duration & energy
        duration = float(librosa.get_duration(y=data_array, sr=self.sr) or 0.0)
        features["duration"] = duration
        try:
            rms_feat = librosa.feature.rms(y=data_array)
            features["rms"] = float(np.mean(rms_feat)) if rms_feat.size else 0.0
        except Exception:
            features["rms"] = 0.0

        # pitch (f0) using pyin (robust)
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                data_array, fmin=50, fmax=400, sr=self.sr
            )
            f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            features["f0_mean"] = float(np.mean(f0_clean)) if len(f0_clean) else 0.0
            features["f0_std"] = float(np.std(f0_clean)) if len(f0_clean) else 0.0
        except Exception:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0

        # write temp wav for parselmouth (Praat) measures
        tmp_wav = self._safe_write_temp_audio(data_array)
        snd = parselmouth.Sound(tmp_wav)

        # jitter & shimmer
        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call(
                [snd, point_process],
                "Get shimmer (local)",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )
            features["jitter"] = float(jitter)
            features["shimmer"] = float(shimmer)
        except Exception:
            features["jitter"] = 0.0
            features["shimmer"] = 0.0

        # HNR
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            features["hnr"] = float(hnr)
        except Exception:
            features["hnr"] = 0.0

        # formants (F1, F2)
        try:
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.02, 50)
            t = snd.duration / 2.0
            f1 = call(formant, "Get value at time", 1, t, "Hertz")
            f2 = call(formant, "Get value at time", 2, t, "Hertz")
            features["formant_f1"] = float(f1)
            features["formant_f2"] = float(f2)
        except Exception:
            features["formant_f1"] = 0.0
            features["formant_f2"] = 0.0

        # voiced segments & pauses
        try:
            intervals = librosa.effects.split(data_array, top_db=30)
            total_voiced = sum(((e - s) / self.sr) for s, e in intervals)
            features["voiced_ratio"] = (
                float(total_voiced / duration) if duration > 0 else 0.0
            )
            features["voiced_segments"] = int(len(intervals))
            # mean pause length (approx)
            pauses = []
            for i in range(len(intervals) - 1):
                prev_end = intervals[i][1] / self.sr
                next_start = intervals[i + 1][0] / self.sr
                pauses.append(next_start - prev_end)
            features["mean_pause"] = float(np.mean(pauses)) if pauses else 0.0
            features["long_pauses_count"] = int(sum(1 for p in pauses if p > 0.6))
        except Exception:
            features["voiced_ratio"] = 0.0
            features["voiced_segments"] = 0
            features["mean_pause"] = 0.0
            features["long_pauses_count"] = 0

        # cleanup
        try:
            os.remove(tmp_wav)
        except Exception:
            pass

        return features

    def _extract_features_from_bytes(self, data_bytes):
        y = self._load_audio_bytes(data_bytes)
        features = self._extract_features_from_array(y)
        return features

    def _save_feature_row(self, row):
        df = pd.DataFrame([row])
        if os.path.exists(self.FEATURES_CSV):
            df_existing = pd.read_csv(self.FEATURES_CSV)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(self.FEATURES_CSV, index=False)

    def _load_features_matrix(self):
        if not os.path.exists(self.FEATURES_CSV):
            return None, None
        df = pd.read_csv(self.FEATURES_CSV)
        num_cols = [c for c in df.columns if c not in ("file",)]
        X = df[num_cols].fillna(0.0).values
        return df, X

    def _train(self, min_samples=5):
        df, X = self._load_features_matrix()
        if df is None or len(df) < min_samples:
            raise RuntimeError(
                f"Need at least {min_samples} healthy samples to train. Have: {0 if df is None else len(df)}"
            )
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        clf.fit(Xs)
        dump(clf, self.MODEL_PATH)
        dump(scaler, self.SCALER_PATH)
        return {"trained_samples": len(df)}

    def _load_model_and_scaler(self):
        # ensure both model and scaler exist
        if not os.path.exists(self.MODEL_PATH) or not os.path.exists(self.SCALER_PATH):
            return None, None
        clf = load(self.MODEL_PATH)
        scaler = load(self.SCALER_PATH)
        return clf, scaler

    def analyze(self, data_bytes, file_name):
        feats = self._extract_features_from_bytes(data_bytes)
        feats["file"] = file_name
        clf, scaler = self.clf, self.scaler

        if clf is None or scaler is None:
            clf, scaler = self._load_model_and_scaler()
            if clf is None or scaler is None:
                raise RuntimeError("Anomaly model not found. Train the model first.")
            self.clf, self.scaler = clf, scaler

        df_ref = pd.read_csv(self.FEATURES_CSV)
        num_cols = [c for c in df_ref.columns if c not in ("file",)]
        x = np.array([feats.get(c, 0.0) for c in num_cols], dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        score = -float(clf.decision_function(x_scaled)[0])
        label = "suspected" if score > 0.5 else "normal"
        return {"score": score, "label": label}

    def save_healthy_sample(self, data_bytes, file_name):
        fname = file_name or f"healthy_{np.random.randint(1e6)}.wav"
        path = os.path.join(self.HEALTHY_DIR, fname)
        with open(path, "wb") as f:
            f.write(data_bytes)
        feats = self._extract_features_from_bytes(data_bytes)
        feats["file"] = fname
        self._save_feature_row(feats)
        try:
            self._train()
            # reload model/scaler into memory
            self.clf, self.scaler = self._load_model_and_scaler()
        except Exception:
            pass
        return feats
