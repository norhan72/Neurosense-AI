import os
import re
import subprocess
import tempfile
import threading
import uuid
from difflib import SequenceMatcher

import librosa
import numpy as np
import pandas as pd
import parselmouth
import soundfile as sf
from joblib import dump
from parselmouth.praat import call
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "duration",
    "rms",
    "f0_mean",
    "f0_std",
    "jitter",
    "shimmer",
    "hnr",
    "formant_f1",
    "formant_f2",
    "voiced_ratio",
    "voiced_segments",
    "mean_pause",
    "long_pauses_count",
]
AUDIO_EXTENSIONS = (".wav", ".webm", ".ogg", ".mp3", ".m4a", ".flac")


def _finite_or_zero(value):
    value = float(value)
    return value if np.isfinite(value) else 0.0


class SpeechTest:
    # Anomaly score above which a recording is flagged as suspected. 0 is the
    # IsolationForest's own decision boundary (set by `contamination`).
    SUSPECTED_THRESHOLD = 0.0
    MIN_TRAIN_SAMPLES = 5

    def __init__(self, sr=16000):
        self.sr = sr
        self.BASE = os.getcwd()
        self.DATA_DIR = os.path.join(self.BASE, "data", "speech")
        self.HEALTHY_DIR = os.path.join(self.DATA_DIR, "healthy_audio")
        # v2: formants are now actually extracted (v1 rows have them stuck at 0),
        # so v1 rows are not comparable and the cache is rebuilt from HEALTHY_DIR.
        self.FEATURES_CSV = os.path.join(self.DATA_DIR, "healthy_features_v2.csv")
        self.MODEL_PATH = os.path.join(self.DATA_DIR, "anomaly_model.joblib")
        self.SCALER_PATH = os.path.join(self.DATA_DIR, "scaler.joblib")
        os.makedirs(self.HEALTHY_DIR, exist_ok=True)

        self.whisper_model = None
        self.clf = None
        self.scaler = None
        self._lock = threading.Lock()
        with self._lock:
            self._train()

    # ---------- transcription (optional) ----------

    def _transcribe_with_timestamps(self, audio_path):
        if self.whisper_model is None:
            # Imported lazily: faster_whisper pulls in torch/ctranslate2, which is
            # heavy and fails to import on some Windows setups.
            from faster_whisper import WhisperModel

            self.whisper_model = WhisperModel("base", device="cpu")
        segments, _ = self.whisper_model.transcribe(audio_path, beam_size=5)
        segs = [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in segments
        ]
        return " ".join(s["text"] for s in segs).strip(), segs

    @staticmethod
    def _calc_text_similarity(expected, actual):
        if not expected:
            return -1.0
        return SequenceMatcher(None, expected, actual).ratio()

    # ---------- audio decoding ----------

    def _load_audio_bytes(self, data):
        """Decode an audio blob to a mono float32 array at self.sr.

        Tries soundfile (WAV/FLAC), then librosa, then an ffmpeg conversion
        for browser-recorded formats (webm/ogg).
        """
        with tempfile.NamedTemporaryFile(suffix="", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        wav_tmp = None
        try:
            try:
                y, sr = sf.read(tmp_path, dtype="float32")
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr != self.sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
                return y.astype(np.float32)
            except Exception:
                pass

            try:
                y, _ = librosa.load(tmp_path, sr=self.sr, mono=True)
                return y.astype(np.float32)
            except Exception:
                pass

            wav_tmp = os.path.join(
                tempfile.gettempdir(), f"tmp_conv_{uuid.uuid4().hex}.wav"
            )
            ffmpeg_cmd = [
                "ffmpeg", "-y", "-i", tmp_path,
                "-ar", str(self.sr), "-ac", "1", wav_tmp,
            ]
            try:
                subprocess.run(
                    ffmpeg_cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                y, _ = sf.read(wav_tmp, dtype="float32")
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                return y.astype(np.float32)
            except FileNotFoundError:
                raise RuntimeError(
                    "Unable to decode audio: ffmpeg not found. Install ffmpeg or send WAV/PCM audio."
                )
            except Exception as e:
                raise RuntimeError(f"Unable to decode audio: {e}")
        finally:
            for path in (tmp_path, wav_tmp):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    # ---------- feature extraction ----------

    def _extract_features_from_array(self, y):
        features = {}
        duration = float(librosa.get_duration(y=y, sr=self.sr) or 0.0)
        features["duration"] = duration

        try:
            rms_feat = librosa.feature.rms(y=y)
            features["rms"] = float(np.mean(rms_feat)) if rms_feat.size else 0.0
        except Exception:
            features["rms"] = 0.0

        # pitch (f0) using pyin
        try:
            f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=self.sr)
            f0_clean = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            features["f0_mean"] = float(np.mean(f0_clean)) if len(f0_clean) else 0.0
            features["f0_std"] = float(np.std(f0_clean)) if len(f0_clean) else 0.0
        except Exception:
            features["f0_mean"] = 0.0
            features["f0_std"] = 0.0

        # Praat measures, built straight from the array (no temp file needed)
        snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=self.sr)

        try:
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
            jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call(
                [snd, point_process],
                "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6,
            )
            features["jitter"] = _finite_or_zero(jitter)
            features["shimmer"] = _finite_or_zero(shimmer)
        except Exception:
            features["jitter"] = 0.0
            features["shimmer"] = 0.0

        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            features["hnr"] = _finite_or_zero(call(harmonicity, "Get mean", 0, 0))
        except Exception:
            features["hnr"] = 0.0

        # Mean F1/F2 over the whole recording (a single mid-point sample is
        # often unvoiced and returns NaN).
        try:
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            features["formant_f1"] = _finite_or_zero(
                call(formant, "Get mean", 1, 0, 0, "hertz")
            )
            features["formant_f2"] = _finite_or_zero(
                call(formant, "Get mean", 2, 0, 0, "hertz")
            )
        except Exception:
            features["formant_f1"] = 0.0
            features["formant_f2"] = 0.0

        # voiced segments & pauses
        try:
            intervals = librosa.effects.split(y, top_db=30)
            total_voiced = sum((e - s) / self.sr for s, e in intervals)
            features["voiced_ratio"] = (
                float(total_voiced / duration) if duration > 0 else 0.0
            )
            features["voiced_segments"] = int(len(intervals))
            pauses = [
                (intervals[i + 1][0] - intervals[i][1]) / self.sr
                for i in range(len(intervals) - 1)
            ]
            features["mean_pause"] = float(np.mean(pauses)) if pauses else 0.0
            features["long_pauses_count"] = int(sum(1 for p in pauses if p > 0.6))
        except Exception:
            features["voiced_ratio"] = 0.0
            features["voiced_segments"] = 0
            features["mean_pause"] = 0.0
            features["long_pauses_count"] = 0

        return features

    def _extract_features_from_bytes(self, data_bytes):
        return self._extract_features_from_array(self._load_audio_bytes(data_bytes))

    # ---------- healthy reference set ----------

    def _load_features_df(self):
        if not os.path.exists(self.FEATURES_CSV):
            return pd.DataFrame(columns=FEATURE_COLUMNS + ["file"])
        return pd.read_csv(self.FEATURES_CSV)

    def _sync_features_with_audio_dir(self, df):
        """Extract features for healthy recordings not yet in the CSV."""
        known = set(df["file"].astype(str)) if "file" in df.columns else set()
        new_rows = []
        for fname in sorted(os.listdir(self.HEALTHY_DIR)):
            if fname in known or not fname.lower().endswith(AUDIO_EXTENSIONS):
                continue
            try:
                with open(os.path.join(self.HEALTHY_DIR, fname), "rb") as f:
                    feats = self._extract_features_from_bytes(f.read())
            except Exception:
                continue
            feats["file"] = fname
            new_rows.append(feats)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = new_df if df.empty else pd.concat([df, new_df], ignore_index=True)
            df.to_csv(self.FEATURES_CSV, index=False)
        return df

    def _train(self):
        """(Re)fit the anomaly model on the healthy set. Caller must hold self._lock."""
        df = self._sync_features_with_audio_dir(self._load_features_df())
        if len(df) < self.MIN_TRAIN_SAMPLES:
            self.clf, self.scaler = None, None
            return {"trained_samples": len(df), "trained": False}

        X = df.reindex(columns=FEATURE_COLUMNS).fillna(0.0).to_numpy(dtype=float)
        scaler = StandardScaler().fit(X)
        clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
        clf.fit(scaler.transform(X))

        dump(clf, self.MODEL_PATH)
        dump(scaler, self.SCALER_PATH)
        self.clf, self.scaler = clf, scaler
        return {"trained_samples": len(df), "trained": True}

    # ---------- public API ----------

    def analyze(self, data_bytes, file_name):
        clf, scaler = self.clf, self.scaler
        if clf is None or scaler is None:
            raise RuntimeError(
                f"Anomaly model not trained: need at least {self.MIN_TRAIN_SAMPLES} healthy samples."
            )

        feats = self._extract_features_from_bytes(data_bytes)
        x = np.array([[feats[c] for c in FEATURE_COLUMNS]], dtype=float)
        score = -float(clf.decision_function(scaler.transform(x))[0])

        if score > self.SUSPECTED_THRESHOLD:
            label_en, label_ar = "Suspected", "محتمل"
        else:
            label_en, label_ar = "Normal", "طبيعي"
        return {"score": score, "label_en": label_en, "label_ar": label_ar}

    @staticmethod
    def _safe_filename(file_name):
        """Strip directories and unsafe characters from a client-supplied filename."""
        name = os.path.basename((file_name or "").replace("\\", "/"))
        name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
        if not name:
            name = f"healthy_{uuid.uuid4().hex[:8]}"
        if not name.lower().endswith(AUDIO_EXTENSIONS):
            name += ".wav"
        return name

    def save_healthy_sample(self, data_bytes, file_name):
        feats = self._extract_features_from_bytes(data_bytes)  # validates the audio first
        fname = self._safe_filename(file_name)

        with self._lock:
            path = os.path.join(self.HEALTHY_DIR, fname)
            if os.path.exists(path):
                stem, ext = os.path.splitext(fname)
                fname = f"{stem}_{uuid.uuid4().hex[:8]}{ext}"
                path = os.path.join(self.HEALTHY_DIR, fname)

            with open(path, "wb") as f:
                f.write(data_bytes)

            row = pd.DataFrame([{**feats, "file": fname}])
            df = pd.concat([self._load_features_df(), row], ignore_index=True)
            df.to_csv(self.FEATURES_CSV, index=False)
            self._train()

        return fname, feats
