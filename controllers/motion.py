import os
import glob
import json
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler


DEFAULT_SAMPLE_RATE_HZ = 50.0
MIN_SAMPLES = 15

FEATURE_NAMES = [
    "std_ax",
    "std_ay",
    "std_az",
    "rms_accel",
    "std_alpha",
    "std_beta",
    "std_gamma",
    "rms_gyro",
    "mean_abs_ax",
    "mean_abs_ay",
    "sma",
    "ptp_accel_mag",
    "dom_freq_accel",
    "jerk_accel",
    "jerk_gyro",
    "entropy_accel",
    "entropy_gyro",
    "zcr_accel",
    "zcr_gyro",
]


class MotionTest:
    def __init__(self, contamination=0.09, random_seed=42):
        self.data_folder = os.path.join(os.getcwd(), "data", "motion")
        self.contamination = contamination
        self.random_seed = random_seed
        self.pipeline = {}
        self._train()

    # ---------- feature helpers ----------

    @staticmethod
    def _time_seconds(t):
        """Return timestamps in seconds, or None if they are unusable.

        Browsers send `t` in milliseconds; detect that from the median step.
        """
        if len(t) < 2 or not np.all(np.isfinite(t)):
            return None
        if np.median(np.diff(t)) > 1.0:
            t = t / 1000.0
        return t

    @staticmethod
    def _jerk(signal, t):
        if len(signal) < 3:
            return 0.0
        if t is None:
            dt = np.full(len(signal) - 1, 1.0 / DEFAULT_SAMPLE_RATE_HZ)
        else:
            dt = np.diff(t)
            positive = dt[dt > 0]
            dt[dt <= 0] = (
                np.median(positive) if positive.size else 1.0 / DEFAULT_SAMPLE_RATE_HZ
            )
        return float(np.mean(np.abs(np.diff(signal) / dt)))

    @staticmethod
    def _entropy(v):
        """Shannon entropy of the value distribution (20-bin histogram)."""
        if len(v) == 0:
            return 0.0
        counts, _ = np.histogram(v, bins=20)
        p = counts[counts > 0] / counts.sum()
        return float(-np.sum(p * np.log(p)))

    @staticmethod
    def _zcr(v):
        """Zero-crossing rate around the mean (magnitudes are always >= 0)."""
        if len(v) < 2:
            return 0.0
        centered = v - np.mean(v)
        return float(((centered[:-1] * centered[1:]) < 0).mean())

    @staticmethod
    def _std(v):
        return float(np.std(v)) if len(v) > 0 else 0.0

    @staticmethod
    def _rms(v):
        return float(np.sqrt(np.mean(np.square(v)))) if len(v) > 0 else 0.0

    @staticmethod
    def _mean_abs(v):
        return float(np.mean(np.abs(v))) if len(v) > 0 else 0.0

    @staticmethod
    def _sma(ax, ay, az):
        if len(ax) == 0:
            return 0.0
        return float(np.mean(np.abs(ax) + np.abs(ay) + np.abs(az)))

    @staticmethod
    def _peak_to_peak(v):
        if len(v) == 0:
            return 0.0
        return float(np.max(v) - np.min(v))

    @staticmethod
    def _dominant_freq(v, t):
        if len(v) < 4:
            return 0.0
        fs = DEFAULT_SAMPLE_RATE_HZ
        if t is not None:
            dt = np.median(np.diff(t))
            if dt > 0:
                fs = 1.0 / dt
        mag = np.abs(np.fft.rfft(v - np.mean(v)))
        if mag.size <= 1:
            return 0.0
        freqs = np.fft.rfftfreq(len(v), 1.0 / fs)
        return float(freqs[np.argmax(mag[1:]) + 1])

    def _extract_features(self, samples):
        ax = np.array([s.get("ax", s.get("x", 0)) for s in samples], dtype=float)
        ay = np.array([s.get("ay", s.get("y", 0)) for s in samples], dtype=float)
        az = np.array([s.get("az", s.get("z", 0)) for s in samples], dtype=float)

        alpha = np.array([s.get("alpha", 0) for s in samples], dtype=float)
        beta = np.array([s.get("beta", 0) for s in samples], dtype=float)
        gamma = np.array([s.get("gamma", 0) for s in samples], dtype=float)

        t_raw = np.array(
            [s.get("t", s.get("time", np.nan)) for s in samples], dtype=float
        )
        t = self._time_seconds(t_raw)

        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(alpha**2 + beta**2 + gamma**2)

        features = {
            "std_ax": self._std(ax),
            "std_ay": self._std(ay),
            "std_az": self._std(az),
            "rms_accel": self._rms(np.concatenate([ax, ay, az])),
            "std_alpha": self._std(alpha),
            "std_beta": self._std(beta),
            "std_gamma": self._std(gamma),
            "rms_gyro": self._rms(np.concatenate([alpha, beta, gamma])),
            "mean_abs_ax": self._mean_abs(ax),
            "mean_abs_ay": self._mean_abs(ay),
            "sma": self._sma(ax, ay, az),
            "ptp_accel_mag": self._peak_to_peak(accel_mag),
            "dom_freq_accel": self._dominant_freq(accel_mag, t),
            "jerk_accel": self._jerk(accel_mag, t),
            "jerk_gyro": self._jerk(gyro_mag, t),
            "entropy_accel": self._entropy(accel_mag),
            "entropy_gyro": self._entropy(gyro_mag),
            "zcr_accel": self._zcr(accel_mag),
            "zcr_gyro": self._zcr(gyro_mag),
        }

        vec = np.array([features[name] for name in FEATURE_NAMES], dtype=float)
        return features, vec.reshape(1, -1)

    # ---------- training ----------

    def _load_data(self):
        files = sorted(glob.glob(os.path.join(self.data_folder, "*.json")))
        if not files:
            raise FileNotFoundError(
                f"No JSON files found in samples folder: {self.data_folder}"
            )

        vecs = []
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (OSError, ValueError):
                continue
            if not isinstance(data, list) or len(data) < MIN_SAMPLES:
                continue
            _, vec = self._extract_features(data)
            vecs.append(vec.ravel())

        if not vecs:
            raise ValueError(
                f"No valid recordings in {self.data_folder} "
                f"(each needs >= {MIN_SAMPLES} samples)"
            )

        X = np.vstack(vecs)
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)

        self.pipeline["X_raw"] = X
        self.pipeline["scaler"] = scaler
        self.pipeline["X_scaled"] = Xs
        self.pipeline["feature_names"] = FEATURE_NAMES
        return Xs

    @staticmethod
    def _get_best_model(results):
        """Pick the model with the lowest median anomaly score on the training normals."""
        return min(results, key=lambda name: float(np.median(results[name]["scores"])))

    def _train(self):
        X = self._load_data()
        n_neighbors = max(2, min(20, len(X) - 1))

        models = {
            "iforest": IsolationForest(
                n_estimators=300,
                contamination=self.contamination,
                random_state=self.random_seed,
            ),
            "lof": LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True),
            "ocsvm": OneClassSVM(kernel="rbf", gamma="scale", nu=0.02),
        }

        results = {}
        for name, model in models.items():
            model.fit(X)
            results[name] = dict(model=model, scores=-model.decision_function(X))

        best = self._get_best_model(results)
        self.pipeline["model_name"] = best
        self.pipeline["model"] = results[best]["model"]
        # Keep the full training score distribution; it is the reference for 0..100 scaling.
        self.pipeline["train_scores"] = np.asarray(results[best]["scores"], dtype=float)

    # ---------- inference ----------

    @staticmethod
    def _get_interpretation(likelihood, lang="en"):
        """Map the 0..100 likelihood to a label."""
        if lang == "ar":
            if likelihood < 20:
                return "من غير المحتمل أن يكون لديك MS"
            if likelihood < 50:
                return "احتمال متوسط للإصابة بمرض MS"
            if likelihood < 70:
                return "من المحتمل أن يكون لديك MS"
            return "من المرجح أن يكون لديك MS"

        if likelihood < 20:
            return "Unlikely to have MS"
        if likelihood < 50:
            return "Moderate likelihood of MS"
        if likelihood < 70:
            return "Likely to have MS"
        return "Highly likely to have MS"

    def analyze_samples(self, samples):
        if not isinstance(samples, list) or len(samples) < MIN_SAMPLES:
            raise ValueError(f"samples must be a list of at least {MIN_SAMPLES} readings")

        features, vec = self._extract_features(samples)
        scaler = self.pipeline["scaler"]
        model = self.pipeline["model"]
        train_scores = self.pipeline["train_scores"]

        score = float(-model.decision_function(scaler.transform(vec))[0])

        # Map score to 0..100 using the training distribution: min..95th percentile -> 0..100
        ref_min = float(np.min(train_scores))
        ref_high = float(np.percentile(train_scores, 95))
        span = ref_high - ref_min if ref_high > ref_min else 1.0
        ms_likelihood = float(np.clip((score - ref_min) / span * 100.0, 0, 100))

        return {
            "ms_likelihood": ms_likelihood,
            "features": features,
            "score": score,
            "label_en": self._get_interpretation(ms_likelihood),
            "label_ar": self._get_interpretation(ms_likelihood, "ar"),
        }

    # ---------- persistence ----------

    def save(self, path="model"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.pipeline, os.path.join(path, "pipeline.pkl"))

    @classmethod
    def load(cls, path="model", contamination=0.09, random_seed=42):
        obj = cls.__new__(cls)
        obj.data_folder = os.path.join(os.getcwd(), "data", "motion")
        obj.contamination = contamination
        obj.random_seed = random_seed
        obj.pipeline = joblib.load(os.path.join(path, "pipeline.pkl"))
        return obj
