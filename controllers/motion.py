import os
import glob
import json
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler


class MotionTest:
    def __init__(self, contamination=0.09, random_seed=42):
        self.data_folder = os.path.join(os.getcwd(), "data", "motion")
        self.contamination = contamination
        self.random_seed = random_seed
        self.pipeline = {}
        self._train()

    def _jerk(self, signal, t):
        if len(signal) < 3:
            return 0.0
        dt = np.diff(t)
        dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 0.02
        return float(np.mean(np.abs(np.diff(signal) / dt)))

    def _entropy(self, v):
        hist, _ = np.histogram(v, bins=20, density=True)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log(hist)))

    def _zcr(self, v):
        return float(((v[:-1] * v[1:]) < 0).mean())

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
        if np.all(np.isfinite(t)) and len(t) > 1:
            dt = np.median(np.diff(t))
            if dt > 1.0:
                dt = dt / 1000.0
            if dt <= 0:
                fs = 50.0
            else:
                fs = 1.0 / dt
        else:
            fs = 50.0
        sig = v - np.mean(v)
        try:
            fft = np.fft.rfft(sig)
            freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
            mag = np.abs(fft)
            if mag.size <= 1:
                return 0.0
            idx = np.argmax(mag[1:]) + 1
            return float(freqs[idx])
        except Exception:
            return 0.0

    def _extract_features(self, samples):
        # Extract raw channels
        ax = np.array([s.get("ax", s.get("x", 0)) for s in samples], dtype=float)
        ay = np.array([s.get("ay", s.get("y", 0)) for s in samples], dtype=float)
        az = np.array([s.get("az", s.get("z", 0)) for s in samples], dtype=float)

        alpha = np.array([s.get("alpha", 0) for s in samples], dtype=float)
        beta = np.array([s.get("beta", 0) for s in samples], dtype=float)
        gamma = np.array([s.get("gamma", 0) for s in samples], dtype=float)

        t = np.array([s.get("t", s.get("time", np.nan)) for s in samples], dtype=float)

        # Magnitudes
        accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
        gyro_mag = np.sqrt(alpha**2 + beta**2 + gamma**2)

        # Existing features
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
        }

        # --- New features ---
        features.update(
            {
                "jerk_accel": self._jerk(accel_mag, t),
                "jerk_gyro": self._jerk(gyro_mag, t),
                "entropy_accel": self._entropy(accel_mag),
                "entropy_gyro": self._entropy(gyro_mag),
                "zcr_accel": self._zcr(accel_mag),
                "zcr_gyro": self._zcr(gyro_mag),
            }
        )

        # Feature vector (keep stable order!)
        vec = np.array(
            [
                # original features
                features["std_ax"],
                features["std_ay"],
                features["std_az"],
                features["rms_accel"],
                features["std_alpha"],
                features["std_beta"],
                features["std_gamma"],
                features["rms_gyro"],
                features["mean_abs_ax"],
                features["mean_abs_ay"],
                features["sma"],
                features["ptp_accel_mag"],
                features["dom_freq_accel"],
                # new features
                features["jerk_accel"],
                features["jerk_gyro"],
                features["entropy_accel"],
                features["entropy_gyro"],
                features["zcr_accel"],
                features["zcr_gyro"],
            ],
            dtype=float,
        ).reshape(1, -1)

        return features, vec

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
            except Exception:
                # skip unreadable files but continue
                continue

            _, vec = self._extract_features(data)
            vecs.append(vec.ravel())

        if not vecs:
            raise ValueError(
                f"No valid sample segments found in {self.data_folder} (need >=15 samples each)"
            )

        X = np.vstack(vecs)
        self.pipeline["X_raw"] = X

        scaler = RobustScaler()
        self.pipeline["scaler"] = scaler

        Xs = scaler.fit_transform(X)
        self.pipeline["X_scaled"] = Xs

        self.pipeline["feature_names"] = [
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
        return Xs

    def _get_best_model(self, results):
        # choose model with lowest median anomaly score (on training normals)
        best = None
        best_med = None
        for name, info in results.items():
            med = float(np.median(info["scores"]))
            if best is None or med < best_med:
                best = name
                best_med = med
        return best, best_med

    def _train(self):
        # TODO: load saved model if exists
        X = self._load_data()
        results = {}
        iso = IsolationForest(
            n_estimators=300,
            contamination=self.contamination,
            random_state=self.random_seed,
        )
        iso.fit(X)
        iso_scores = -iso.decision_function(X)
        results["iforest"] = dict(model=iso, scores=iso_scores)

        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(X)
        lof_scores = -lof.decision_function(X)
        results["lof"] = dict(model=lof, scores=lof_scores)

        # One-Class SVM
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02)
        ocsvm.fit(X)
        ocsvm_scores = -ocsvm.decision_function(X)
        results["ocsvm"] = dict(model=ocsvm, scores=ocsvm_scores)

        # TODO: save best model
        best_model, best_score = self._get_best_model(results)
        self.pipeline["model"] = best_model
        self.pipeline["train_scores"] = best_score

    def _get_interpretation(self, score, lang="en"):
        if lang == "ar":
            if score < 20:
                return "من غير المحتمل أن يكون لديك MS"
            elif score < 50:
                return "احتمال متوسط للإصابة بمرض MS"
            elif score < 70:
                return "من المحتمل أن يكون لديك MS"
            else:
                return "من المرجح أن يكون لديك MS"
        else:
            if score < 20:
                return "Unlikely to have MS"
            elif score < 50:
                return "Moderate likelihood of MS"
            elif score < 70:
                return "Likely to have MS"
            else:
                return "Highly likely to have MS"

    def analyze_samples(self, samples):
        features, vec = self._extract_features(samples)
        scaler = self.pipeline["scaler"]
        model = self.pipeline["model"]
        train_scores = self.pipeline["train_scores"]

        vecs = scaler.transform(vec)
        score = -model.decision_function(vecs)[0]

        # Map score to 0..100 using training distribution: treat min..95th percentile as 0..100
        ref_min = float(np.min(train_scores))
        ref_high = float(np.percentile(train_scores, 95))

        if ref_high <= ref_min:
            ms_likelihood = float(np.clip((score - ref_min) * 100.0, 0, 100))
        else:
            ms_likelihood = float(
                np.clip((score - ref_min) / (ref_high - ref_min) * 100.0, 0, 100)
            )

        result = {
            "msLikelihoodPercent": round(ms_likelihood, 2),
            "features": features,
            "anomaly_raw": float(score),
            "interpretation_en": self._get_interpretation(ms_likelihood),
            "interpretation_ar": self._get_interpretation(ms_likelihood, "ar"),
        }
        return result

    def save(self, path="model"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.pipeline, f"{path}/pipeline.pkl")


@classmethod
def load(cls, path="model"):
    obj = cls.__new__(cls)
    obj.pipeline = joblib.load(f"{path}/pipeline.pkl")
    return obj
