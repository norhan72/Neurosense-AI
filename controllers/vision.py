import os
import re
import random


class VisionTest:
    def __init__(self, max_attempts=5):
        self.images_folder = os.path.join(os.getcwd(), "data", "vision")
        self.images = []  # list of (image_name, difficulty)
        self.attempts = []
        self.max_attempts = max_attempts
        self._load_images()

    def _load_images(self):
        """
        Reads all image files from folder and extracts difficulty from filename.
        Expected format: anything_diff_0.35.png
        """
        difficulty_regex = re.compile(r"_diff_([0-9]*\.?[0-9]+)")
        for file in os.listdir(self.images_folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                match = difficulty_regex.search(file)
                if match:
                    diff = float(match.group(1))
                    self.images.append((file, diff))

        if not self.images:
            raise ValueError(
                "No valid images found in folder or no difficulty in filenames."
            )

    def get_random_image(self, target_difficulty=None):
        """
        Choose image closest to target difficulty.
        If no target → choose random.
        """
        if target_difficulty is None:
            file, diff = random.choice(self.images)
            return file, diff

        # pick image with closest difficulty
        file, diff = min(self.images, key=lambda x: abs(x[1] - target_difficulty))
        return file, diff

    def record_answer(self, difficulty, seen):
        self.attempts.append((difficulty, seen))

    def compute_threshold(self):
        clears = [d for (d, s) in self.attempts if s]
        blurs = [d for (d, s) in self.attempts if not s]

        if not clears or not blurs:
            return None

        last_clear = max(clears)
        first_blur = min(blurs)
        threshold = (last_clear + first_blur) / 2
        return round(threshold, 3)

    def classify_result(self, threshold):
        if threshold < 0.25:
            return "Excellent vision"
        if threshold < 0.5:
            return "Good vision"
        if threshold < 0.7:
            return "Moderate vision"
        if threshold < 0.85:
            return "Weak vision"
        return "Severe vision impairment"

    def next_step(self):
        if len(self.attempts) >= self.max_attempts:
            threshold = self.compute_threshold()
            label = self.classify_result(threshold)
            return {
                "status": "finished",
                "vision_score": threshold,
                "interpretation": label,
            }

        if len(self.attempts) == 0:
            next_img, next_diff = self.get_random_image()
            return {"status": "continue", "difficulty": next_diff, "image": next_img}

        # adaptive difficulty logic
        last_difficulty, seen = self.attempts[-1]
        target = 0.0

        if seen:
            target = min(last_difficulty + 0.15, 1.0)
        else:
            target = max(last_difficulty - 0.15, 0.0)

        next_img, next_diff = self.get_random_image(target)
        return {"status": "continue", "difficulty": next_diff, "image": next_img}
