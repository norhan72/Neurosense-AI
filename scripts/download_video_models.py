"""Download the MediaPipe models used by the video analysis test into models/.

Usage: python scripts/download_video_models.py
(The API also downloads them lazily on first use.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controllers.video_analysis import MODEL_URLS, ModelUnavailableError, ensure_model  # noqa: E402


def main() -> int:
    failed = False
    for name in MODEL_URLS:
        try:
            path = ensure_model(name)
            print(f"ok  {name}: {path} ({path.stat().st_size / 1e6:.1f} MB)")
        except ModelUnavailableError as exc:
            failed = True
            print(f"ERR {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
