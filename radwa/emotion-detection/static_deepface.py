"""
static_deepface.py
------------------
Static image and batch folder emotion analysis using DeepFace with RetinaFace.
DeepFace is chosen here for its superior accuracy when latency is not a constraint.

Usage:
    python static_deepface.py --image path/to/image.jpg
    python static_deepface.py --folder path/to/images/

Supported image formats: .jpg, .jpeg, .png, .bmp, .webp
"""

import argparse
import os
from pathlib import Path

from deepface import DeepFace

# ── Configuration ────────────────────────────────────────────────────────────
DETECTOR_BACKEND = "retinaface"   # Most accurate; options: retinaface, mtcnn, opencv, ssd
ENFORCE_DETECTION = True          # Set False to skip images where no face is found
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────


def analyze_image(image_path: str) -> dict | None:
    """
    Analyze a single image and return the dominant emotion and full scores.

    Args:
        image_path: Path to the image file.

    Returns:
        dict with keys 'dominant_emotion', 'confidence', 'all_scores', or None on failure.
    """
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=ENFORCE_DETECTION,
        )
        # DeepFace returns a list (one entry per detected face)
        face = results[0]
        dominant = face["dominant_emotion"]
        return {
            "dominant_emotion": dominant,
            "confidence": face["emotion"][dominant],
            "all_scores": face["emotion"],
        }
    except ValueError as e:
        print(f"  [No face detected] {image_path}: {e}")
        return None
    except Exception as e:
        print(f"  [Error] {image_path}: {e}")
        return None


def print_result(image_path: str, result: dict) -> None:
    """Pretty-print the emotion analysis result for one image."""
    name = Path(image_path).name
    print(f"\nImage : {name}")
    print(f"  Dominant emotion : {result['dominant_emotion']}")
    print(f"  Confidence       : {result['confidence']:.2f}%")
    scores_str = ", ".join(
        f"{k}: {v:.2f}" for k, v in sorted(result["all_scores"].items(), key=lambda x: -x[1])
    )
    print(f"  All scores       : {scores_str}")


def run_single(image_path: str) -> None:
    """Analyze a single image file."""
    if not os.path.isfile(image_path):
        print(f"Error: File not found — {image_path}")
        return
    result = analyze_image(image_path)
    if result:
        print_result(image_path, result)


def run_folder(folder_path: str) -> None:
    """Analyze all supported images in a folder."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"Error: Folder not found — {folder_path}")
        return

    images = [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        print(f"No supported images found in {folder_path}")
        return

    print(f"Analyzing {len(images)} image(s) in '{folder_path}'...\n{'─' * 50}")
    success = 0
    for img_path in sorted(images):
        result = analyze_image(str(img_path))
        if result:
            print_result(str(img_path), result)
            success += 1

    print(f"\n{'─' * 50}")
    print(f"Done. {success}/{len(images)} images analyzed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze facial emotions in images using DeepFace."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Path to a single image file.")
    group.add_argument("--folder", type=str, help="Path to a folder of images.")
    args = parser.parse_args()

    if args.image:
        run_single(args.image)
    elif args.folder:
        run_folder(args.folder)


if __name__ == "__main__":
    main()
