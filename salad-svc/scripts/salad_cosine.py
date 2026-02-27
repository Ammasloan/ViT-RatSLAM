import argparse
import numpy as np
from PIL import Image

from service.compute_salad import init_model, extract_embeddings


def load_images(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def main():
    parser = argparse.ArgumentParser(description="Compute SALAD embeddings and cosine similarities.")
    parser.add_argument("images", nargs="+", help="Image paths (>=2)")
    args = parser.parse_args()

    if len(args.images) < 2:
        parser.error("Please provide at least two images.")

    init_model()
    imgs = load_images(args.images)
    emb = extract_embeddings(imgs)  # already L2 normalized

    print(f"Loaded {len(imgs)} images.")
    print(f"Embedding shape: {emb.shape}, dtype={emb.dtype}")

    cos = emb @ emb.T
    print("Cosine similarity matrix:")
    with np.printoptions(precision=4, suppress=True):
        print(cos)


if __name__ == "__main__":
    main()
