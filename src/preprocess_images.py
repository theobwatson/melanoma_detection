"""
Pipeline:
  1) remove hair (black-hat + inpaint)
  2) boost contrast
  3) light sharpen
"""

# imports
import argparse
from pathlib import Path
import cv2  # OpenCV

extensions = (".jpg", ".jpeg")


# step 1: hair removal
def hair_removal_blackhat_inpaint(bgr, kernel_size=21, thresh=10, inpaint_radius=1):
    # find thin dark hairs using black-hat on grayscale, then inpaint them
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, thresh, 255, cv2.THRESH_BINARY)
    clean = cv2.inpaint(bgr, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return clean


# step 2: contrast (CLAHE)
def apply_clahe_bgr(bgr, clip_limit=2.0, tile_grid=8):
    # CLAHE on L channel so colours stay natural
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    L2 = clahe.apply(L)
    lab2 = cv2.merge([L2, A, B])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)


# step 3: sharpen (unsharp mask)
def unsharp(bgr, radius=1.0, amount=0.8):
    k = max(3, int(round(radius * 3) * 2 + 1))
    blur = cv2.GaussianBlur(bgr, (k, k), radius)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


# resize
def maybe_resize(bgr, size=None, square=False):
    if size is None:
        return bgr
    if square:
        h, w = bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = bgr[y0 : y0 + side, x0 : x0 + side]
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LANCZOS4)
    else:
        new_w = size
        new_h = int(bgr.shape[0] * size / bgr.shape[1])
        return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


def iter_images(in_path):
    p = Path(in_path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in extensions:
                yield f
    elif p.is_file() and p.suffix.lower() in extensions:
        yield p


def imwrite_jpg(path, img, quality=95):
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def progress_bar(i, n, width=30):
    # progress bar
    done = int(width * (i + 1) / max(1, n))
    bar = "█" * done + "·" * (width - done)
    print(f"\r[{bar}] {i+1}/{n}", end="", flush=True)
    if i + 1 == n:
        print()


def process_one(img_path, out_dir, args):
    # read image
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        return False

    # resize first for speed
    if args.resize is not None:
        img = maybe_resize(img, args.resize, square=args.square)

    x = hair_removal_blackhat_inpaint(
        img,
        kernel_size=args.hair_kernel,
        thresh=args.hair_thresh,
        inpaint_radius=args.inpaint,
    )
    x = apply_clahe_bgr(x, clip_limit=args.clip, tile_grid=args.tiles)
    x = unsharp(x, radius=args.sigma, amount=args.amount)

    out_dir.mkdir(parents=True, exist_ok=True)
    imwrite_jpg(out_dir / f"{img_path.stem}.jpg", x, quality=args.quality)
    return True


def main():
    parser = argparse.ArgumentParser(description="Preprocessing")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--hair-kernel", type=int, default=21)
    parser.add_argument("--hair-thresh", type=int, default=10)
    parser.add_argument("--inpaint", type=int, default=1)
    parser.add_argument("--clip", type=float, default=2.0)
    parser.add_argument("--tiles", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--amount", type=float, default=0.8)
    parser.add_argument("--quality", type=int, default=95)

    # size
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
    )
    parser.add_argument("--square", action="store_true")

    args = parser.parse_args()

    files = list(iter_images(args.input))
    if not files:
        print("No images found")
        return

    out_dir = Path(args.output)
    total = len(files)

    for i, f in enumerate(files):
        process_one(f, out_dir, args)
        progress_bar(i, total)


if __name__ == "__main__":
    main()
