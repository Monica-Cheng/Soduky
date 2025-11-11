import argparse
import os
import time
import json
import cv2
import numpy as np
import pytesseract

def _ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

# Set Tesseract path (Windows)
def _set_tesseract_cmd():
    cmd = os.environ.get("TESSERACT_CMD", None)
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    else:
        c1 = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        c2 = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        if os.path.exists(c1):
            pytesseract.pytesseract.tesseract_cmd = c1
        elif os.path.exists(c2):
            pytesseract.pytesseract.tesseract_cmd = c2

# Re-order contour points → (TL, TR, BR, BL)
def _order_pts(pts):
    pts = pts.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).reshape(-1)
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl,tr,br,bl], dtype=np.float32)

# Perspective transform to straighten Sudoku board
def _four_point_transform(img, pts, size):
    rect = _order_pts(pts)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (size,size))

# Find biggest 4-point contour (the Sudoku grid)
def _find_largest_quad(bin_img):
    cnts, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    area_best = 0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 10000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4 and a>area_best:
            best = approx
            area_best = a
    return best

# Crop resize threshold (prepare a single cell image)
def _preprocess_cell(bgr, inner_crop):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    cut = int(min(h,w)*inner_crop)
    if h>2*cut and w>2*cut:
        g = g[cut:h-cut, cut:w-cut]
    g = cv2.resize(g, (80,80), interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (3,3), 0)
    th = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    th = cv2.medianBlur(th, 3)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    return th

# OCR a cell, return digit (or 0 if empty)
def _ocr_digit(cell_gray_inv, min_ink):
    ink = (cell_gray_inv>0).mean()
    if ink < min_ink:
        return 0
    cfg1 = r"--oem 1 --psm 10 -l eng -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
    cfg2 = r"--oem 1 --psm 13 -l eng -c tessedit_char_whitelist=0123456789 -c classify_bln_numeric_mode=1"
    s = pytesseract.image_to_string(cell_gray_inv, config=cfg1).strip()
    if not s or not any(ch.isdigit() for ch in s):
        s = pytesseract.image_to_string(cell_gray_inv, config=cfg2).strip()
    for ch in s:
        if ch.isdigit():
            return int(ch)
    return 0


# Core: recognize + metrics
def recognize(image_path, board_size, inner_crop, min_ink):
    _set_tesseract_cmd()
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    ratio = 1200.0 / max(img.shape[:2])
    img = cv2.resize(img, (int(img.shape[1]*ratio), int(img.shape[0]*ratio)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    binv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    quad = _find_largest_quad(binv)
    if quad is None:
        raise RuntimeError("no_sudoku_quad_found")
    warp = _four_point_transform(img, quad, size=board_size)
    step = board_size//9
    digits = []
    for r in range(9):
        row = []
        for c in range(9):
            y0, y1 = r*step, (r+1)*step
            x0, x1 = c*step, (c+1)*step
            cell = warp[y0:y1, x0:x1]
            cell_bin = _preprocess_cell(cell, inner_crop)
            d = _ocr_digit(cell_bin, min_ink)
            row.append(d)
        digits.append(row)
    return np.array(digits, dtype=int)

def _load_ground_truth(gt_path: str) -> np.ndarray:
    """
    Load ground truth Sudoku as 9x9 integers.
    Accepts whitespace or comma separated plain text.
    """
    if gt_path is None:
        return None
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    # Try reading as ints loosely
    with open(gt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    vals = []
    for ln in lines:
        parts = [p for p in ln.replace(",", " ").split() if p]
        row = [int(p) for p in parts]
        vals.append(row)
    arr = np.array(vals, dtype=int)
    if arr.shape != (9,9):
        raise ValueError(f"Ground truth must be 9x9, got {arr.shape}")
    return arr

def _compute_accuracy(pred: np.ndarray, gt: np.ndarray):
    """
    Returns:
      - cell_accuracy: correct cells / 81 (float 0~1)
      - filled_accuracy: correct among gt>0 cells (float 0~1)
      - num_correct, num_total, num_filled
    """
    if gt is None:
        return None
    if pred.shape != gt.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    total = pred.size
    correct = int((pred == gt).sum())
    mask_filled = (gt > 0)
    filled_total = int(mask_filled.sum())
    filled_correct = int((pred[mask_filled] == gt[mask_filled]).sum()) if filled_total > 0 else 0
    return {
        "cell_accuracy": correct / total,
        "filled_accuracy": (filled_correct / filled_total) if filled_total > 0 else None,
        "num_correct": correct,
        "num_total": total,
        "num_filled": filled_total,
        "num_filled_correct": filled_correct
    }

def run_and_log(image_path, out_dir, out_name, board_size, inner_crop, min_ink, gt_path=None):
    _ensure_dir(out_dir)

    # 1) Run with timer
    t0 = time.perf_counter()
    pred = recognize(image_path, board_size, inner_crop, min_ink)
    elapsed = time.perf_counter() - t0

    # 2) Save recognized grid
    out_txt = os.path.join(out_dir, f"{out_name}.txt")
    np.savetxt(out_txt, pred, fmt="%d")

    # 3) Accuracy (optional)
    gt = _load_ground_truth(gt_path) if gt_path else None
    acc = _compute_accuracy(pred, gt) if gt is not None else None

    # 4) Write metrics sidecar files
    metrics = {
        "image": image_path,
        "output_txt": out_txt,
        "runtime_seconds": elapsed,
        "board_size": board_size,
        "inner_crop": inner_crop,
        "min_ink": min_ink,
        "has_ground_truth": bool(gt is not None),
        "metrics": acc
    }
    out_json = os.path.join(out_dir, f"{out_name}.metrics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Also a short human-readable summary
    out_summary = os.path.join(out_dir, f"{out_name}.metrics.txt")
    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Output TXT: {out_txt}\n")
        f.write(f"Runtime: {elapsed:.4f} s\n")
        if acc is not None:
            f.write(f"Accuracy (cells): {acc['num_correct']}/{acc['num_total']} = {acc['cell_accuracy']*100:.2f}%\n")
            f.write("Accuracy: (no ground truth provided)\n")

    # 5) Print brief console output
    print(pred)
    print(f"[Saved] {out_txt}")
    print(f"[Runtime] {elapsed:.4f} s")
    if acc is not None:
        print(f"[Accuracy] cells: {acc['cell_accuracy']*100:.2f}%  ({acc['num_correct']}/{acc['num_total']})")
        

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Path to Sudoku image")
    p.add_argument("--out_dir", default="output", help="Folder to save outputs")
    p.add_argument("--out_name", default="recognized_sudoku", help="Base name for outputs (without extension)")
    p.add_argument("--board_size", type=int, default=900)
    p.add_argument("--inner_crop", type=float, default=0.22)
    p.add_argument("--min_ink", type=float, default=0.03)
    p.add_argument("--gt", default=None, help="Optional path to 9x9 ground-truth TXT for accuracy")
    args = p.parse_args()

    run_and_log(
        image_path=args.image,
        out_dir=args.out_dir,
        out_name=args.out_name,
        board_size=args.board_size,
        inner_crop=args.inner_crop,
        min_ink=args.min_ink,
        gt_path=args.gt
    )

if __name__ == "__main__":
    main()
