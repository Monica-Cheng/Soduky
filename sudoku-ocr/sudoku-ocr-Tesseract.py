
import argparse
import os
import cv2
import numpy as np
import pytesseract

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

#Main pipeline: detect board- warp -split- OCR- save txt
def recognize(image_path, output_path, board_size, inner_crop, min_ink):
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
    arr = np.array(digits, dtype=int)
    np.savetxt(output_path, arr, fmt="%d")
    print(arr)
    print(output_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="recognized_sudoku.txt")
    p.add_argument("--board_size", type=int, default=900)
    p.add_argument("--inner_crop", type=float, default=0.22)
    p.add_argument("--min_ink", type=float, default=0.03)
    args = p.parse_args()
    recognize(args.image, args.out, args.board_size, args.inner_crop, args.min_ink)

if __name__ == "__main__":
    main()
