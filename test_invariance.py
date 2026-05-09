import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time

# torch.inference_mode was added in 1.9; patch it for older versions
if not hasattr(torch, 'inference_mode'):
    torch.inference_mode = torch.no_grad

# Pure-torch fallback for deformable_sample_project (replaces triton kernel).
def _deformable_sample_project_torch(input, grid, weight, bias, *, is_input_nhwc=False, align_corners=False):
    if not is_input_nhwc:
        inp = input
    else:
        inp = input.permute(0, 3, 1, 2)

    B, C, H, W = inp.shape
    _, N, M, _ = grid.shape

    flat_grid = grid.reshape(B, N * M, 1, 2)
    sampled = F.grid_sample(inp, flat_grid, mode='bilinear',
                             padding_mode='zeros', align_corners=align_corners)
    sampled = sampled[:, :, :, 0].permute(0, 2, 1).reshape(B, N, M, C)

    Cout = weight.shape[0]
    w = weight[:, :, 0, :].permute(2, 1, 0)  # (M, Cin, Cout)

    out = torch.einsum('bnmc,mco->bno', sampled.to(w.dtype), w)
    if bias is not None:
        out = out + bias
    return out


# Monkey-patch triton before any model import
import types

class _AnyAttr:
    def __getattr__(self, name):
        return self
    def __call__(self, *a, **kw):
        if len(a) == 1 and callable(a[0]) and not kw:
            return a[0]
        return self
    def __getitem__(self, item):
        return self

_tl = _AnyAttr()
_tl.constexpr = type('constexpr', (), {})
_tl.int32 = None
_tl.float32 = None

_triton_mod = types.ModuleType('triton')
_triton_mod.jit = lambda fn: fn
_triton_mod.language = _tl
_triton_mod.cdiv = lambda a, b: (a + b - 1) // b
_triton_mod.__getattr__ = lambda self, name: _AnyAttr()

sys.modules['triton'] = _triton_mod
sys.modules['triton.language'] = _tl
sys.modules['triton.runtime'] = _AnyAttr()
sys.modules['triton.runtime.jit'] = _AnyAttr()

import model.triton_plugin as _tp
_tp.deformable_sample_project = _deformable_sample_project_torch
import model.model as _mm
_mm.deformable_sample_project = _deformable_sample_project_torch

from clidd import CLIDD

# ── Config ────────────────────────────────────────────────────────────────────
ROTATIONS    = [15, 30, 45, 60, 90, 120, 135, 150, 180]
TRANSLATIONS = [(60, 0), (0, 60), (60, 60), (-60, 40), (80, -60)]
MODELS       = ['L64', 'U128']
TOP_K        = 2048


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(path)
    assert img is not None, f"Cannot read {path}"
    return img


def image_to_tensor(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    if torch.cuda.is_available():
        t = t.cuda()
    return t


def _largest_inscribed_rect(w, h, angle_rad):
    """
    Largest axis-aligned rectangle inscribed inside a (w x h) rectangle
    rotated by angle_rad, preserving the original aspect ratio.
    Source: stackoverflow.com/questions/16702966 + leimao/Rotated-Rectangle-Crop-OpenCV
    """
    angle = angle_rad % np.pi
    if angle > np.pi / 2:
        angle = np.pi - angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    bb_w = w * cos_a + h * sin_a   # bounding-box width
    bb_h = w * sin_a + h * cos_a   # bounding-box height
    aspect = w / h

    # Candidate 1: inscribed rect width limited by bounding box width
    iw1 = bb_w;  ih1 = iw1 / aspect
    # Candidate 2: inscribed rect height limited by bounding box height
    ih2 = bb_h;  iw2 = ih2 * aspect

    if ih1 <= bb_h:
        return int(iw1), int(ih1)
    elif iw2 <= bb_w:
        return int(iw2), int(ih2)
    else:
        return int(min(iw1, iw2)), int(min(ih1, ih2))


def rotate_crop(img, angle_deg):
    """
    Rotate img around its own centre (scale=1.0, aspect ratio untouched),
    then crop to the largest black-border-free inscribed rectangle.
    Formula: stackoverflow.com/questions/16702966
    """
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(M[0, 0]);  sin_a = abs(M[0, 1])

    # Full bounding-box canvas so no pixel is clipped during warpAffine
    bb_w = int(h * sin_a + w * cos_a)
    bb_h = int(h * cos_a + w * sin_a)
    M[0, 2] += bb_w / 2.0 - cx
    M[1, 2] += bb_h / 2.0 - cy

    rotated = cv2.warpAffine(img, M, (bb_w, bb_h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=0)

    iw, ih = _largest_inscribed_rect(w, h, np.deg2rad(angle_deg))
    iw = min(iw, bb_w);  ih = min(ih, bb_h)
    ix0 = (bb_w - iw) // 2;  iy0 = (bb_h - ih) // 2
    return rotated[iy0:iy0+ih, ix0:ix0+iw]


def translate_crop(img, tx, ty):
    """
    Return only the valid intersection region of img shifted by (tx, ty).
    No black pixels, no padding — pure crop of the overlapping area.
    """
    h, w = img.shape[:2]
    src_x0 = max(0, -tx);  src_y0 = max(0, -ty)
    src_x1 = min(w, w - tx); src_y1 = min(h, h - ty)
    return img[src_y0:src_y1, src_x0:src_x1].copy()


def extract(model, img):
    t = image_to_tensor(img)
    result = model(t)
    kpts   = result[0]['keypoints'].cpu().numpy()
    desc   = result[0]['descriptors']
    scores = result[0]['scores'].cpu().numpy()
    return kpts, desc, scores


def count_inliers(kpts1, kpts2, idxs1, idxs2):
    if len(idxs1) < 4:
        return 0, len(idxs1)
    pts1 = kpts1[idxs1].astype(np.float32)
    pts2 = kpts2[idxs2].astype(np.float32)
    _, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, 3.0,
                                 maxIters=5000, confidence=0.999)
    if mask is None:
        return 0, len(idxs1)
    return int(mask.sum()), len(idxs1)


def run_matching(model, img_ref, img_query, label):
    kpts1, desc1, _ = extract(model, img_ref)
    kpts2, desc2, _ = extract(model, img_query)
    t0 = time.perf_counter()
    idxs1, idxs2 = model.match(desc1, desc2)
    elapsed = time.perf_counter() - t0
    inliers, total_matches = count_inliers(kpts1, kpts2, idxs1, idxs2)
    return {
        'label':    label,
        'kpts_ref': len(kpts1),
        'kpts_q':   len(kpts2),
        'matches':  total_matches,
        'inliers':  inliers,
        'ms':       elapsed * 1000,
    }


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_match_grid(model, model_name, img_ref, queries, labels, out_path, max_cols=3):
    """Draw a grid of (img1 | transformed_img2) pairs with match lines."""
    rows = []
    h_ref, w_ref = img_ref.shape[:2]
    thumb_h, thumb_w = 240, 320

    for img_q, lbl in zip(queries, labels):
        kpts1, desc1, _ = extract(model, img_ref)
        kpts2, desc2, _ = extract(model, img_q)
        idxs1, idxs2    = model.match(desc1, desc2)
        inliers, total  = count_inliers(kpts1, kpts2, idxs1, idxs2)

        ref_r = cv2.resize(img_ref, (thumb_w, thumb_h))
        q_r   = cv2.resize(img_q,   (thumb_w, thumb_h))
        pair  = np.concatenate([ref_r, q_r], axis=1)

        if total >= 4:
            pts1 = kpts1[idxs1].astype(np.float32)
            pts2 = kpts2[idxs2].astype(np.float32)
            sx = thumb_w / w_ref
            sy = thumb_h / h_ref
            for p1, p2 in zip(pts1[:40], pts2[:40]):
                x1 = int(p1[0] * sx)
                y1 = int(p1[1] * sy)
                x2 = int(p2[0] * sx) + thumb_w
                y2 = int(p2[1] * sy)
                cv2.line(pair, (x1, y1), (x2, y2), (0, 200, 0), 1)

        cv2.putText(pair, f"{lbl}  matches:{total} inliers:{inliers}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(pair)

    # Pad to full grid
    while len(rows) % max_cols:
        rows.append(np.zeros_like(rows[0]))

    grid_rows = [np.concatenate(rows[i:i+max_cols], axis=1)
                 for i in range(0, len(rows), max_cols)]
    grid = np.concatenate(grid_rows, axis=0)

    header = np.full((36, grid.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(header, f"ref=img1  query=img2_transformed  model={model_name}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 220, 80), 2, cv2.LINE_AA)
    grid = np.concatenate([header, grid], axis=0)

    cv2.imwrite(out_path, grid)
    print(f"  Saved -> {out_path}")


def _put_text_rotated(canvas, text, cx, cy, font_scale, color, bg):
    """Render text rotated 90 degrees CCW, centred at (cx, cy)."""
    fh = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, fh, font_scale, 1)
    tmp = np.full((th + 6, tw + 4, 3), list(bg), dtype=np.uint8)
    cv2.putText(tmp, text, (2, th + 2), fh, font_scale, color, 1, cv2.LINE_AA)
    rot = cv2.rotate(tmp, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rh, rw = rot.shape[:2]
    x0 = cx - rw // 2;  y0 = cy - rh // 2
    x1 = x0 + rw;       y1 = y0 + rh
    # clip to canvas bounds
    sx0 = max(0, -x0);  sy0 = max(0, -y0)
    x0  = max(0, x0);   y0  = max(0, y0)
    x1  = min(canvas.shape[1], x1)
    y1  = min(canvas.shape[0], y1)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = rot[sy0:sy0+(y1-y0), sx0:sx0+(x1-x0)]


def _draw_panel(canvas, ox, oy, pw, ph, res_l64, res_u128, panel_title):
    BG       = (250, 250, 250)
    BORDER   = (160, 160, 160)
    GRID_C   = (215, 215, 215)
    TEXT_D   = (30,  30,  30)
    TEXT_LBL = (90,  90,  90)
    COL      = {'L64': (52, 88, 205),  'U128': (42, 170, 90)}
    LIGHT    = {'L64': (160, 180, 235),'U128': (155, 220, 175)}

    n       = len(res_l64)
    # Layout: leave room for rotated x-labels below baseline
    ml = 68   # left margin (Y-axis labels)
    mr = 14
    mt = 46   # top margin (title)
    mb = 90   # bottom margin (x-labels, rotated up to ~80px)

    plot_w = pw - ml - mr
    plot_h = ph - mt - mb

    # Bar geometry: fit all groups into plot_w
    gap_in  = 6                              # gap between two bars in a group
    gap_out = max(8, plot_w // (n * 5))      # gap between groups
    bar_w   = max(14, (plot_w - (n-1)*gap_out - n*gap_in) // (n*2))
    group_w = bar_w * 2 + gap_in
    total_used = n * group_w + (n-1) * gap_out
    x_start = ox + ml + (plot_w - total_used) // 2

    max_inl = max(max(r['inliers'] for r in res_l64 + res_u128), 1)
    base_y  = oy + mt + plot_h

    # Panel background & border
    cv2.rectangle(canvas, (ox, oy), (ox+pw, oy+ph), BG, -1)
    cv2.rectangle(canvas, (ox, oy), (ox+pw, oy+ph), BORDER, 1)

    # Title
    cv2.putText(canvas, panel_title,
                (ox + ml, oy + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_D, 2, cv2.LINE_AA)

    # Y-axis ticks + grid lines
    n_ticks = 5
    for ti in range(n_ticks + 1):
        val = int(max_inl * ti / n_ticks)
        gy  = base_y - int(plot_h * ti / n_ticks)
        cv2.line(canvas, (ox+ml, gy), (ox+ml+plot_w, gy), GRID_C, 1)
        label = str(val)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
        cv2.putText(canvas, label,
                    (ox + ml - lw - 6, gy + lh//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, TEXT_LBL, 1, cv2.LINE_AA)

    # Y-axis title "Inliers" rotated
    _put_text_rotated(canvas, "Inliers",
                      ox + 14, oy + mt + plot_h // 2,
                      0.40, TEXT_LBL, (255, 255, 255))

    # Axes
    cv2.line(canvas, (ox+ml, oy+mt), (ox+ml, base_y), (80,80,80), 2)
    cv2.line(canvas, (ox+ml, base_y), (ox+ml+plot_w, base_y), (80,80,80), 2)

    # Bars + labels
    for i, (rl, ru) in enumerate(zip(res_l64, res_u128)):
        gx = x_start + i * (group_w + gap_out)
        group_cx = gx + group_w // 2

        for r, mn, boff in [(rl, 'L64', 0), (ru, 'U128', bar_w + gap_in)]:
            inl = r['inliers']
            hb  = max(int(inl / max_inl * plot_h), 3)
            x   = gx + boff
            yt  = base_y - hb
            mid = yt + hb // 2

            # Two-tone bar
            cv2.rectangle(canvas, (x, yt),  (x+bar_w, mid),    LIGHT[mn], -1)
            cv2.rectangle(canvas, (x, mid), (x+bar_w, base_y), COL[mn],   -1)
            cv2.rectangle(canvas, (x, yt),  (x+bar_w, base_y), COL[mn],    1)

            # Inlier value above bar — centred, never overlapping bar top
            val_str = str(inl)
            (vw, vh), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 0.36, 1)
            vx = x + (bar_w - vw) // 2
            vy = max(yt - 5, oy + mt + vh + 2)
            cv2.putText(canvas, val_str, (vx, vy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, COL[mn], 1, cv2.LINE_AA)

        # X-axis label: rotated, centred under group
        _put_text_rotated(canvas, rl['label'],
                          group_cx, base_y + 46,
                          0.34, TEXT_LBL, BG)


def draw_combined_comparison(rot_l64, rot_u128, tr_l64, tr_u128, out_path):
    PAD      = 24
    HEADER_H = 56
    LEGEND_H = 44
    PW, PH   = 820, 440
    IMG_W    = PAD + PW + PAD + PW + PAD
    IMG_H    = PAD + HEADER_H + PH + LEGEND_H + PAD

    canvas = np.full((IMG_H, IMG_W, 3), 255, dtype=np.uint8)

    # Header
    title = "CLIDD Geometric Invariance: L64 vs U128  (ref=img1, query=img2 transformed)"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)
    cv2.putText(canvas, title,
                ((IMG_W - tw) // 2, PAD + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(canvas,
             (PAD, PAD + HEADER_H - 6),
             (IMG_W - PAD, PAD + HEADER_H - 6),
             (180, 180, 180), 1)

    py = PAD + HEADER_H
    _draw_panel(canvas, PAD,            py, PW, PH,
                rot_l64, rot_u128, "Rotation Invariance (inliers)")
    _draw_panel(canvas, PAD + PW + PAD, py, PW, PH,
                tr_l64,  tr_u128,  "Translation Invariance (inliers)")

    # Legend
    COL = {'L64': (52, 88, 205), 'U128': (42, 170, 90)}
    ly  = py + PH + 12
    lx  = PAD + 10
    for mn, col in COL.items():
        cv2.rectangle(canvas, (lx, ly+6), (lx+18, ly+22), col, -1)
        cv2.rectangle(canvas, (lx, ly+6), (lx+18, ly+22), (80,80,80), 1)
        cv2.putText(canvas, mn, (lx + 24, ly + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30,30,30), 1, cv2.LINE_AA)
        lx += 100
    cv2.putText(canvas,
                "Numbers above bars = RANSAC inliers",
                (lx + 20, ly + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,100,100), 1, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f"  Saved -> {out_path}")


def print_table(title, res_l64, res_u128):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")
    print(f"  {'Transform':<22} {'Model':<7} {'Kpts1':>6} {'Kpts2':>6} "
          f"{'Matches':>8} {'Inliers':>8} {'ms':>7}")
    print('  ' + '-' * 74)
    for rl, ru in zip(res_l64, res_u128):
        for r, mn in [(rl, 'L64'), (ru, 'U128')]:
            print(f"  {r['label']:<22} {mn:<7} {r['kpts_ref']:>6} {r['kpts_q']:>6} "
                  f"{r['matches']:>8} {r['inliers']:>8} {r['ms']:>7.1f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('results', exist_ok=True)

    print("Loading models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    models = {}
    for mn in MODELS:
        m = CLIDD(mn, top_k=TOP_K, radius=2).eval()
        if torch.cuda.is_available():
            m = m.cuda()
        models[mn] = m
        print(f"  Loaded {mn}")

    img1 = load_image('images/1.jpg')
    img2 = load_image('images/2.jpg')
    print(f"  img1: {img1.shape}  img2: {img2.shape}")

    # Resize both to same size for a clean comparison
    H, W = img1.shape[:2]
    img2 = cv2.resize(img2, (W, H))

    # ── Rotation: ref=img1, query=img2 rotated around its own centre ──
    # rotate_crop uses the mathematically correct inscribed-rectangle formula
    # (stackoverflow.com/questions/16702966) — aspect ratio preserved, no black borders.
    img1_crop = img1  # full image as reference

    rot_pairs  = [(rotate_crop(img2, a), f"rot_{a}deg") for a in ROTATIONS]
    rot_q      = [p[0] for p in rot_pairs]
    rot_labels = [p[1] for p in rot_pairs]

    rot_res = {mn: [run_matching(models[mn], img1_crop, q, lbl)
                    for q, lbl in rot_pairs]
               for mn in MODELS}

    print_table("Rotation Invariance  (ref=img1_crop, query=img2_crop rotated from centre)",
                rot_res['L64'], rot_res['U128'])

    for mn in MODELS:
        draw_match_grid(models[mn], mn, img1_crop, rot_q, rot_labels,
                        f"results/{mn}_rotation_grid.jpg")


    # ── Translation: ref=img1 valid crop, query=img2 translated valid crop ──
    # Crop the overlapping region so no black/reflected pixels appear.
    tr_pairs  = [(translate_crop(img2, tx, ty), f"tx{tx:+d}_ty{ty:+d}")
                 for tx, ty in TRANSLATIONS]
    # Corresponding ref crops (same overlap region from img1)
    tr_ref_crops = []
    for tx, ty in TRANSLATIONS:
        sx0 = max(0,  tx);  sy0 = max(0,  ty)
        sx1 = min(W, W+tx); sy1 = min(H, H+ty)
        tr_ref_crops.append(img1[sy0:sy1, sx0:sx1].copy())

    tr_q      = [p[0] for p in tr_pairs]
    tr_labels = [p[1] for p in tr_pairs]

    tr_res = {mn: [run_matching(models[mn], tr_ref_crops[i], tr_q[i], tr_labels[i])
                   for i in range(len(tr_pairs))]
              for mn in MODELS}

    print_table("Translation Invariance  (ref=img1_valid_crop, query=img2_translated_valid_crop)",
                tr_res['L64'], tr_res['U128'])

    for mn in MODELS:
        # draw_match_grid expects a single ref; use img1_crop for display context
        draw_match_grid(models[mn], mn, img1_crop, tr_q, tr_labels,
                        f"results/{mn}_translation_grid.jpg")

    draw_combined_comparison(
        rot_res['L64'], rot_res['U128'],
        tr_res['L64'],  tr_res['U128'],
        "results/comparison.jpg"
    )

    # ── Overall summary ──
    print("=" * 78)
    print("  OVERALL SUMMARY  (mean inliers across all transforms)")
    print("=" * 78)
    print(f"  {'Experiment':<35} {'L64':>8} {'U128':>8}")
    print("  " + "-" * 54)
    for tname, res in [('Rotation', rot_res), ('Translation', tr_res)]:
        l64_avg  = np.mean([r['inliers'] for r in res['L64']])
        u128_avg = np.mean([r['inliers'] for r in res['U128']])
        print(f"  {tname:<35} {l64_avg:>8.1f} {u128_avg:>8.1f}")

    print("\nDone. Results saved in ./results/")


if __name__ == '__main__':
    main()
