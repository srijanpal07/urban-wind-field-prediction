"""
Geometry Loader
Converts a .stl file (solid buildings + ground) to a 2D binary obstacle mask
via horizontal-slice projection.

Auto-detects the vertical axis as the dimension with the smallest bounding-box
range (e.g. Y-up models common in CAD/wind-tunnel exports).
"""

import numpy as np
import os


def load_stl_binary(filepath):
    """Parse binary STL, return triangles as [N, 3, 3] array (N triangles, 3 verts, xyz)."""
    with open(filepath, 'rb') as f:
        f.read(80)  # header
        n_tri = np.frombuffer(f.read(4), dtype=np.uint32)[0]
        triangles = []
        for _ in range(n_tri):
            f.read(12)  # normal
            verts = np.frombuffer(f.read(36), dtype=np.float32).reshape(3, 3)
            triangles.append(verts)
            f.read(2)  # attribute
    return np.array(triangles)


def load_stl_ascii(filepath):
    """Parse ASCII STL, return triangles as [N, 3, 3] array."""
    triangles = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('facet normal'):
            verts = []
            i += 2  # skip 'outer loop'
            for _ in range(3):
                parts = lines[i].strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                i += 1
            triangles.append(verts)
        i += 1
    return np.array(triangles)


def load_stl(filepath):
    """Auto-detect binary vs ASCII STL and load."""
    with open(filepath, 'rb') as f:
        header = f.read(5)
    if header == b'solid':
        try:
            return load_stl_ascii(filepath)
        except Exception:
            return load_stl_binary(filepath)
    else:
        return load_stl_binary(filepath)


def stl_to_obstacle_mask(stl_path: str, grid_size: int = 128,
                          slice_height: float = None,
                          up_axis: str = 'auto',
                          margin_frac: float = 0.05):
    """
    Project STL geometry to 2D binary obstacle mask.

    Auto-detects the vertical axis as the dimension with the smallest
    bounding-box range (handles both Z-up and Y-up STL exports).

    Parameters
    ----------
    stl_path     : path to .stl file
    grid_size    : resolution of output grid (grid_size x grid_size)
    slice_height : absolute height along up_axis to slice at
                   (None = 30% of up-axis range, ~20-30m real at 1:40 scale)
    up_axis      : 'auto' | 'x' | 'y' | 'z'
    margin_frac  : fractional margin (unused, kept for API compat)

    Returns
    -------
    mask   : np.ndarray bool [grid_size, grid_size], True = solid
    bounds : dict with geometry info
    """
    print(f"Loading STL: {stl_path}")
    triangles = load_stl(stl_path)
    print(f"  Loaded {len(triangles)} triangles")

    all_verts = triangles.reshape(-1, 3)
    mins  = all_verts.min(axis=0)
    maxs  = all_verts.max(axis=0)
    spans = maxs - mins
    axis_names = ['x', 'y', 'z']

    print(f"  Bounds X: [{mins[0]:.3f}, {maxs[0]:.3f}]  (span {spans[0]:.1f})")
    print(f"  Bounds Y: [{mins[1]:.3f}, {maxs[1]:.3f}]  (span {spans[1]:.1f})")
    print(f"  Bounds Z: [{mins[2]:.3f}, {maxs[2]:.3f}]  (span {spans[2]:.1f})")

    # ── Detect vertical axis ──────────────────────────────────────────────
    if up_axis == 'auto':
        up_idx = int(np.argmin(spans))
        print(f"  Up axis: {axis_names[up_idx].upper()} "
              f"(smallest span {spans[up_idx]:.1f} — auto-detected)")
    else:
        up_idx = axis_names.index(up_axis.lower())
        print(f"  Up axis: {axis_names[up_idx].upper()} (user-specified)")

    horiz_indices = [i for i in range(3) if i != up_idx]
    # Swap so that the second horizontal axis (Z for Y-up models) becomes h0→cols
    # and the first (X) becomes h1→rows.  This makes Z the horizontal display axis
    # and X the vertical axis, matching the requested 90° CW orientation.
    h0_idx, h1_idx = horiz_indices[1], horiz_indices[0]

    up_min,  up_max  = mins[up_idx],  maxs[up_idx]
    h0_min,  h0_max  = mins[h0_idx],  maxs[h0_idx]
    h1_min,  h1_max  = mins[h1_idx],  maxs[h1_idx]

    # ── Slice height ──────────────────────────────────────────────────────
    # Default: 30% up from base ≈ 20–30 m real altitude at 1:40 scale
    if slice_height is None:
        slice_height = up_min + 0.3 * (up_max - up_min)
    print(f"  Slicing at {axis_names[up_idx].upper()} = {slice_height:.2f}  "
          f"({100*(slice_height-up_min)/(up_max-up_min):.0f}% of height range)")

    # ── Rasterise triangles onto horizontal grid ──────────────────────────
    # Strategy: include every triangle whose highest vertex reaches or exceeds
    # the slice height (the building material is present at that altitude).
    # Use proper 2D polygon rasterization (not bounding boxes) so diagonal
    # and angular buildings are rendered with their actual footprint shapes.
    # Roof triangles (horizontal, the shape-defining faces) are now included.
    from skimage.draw import polygon as skpoly, line as skline
    H = W = grid_size
    mask = np.zeros((H, W), dtype=bool)
    dh0 = (h0_max - h0_min) / W
    dh1 = (h1_max - h1_min) / H

    for tri in triangles:
        up_verts = tri[:, up_idx]

        # Only skip triangles entirely below the slice (ground plane, short curbs)
        if up_verts.max() < slice_height - 1e-6:
            continue

        h0_verts = tri[:, h0_idx]
        h1_verts = tri[:, h1_idx]

        # Convert to pixel coordinates (col = Z axis, row = X axis inverted)
        # Z_min → col 0 (array left, displayed on right after axis inversion)
        # X_max → row 0 (bottom), X_min → row H-1 (top) — X increases downward in display
        cols = np.clip((h0_verts - h0_min) / dh0, 0, W - 1)
        rows = np.clip((h1_max - h1_verts) / dh1, 0, H - 1)

        # Filled polygon (handles horizontal roof triangles → correct diagonal shapes)
        rr, cc = skpoly(rows, cols, shape=(H, W))
        mask[rr, cc] = True

        # Perimeter lines (handles near-vertical wall triangles whose 2D projection
        # degenerates to a line segment — marks the building outline)
        for i in range(3):
            j = (i + 1) % 3
            r0, c0 = int(round(rows[i])), int(round(cols[i]))
            r1, c1 = int(round(rows[j])), int(round(cols[j]))
            lrr, lcc = skline(r0, c0, r1, c1)
            lrr = np.clip(lrr, 0, H - 1)
            lcc = np.clip(lcc, 0, W - 1)
            mask[lrr, lcc] = True

    # Fill enclosed building interiors (wall outlines form closed polygons)
    from scipy.ndimage import binary_fill_holes
    mask = binary_fill_holes(mask)

    # Strip 2-pixel border (removes ground-plane edge artefacts)
    border = np.zeros_like(mask)
    border[0:2, :] = True; border[-2:, :] = True
    border[:, 0:2] = True; border[:, -2:] = True
    mask = mask & ~border

    n_solid = mask.sum()
    print(f"  Obstacle mask: {n_solid} solid cells "
          f"({100*n_solid/mask.size:.1f}% of domain)")

    bounds = dict(
        up_axis=axis_names[up_idx],
        slice_height=slice_height,
        h0_min=h0_min, h0_max=h0_max,
        h1_min=h1_min, h1_max=h1_max,
        up_min=up_min, up_max=up_max,
    )
    return mask, bounds


def make_synthetic_city(grid_size: int = 128, seed: int = 42):
    """
    Generate a synthetic city obstacle mask when no STL is available.
    Creates a regular grid of rectangular buildings of varying sizes.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    H, W = grid_size, grid_size

    block_size = grid_size // 8
    street_width = max(2, block_size // 5)

    for row in range(8):
        for col in range(8):
            y0 = row * block_size + street_width
            y1 = (row + 1) * block_size - street_width
            x0 = col * block_size + street_width
            x1 = (col + 1) * block_size - street_width

            shrink_y = rng.integers(0, block_size // 6)
            shrink_x = rng.integers(0, block_size // 6)
            y0 += shrink_y; y1 -= shrink_y
            x0 += shrink_x; x1 -= shrink_x

            if rng.random() < 0.15:
                continue

            if y1 > y0 and x1 > x0:
                mask[y0:y1, x0:x1] = True

    print(f"Synthetic city: {mask.sum()} solid cells ({100*mask.sum()/mask.size:.1f}%)")
    return mask
