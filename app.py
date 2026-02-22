# streamlit_app.py
# pip install streamlit opencv-python numpy scikit-learn pulp

import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import pulp as pl
from pathlib import Path
import tempfile


# -----------------------------
# Grid extraction (robust 9x9 split)
# -----------------------------
def extract_grid_cells(image_path, grid_size=9, border=5, black_thresh=50, dark_text_thresh=90):
    """
    Returns:
      cells: dict {(i,j): rgb_median}
      colors: np.ndarray shape (81,3) RGB float
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    H, W = img_bgr.shape[:2]
    border = max(0, min(border, H // 10, W // 10))
    img_bgr = img_bgr[border:H-border, border:W-border].copy()

    h, w = img_bgr.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size

    cells = {}
    colors = []

    for i in range(grid_size):
        for j in range(grid_size):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = img_bgr[y1:y2, x1:x2]

            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            good = (gray > black_thresh) & (gray > dark_text_thresh)
            if np.count_nonzero(good) < 30:
                good = (gray > black_thresh)

            pixels = cell[good]
            if pixels.size == 0:
                pixels = cell.reshape(-1, 3)

            med_bgr = np.median(pixels.astype(np.float32), axis=0)
            med_rgb = med_bgr[::-1]  # BGR->RGB

            cells[(i, j)] = med_rgb
            colors.append(med_rgb)

    return cells, np.array(colors, dtype=np.float32)


# -----------------------------
# Color baskets (cluster exactly N colors)
# -----------------------------
def create_baskets_from_colors(cells, colors, n_clusters=9):
    coords = list(cells.keys())

    # Cluster in LAB space for better perceptual separation
    rgb_u8 = np.clip(colors, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = kmeans.fit_predict(lab)

    baskets = {}
    for coord, label in zip(coords, labels):
        baskets.setdefault(int(label), []).append(coord)

    return baskets, labels


# -----------------------------
# PuLP optimizer with no diagonal adjacency
# -----------------------------
def select_one_cell_per_basket_row_col_no_diag(baskets, n_rows, n_cols, exact_one_per_row_col=True, solver=None):
    B = list(baskets.keys())

    # x[b,i,j]
    x = {}
    for b in B:
        for (i, j) in baskets[b]:
            x[(b, i, j)] = pl.LpVariable(f"x_{b}_{i}_{j}", 0, 1, cat=pl.LpBinary)

    prob = pl.LpProblem("OneCellPerBasket_UniqueRowCol_NoDiag", pl.LpMinimize)
    prob += 0

    # exactly one per basket
    for b in B:
        prob += pl.lpSum(x[(b, i, j)] for (i, j) in baskets[b]) == 1

    # y[i,j] occupancy
    y = {}
    for i in range(n_rows):
        for j in range(n_cols):
            y[(i, j)] = pl.LpVariable(f"y_{i}_{j}", 0, 1, cat=pl.LpBinary)
            prob += y[(i, j)] == pl.lpSum(x[(b, i, j)] for b in B if (b, i, j) in x)

    # row/col constraints
    for i in range(n_rows):
        expr = pl.lpSum(y[(i, j)] for j in range(n_cols))
        prob += (expr == 1) if exact_one_per_row_col else (expr <= 1)

    for j in range(n_cols):
        expr = pl.lpSum(y[(i, j)] for i in range(n_rows))
        prob += (expr == 1) if exact_one_per_row_col else (expr <= 1)

    # no diagonal adjacency
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            prob += y[(i, j)] + y[(i + 1, j + 1)] <= 1
            prob += y[(i, j + 1)] + y[(i + 1, j)] <= 1

    if solver is None:
        solver = pl.PULP_CBC_CMD(msg=False)

    status = prob.solve(solver)
    if pl.LpStatus[status] != "Optimal":
        raise ValueError(f"No feasible solution. Status: {pl.LpStatus[status]}")

    chosen = {}
    for b in B:
        for (i, j) in baskets[b]:
            if pl.value(x[(b, i, j)]) > 0.5:
                chosen[b] = (i, j)
                break
    return chosen


# -----------------------------
# Visualization
# -----------------------------
def draw_chosen_on_grid(image_path, chosen, out_path=None, grid_size=9, border=5, draw_cell_box=True):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(image_path)

    H, W = img_bgr.shape[:2]
    border = max(0, min(border, H // 10, W // 10))

    img_crop = img_bgr[border:H-border, border:W-border].copy()
    h, w = img_crop.shape[:2]
    cell_h = h // grid_size
    cell_w = w // grid_size

    if draw_cell_box:
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cv2.rectangle(img_crop, (x1, y1), (x2, y2), (0, 0, 0), 1)

    for basket_id, (i, j) in chosen.items():
        y1, y2 = i * cell_h, (i + 1) * cell_h
        x1, x2 = j * cell_w, (j + 1) * cell_w
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        radius = int(min(cell_h, cell_w) * 0.18)
        cv2.circle(img_crop, (cx, cy), radius + 2, (0, 0, 0), thickness=-1)
        cv2.circle(img_crop, (cx, cy), radius, (255, 255, 255), thickness=-1)

        text = str(basket_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        cv2.putText(img_crop, text, (cx - tw // 2, cy + th // 2),
                    font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    out = img_bgr.copy()
    out[border:H-border, border:W-border] = img_crop

    if out_path:
        cv2.imwrite(out_path, out)

    out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out_rgb


def baskets_to_text(baskets):
    lines = []
    for key in sorted(baskets.keys()):
        lines.append(f"Basket {key}:")
        for coord in baskets[key]:
            lines.append(f"  {coord}")
        lines.append("")
    return "\n".join(lines)


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="9x9 Color Basket Solver", layout="wide")
st.title("9×9 Snapshot → Color Baskets → PuLP Solution")

st.sidebar.header("Settings")
grid_size = st.sidebar.number_input("Grid size", min_value=2, max_value=20, value=9, step=1)
n_clusters = st.sidebar.number_input("Number of colors (clusters)", min_value=2, max_value=20, value=9, step=1)
border = st.sidebar.slider("Border crop (px)", 0, 50, 5)
black_thresh = st.sidebar.slider("Black threshold", 0, 120, 50)
dark_text_thresh = st.sidebar.slider("Dark-text threshold", 0, 200, 90)
exact_one = st.sidebar.checkbox("Exact 1 per row & column (==1)", value=True)

uploaded = st.file_uploader("Upload a snapshot (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Upload an image to start.")
    st.stop()

# Save uploaded file to a temp path for cv2
suffix = Path(uploaded.name).suffix.lower() if uploaded.name else ".png"
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.getbuffer())
    img_path = tmp.name

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input snapshot")
    img_bgr = cv2.imread(img_path)
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

run = st.button("Compute baskets + solve")

if not run:
    st.stop()

try:
    cells, colors = extract_grid_cells(
        img_path,
        grid_size=int(grid_size),
        border=int(border),
        black_thresh=int(black_thresh),
        dark_text_thresh=int(dark_text_thresh),
    )

    st.write(f"✅ Cells extracted: **{len(cells)}** (expected {int(grid_size)*int(grid_size)})")

    baskets, labels = create_baskets_from_colors(cells, colors, n_clusters=int(n_clusters))

    # Optional: show baskets text + download
    baskets_txt = baskets_to_text(baskets)
    st.download_button("Download baskets.txt", data=baskets_txt, file_name="baskets.txt", mime="text/plain")

    # Solve
    chosen = select_one_cell_per_basket_row_col_no_diag(
        baskets, int(grid_size), int(grid_size), exact_one_per_row_col=exact_one
    )

    with col2:
        st.subheader("Solution (chosen cells marked)")
        marked = draw_chosen_on_grid(
            img_path, chosen, out_path=None, grid_size=int(grid_size), border=int(border), draw_cell_box=True
        )
        st.image(marked, use_container_width=True)

    st.subheader("Chosen cells (basket → (row,col))")
    st.json({str(k): list(v) for k, v in chosen.items()})  # json-friendly

except Exception as e:
    st.error(f"❌ Error: {e}")
    st.stop()