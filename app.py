# streamlit_app.py

# Install dependencies:
# pip install streamlit opencv-python-headless numpy scikit-learn "pulp[cbc]"

import streamlit as st
import numpy as np
import cv2
from sklearn.cluster import KMeans
import pulp as pl
from pathlib import Path
import tempfile


# ============================================================
# Grid extraction
# ============================================================

def extract_grid_cells(
    image_path,
    grid_size=9,
    border=5,
    black_thresh=50,
    dark_text_thresh=90
):
    """
    Load an image, remove the outer border, split it into a grid,
    and estimate the dominant/median RGB color of every cell.

    Returns
    -------
    cells : dict
        {(row, col): rgb_median}

    colors : np.ndarray
        Shape (grid_size * grid_size, 3)
    """

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise FileNotFoundError(image_path)

    H, W = img_bgr.shape[:2]

    # Prevent invalid cropping
    border = max(
        0,
        min(
            border,
            H // 10,
            W // 10
        )
    )

    img_bgr = img_bgr[
        border:H - border,
        border:W - border
    ].copy()

    h, w = img_bgr.shape[:2]

    cell_h = h // grid_size
    cell_w = w // grid_size

    cells = {}
    colors = []

    for i in range(grid_size):
        for j in range(grid_size):

            # Handle last row/column safely
            y1 = i * cell_h
            y2 = (i + 1) * cell_h if i < grid_size - 1 else h

            x1 = j * cell_w
            x2 = (j + 1) * cell_w if j < grid_size - 1 else w

            cell = img_bgr[y1:y2, x1:x2]

            if cell.size == 0:
                raise ValueError(
                    f"Cell ({i}, {j}) is empty. "
                    f"Check grid size or border crop."
                )

            # Convert cell to grayscale
            gray = cv2.cvtColor(
                cell,
                cv2.COLOR_BGR2GRAY
            )

            # Ignore very dark pixels:
            # grid lines, black text, symbols, etc.
            good = (
                (gray > black_thresh)
                &
                (gray > dark_text_thresh)
            )

            # If too few pixels remain, relax filtering
            if np.count_nonzero(good) < 30:
                good = gray > black_thresh

            pixels = cell[good]

            # Ultimate fallback
            if pixels.size == 0:
                pixels = cell.reshape(-1, 3)

            # Median is more robust than mean
            med_bgr = np.median(
                pixels.astype(np.float32),
                axis=0
            )

            # OpenCV = BGR
            # KMeans input = RGB
            med_rgb = med_bgr[::-1]

            cells[(i, j)] = med_rgb
            colors.append(med_rgb)

    return (
        cells,
        np.array(colors, dtype=np.float32)
    )


# ============================================================
# Color clustering
# ============================================================

def create_baskets_from_colors(
    cells,
    colors,
    n_clusters=9
):
    """
    Cluster grid cells into color baskets using KMeans in LAB
    color space.

    Returns
    -------
    baskets : dict
        cluster_id -> [(row, col), ...]

    labels : np.ndarray
        Cluster assignment for each cell
    """

    coords = list(cells.keys())

    if len(colors) < n_clusters:
        raise ValueError(
            f"Cannot create {n_clusters} clusters from "
            f"only {len(colors)} cells."
        )

    # RGB values -> uint8
    rgb_u8 = np.clip(
        colors,
        0,
        255
    ).astype(np.uint8)

    rgb_u8 = rgb_u8.reshape(-1, 1, 3)

    # RGB -> LAB
    # LAB tends to separate perceived colors better
    lab = cv2.cvtColor(
        rgb_u8,
        cv2.COLOR_RGB2LAB
    )

    lab = lab.reshape(
        -1,
        3
    ).astype(np.float32)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(lab)

    baskets = {}

    for coord, label in zip(coords, labels):
        basket_id = int(label)

        if basket_id not in baskets:
            baskets[basket_id] = []

        baskets[basket_id].append(coord)

    return baskets, labels


# ============================================================
# PuLP Optimizer
# ============================================================

def select_one_cell_per_basket_row_col_no_diag(
    baskets,
    n_rows,
    n_cols,
    exact_one_per_row_col=True,
    solver=None
):
    """
    Select exactly one cell from each color basket while ensuring:

    1. Exactly one selected cell per basket
    2. Exactly one selected cell per row (optional)
    3. Exactly one selected cell per column (optional)
    4. No two selected cells touch diagonally

    Compatible with PuLP 4.x.
    """

    B = list(baskets.keys())

    if len(B) == 0:
        raise ValueError("No color baskets were found.")

    # For a standard Queens puzzle we normally expect
    # number of baskets == grid size.
    if exact_one_per_row_col and len(B) != n_rows:
        raise ValueError(
            f"There are {len(B)} baskets but {n_rows} rows. "
            "For exact one queen per row/column, these normally "
            "need to match."
        )

    # --------------------------------------------------------
    # Create model FIRST.
    # PuLP 4 requires variables to belong to a problem.
    # --------------------------------------------------------

    prob = pl.LpProblem(
        "Queens_Color_Basket_Solver",
        pl.LpMinimize
    )

    # Dummy objective because we only care about feasibility
    prob += 0

    # --------------------------------------------------------
    # x[b,i,j]
    #
    # x = 1 means cell (i,j) from basket b is selected.
    # --------------------------------------------------------

    x = {}

    for b in B:

        if len(baskets[b]) == 0:
            raise ValueError(
                f"Basket {b} contains no cells."
            )

        for i, j in baskets[b]:

            x[(b, i, j)] = prob.add_variable(
                f"x_{b}_{i}_{j}",
                lowBound=0,
                upBound=1,
                cat=pl.LpBinary
            )

    # --------------------------------------------------------
    # Exactly one selected cell from each basket
    # --------------------------------------------------------

    for b in B:

        prob += (
            pl.lpSum(
                x[(b, i, j)]
                for i, j in baskets[b]
            )
            == 1
        )

    # --------------------------------------------------------
    # y[i,j]
    #
    # y = 1 means there is a selected queen in this cell.
    # --------------------------------------------------------

    y = {}

    for i in range(n_rows):
        for j in range(n_cols):

            y[(i, j)] = prob.add_variable(
                f"y_{i}_{j}",
                lowBound=0,
                upBound=1,
                cat=pl.LpBinary
            )

            possible_x = [
                x[(b, i, j)]
                for b in B
                if (b, i, j) in x
            ]

            if possible_x:

                prob += (
                    y[(i, j)]
                    ==
                    pl.lpSum(possible_x)
                )

            else:

                # This should normally never happen,
                # but it is safe to explicitly force 0.
                prob += y[(i, j)] == 0

    # --------------------------------------------------------
    # Row constraints
    # --------------------------------------------------------

    for i in range(n_rows):

        row_sum = pl.lpSum(
            y[(i, j)]
            for j in range(n_cols)
        )

        if exact_one_per_row_col:
            prob += row_sum == 1
        else:
            prob += row_sum <= 1

    # --------------------------------------------------------
    # Column constraints
    # --------------------------------------------------------

    for j in range(n_cols):

        col_sum = pl.lpSum(
            y[(i, j)]
            for i in range(n_rows)
        )

        if exact_one_per_row_col:
            prob += col_sum == 1
        else:
            prob += col_sum <= 1

    # --------------------------------------------------------
    # No diagonal adjacency
    #
    # Only immediate diagonal neighbors are forbidden.
    #
    # Example:
    #
    # Q .
    # . Q
    #
    # is not allowed.
    #
    # NOTE:
    # This is NOT traditional N-Queens diagonal behavior.
    # Queens farther apart on the same diagonal are allowed.
    # --------------------------------------------------------

    for i in range(n_rows - 1):
        for j in range(n_cols - 1):

            # Down-right diagonal
            prob += (
                y[(i, j)]
                +
                y[(i + 1, j + 1)]
                <= 1
            )

            # Down-left diagonal
            prob += (
                y[(i, j + 1)]
                +
                y[(i + 1, j)]
                <= 1
            )

    # --------------------------------------------------------
    # Solver
    # --------------------------------------------------------

    if solver is None:

        # PuLP 4:
        # PULP_CBC_CMD is gone.
        # CBC is accessed using COIN_CMD.
        solver = pl.COIN_CMD(
            msg=False
        )

    stats = prob.solve(solver)

    # --------------------------------------------------------
    # Check solution
    # --------------------------------------------------------

    if not stats.has_solution:

        raise ValueError(
            f"No feasible solution. "
            f"Solver status: {stats.status_str}"
        )

    # --------------------------------------------------------
    # Extract selected cells
    # --------------------------------------------------------

    chosen = {}

    for b in B:

        for i, j in baskets[b]:

            value = x[(b, i, j)].varValue

            if value is not None and value > 0.5:

                chosen[b] = (i, j)

                break

    if len(chosen) != len(B):

        raise ValueError(
            f"The solver returned a solution, but only "
            f"{len(chosen)} of {len(B)} baskets were selected."
        )

    return chosen


# ============================================================
# Visualization
# ============================================================

def draw_chosen_on_grid(
    image_path,
    chosen,
    out_path=None,
    grid_size=9,
    border=5,
    draw_cell_box=True
):
    """
    Draw the solved queen positions on top of the puzzle image.

    Returns
    -------
    RGB image
    """

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise FileNotFoundError(image_path)

    H, W = img_bgr.shape[:2]

    border = max(
        0,
        min(
            border,
            H // 10,
            W // 10
        )
    )

    img_crop = img_bgr[
        border:H - border,
        border:W - border
    ].copy()

    h, w = img_crop.shape[:2]

    cell_h = h // grid_size
    cell_w = w // grid_size

    # --------------------------------------------------------
    # Draw grid
    # --------------------------------------------------------

    if draw_cell_box:

        for i in range(grid_size):
            for j in range(grid_size):

                y1 = i * cell_h

                y2 = (
                    (i + 1) * cell_h
                    if i < grid_size - 1
                    else h
                )

                x1 = j * cell_w

                x2 = (
                    (j + 1) * cell_w
                    if j < grid_size - 1
                    else w
                )

                cv2.rectangle(
                    img_crop,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    1
                )

    # --------------------------------------------------------
    # Draw selected cells
    # --------------------------------------------------------

    for basket_id, (i, j) in chosen.items():

        y1 = i * cell_h

        y2 = (
            (i + 1) * cell_h
            if i < grid_size - 1
            else h
        )

        x1 = j * cell_w

        x2 = (
            (j + 1) * cell_w
            if j < grid_size - 1
            else w
        )

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        radius = max(
            4,
            int(
                min(cell_h, cell_w)
                * 0.18
            )
        )

        # Black outline
        cv2.circle(
            img_crop,
            (cx, cy),
            radius + 2,
            (0, 0, 0),
            thickness=-1
        )

        # White inner circle
        cv2.circle(
            img_crop,
            (cx, cy),
            radius,
            (255, 255, 255),
            thickness=-1
        )

        # Basket number
        text = str(basket_id)

        font = cv2.FONT_HERSHEY_SIMPLEX

        scale = max(
            0.35,
            min(
                0.7,
                min(cell_h, cell_w) / 60
            )
        )

        thickness = 2

        (tw, th), _ = cv2.getTextSize(
            text,
            font,
            scale,
            thickness
        )

        cv2.putText(
            img_crop,
            text,
            (
                cx - tw // 2,
                cy + th // 2
            ),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Paste crop back into original
    # --------------------------------------------------------

    out = img_bgr.copy()

    out[
        border:H - border,
        border:W - border
    ] = img_crop

    if out_path:
        cv2.imwrite(
            out_path,
            out
        )

    return cv2.cvtColor(
        out,
        cv2.COLOR_BGR2RGB
    )


# ============================================================
# Convert basket dictionary to text
# ============================================================

def baskets_to_text(baskets):

    lines = []

    for key in sorted(baskets.keys()):

        lines.append(
            f"Basket {key}:"
        )

        for coord in baskets[key]:

            lines.append(
                f"  {coord}"
            )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Queens Color Puzzle Solver",
    layout="wide"
)

st.title("👑 Queens Solution")

st.write(
    """
    Upload a screenshot of the colored Queens puzzle.

    The application will:

    1. Split the puzzle into grid cells
    2. Detect the color of every cell
    3. Group cells into color baskets
    4. Solve the queen placement constraints
    5. Display the solution
    """
)


# ============================================================
# Sidebar settings
# ============================================================

st.sidebar.header("Settings")

grid_size = st.sidebar.number_input(
    "Grid size",
    min_value=2,
    max_value=20,
    value=9,
    step=1
)

n_clusters = st.sidebar.number_input(
    "Number of colors",
    min_value=2,
    max_value=20,
    value=9,
    step=1
)

border = st.sidebar.slider(
    "Border crop (px)",
    min_value=0,
    max_value=50,
    value=5
)

black_thresh = st.sidebar.slider(
    "Black threshold",
    min_value=0,
    max_value=120,
    value=50
)

dark_text_thresh = st.sidebar.slider(
    "Dark-text threshold",
    min_value=0,
    max_value=200,
    value=90
)

exact_one = st.sidebar.checkbox(
    "Exact 1 per row & column",
    value=True
)


# ============================================================
# PuLP diagnostic information
# ============================================================

with st.sidebar.expander(
    "Solver information"
):

    st.write(
        "PuLP version:",
        getattr(
            pl,
            "__version__",
            "Unknown"
        )
    )

    try:

        available_solvers = pl.listSolvers(
            onlyAvailable=True
        )

    except Exception as e:

        available_solvers = [
            f"Unable to check: {e}"
        ]

    st.write(
        "Available solvers:",
        available_solvers
    )


# ============================================================
# Image uploader
# ============================================================

uploaded = st.file_uploader(
    "Upload a puzzle snapshot",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)

if uploaded is None:

    st.info(
        "Upload an image to start."
    )

    st.stop()


# ============================================================
# Save uploaded image to temporary path
# ============================================================

suffix = (
    Path(uploaded.name).suffix.lower()
    if uploaded.name
    else ".png"
)

with tempfile.NamedTemporaryFile(
    delete=False,
    suffix=suffix
) as tmp:

    tmp.write(
        uploaded.getbuffer()
    )

    img_path = tmp.name


# ============================================================
# Display original
# ============================================================

col1, col2 = st.columns(
    [1, 1]
)

with col1:

    st.subheader(
        "Input snapshot"
    )

    img_bgr = cv2.imread(
        img_path
    )

    if img_bgr is None:

        st.error(
            "Could not read uploaded image."
        )

        st.stop()

    st.image(
        cv2.cvtColor(
            img_bgr,
            cv2.COLOR_BGR2RGB
        ),
        use_container_width=True
    )


# ============================================================
# Run solver button
# ============================================================

run = st.button(
    "Compute baskets + solve",
    type="primary"
)

if not run:
    st.stop()


# ============================================================
# Main processing
# ============================================================

try:

    # --------------------------------------------------------
    # Ensure a solver exists
    # --------------------------------------------------------

    available = pl.listSolvers(
        onlyAvailable=True
    )

    if len(available) == 0:

        st.error(
            """
            No PuLP solver is installed.

            Install CBC using:

            python -m pip install "pulp[cbc]"

            Then restart Streamlit.
            """
        )

        st.stop()

    # --------------------------------------------------------
    # Extract cells
    # --------------------------------------------------------

    cells, colors = extract_grid_cells(
        img_path,
        grid_size=int(grid_size),
        border=int(border),
        black_thresh=int(black_thresh),
        dark_text_thresh=int(
            dark_text_thresh
        )
    )

    expected_cells = (
        int(grid_size)
        *
        int(grid_size)
    )

    st.success(
        f"Cells extracted: "
        f"{len(cells)} / "
        f"{expected_cells}"
    )

    # --------------------------------------------------------
    # Color clustering
    # --------------------------------------------------------

    baskets, labels = (
        create_baskets_from_colors(
            cells,
            colors,
            n_clusters=int(
                n_clusters
            )
        )
    )

    st.write(
        f"🎨 Color baskets detected: "
        f"**{len(baskets)}**"
    )

    # --------------------------------------------------------
    # Show number of cells in each basket
    # --------------------------------------------------------

    basket_sizes = {
        f"Basket {k}": len(v)
        for k, v
        in sorted(
            baskets.items()
        )
    }

    st.write(
        "Basket sizes:",
        basket_sizes
    )

    # --------------------------------------------------------
    # Download basket coordinates
    # --------------------------------------------------------

    baskets_txt = baskets_to_text(
        baskets
    )

    st.download_button(
        "Download baskets.txt",
        data=baskets_txt,
        file_name="baskets.txt",
        mime="text/plain"
    )

    # --------------------------------------------------------
    # Solve puzzle
    # --------------------------------------------------------

    solver = pl.COIN_CMD(
        msg=False
    )

    chosen = (
        select_one_cell_per_basket_row_col_no_diag(
            baskets=baskets,
            n_rows=int(grid_size),
            n_cols=int(grid_size),
            exact_one_per_row_col=exact_one,
            solver=solver
        )
    )

    # --------------------------------------------------------
    # Display solution
    # --------------------------------------------------------

    with col2:

        st.subheader(
            "Solution"
        )

        marked = draw_chosen_on_grid(
            img_path,
            chosen,
            out_path=None,
            grid_size=int(
                grid_size
            ),
            border=int(border),
            draw_cell_box=True
        )

        st.image(
            marked,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Display selected coordinates
    # --------------------------------------------------------

    st.subheader(
        "Chosen cells"
    )

    chosen_json = {
        str(k): [
            int(v[0]),
            int(v[1])
        ]
        for k, v
        in chosen.items()
    }

    st.json(
        chosen_json
    )


# ============================================================
# Error handling
# ============================================================

except Exception as e:

    st.error(
        f"❌ Error: {e}"
    )

    st.exception(e)
