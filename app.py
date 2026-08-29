import streamlit as st
import ezdxf, tempfile, os
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from calcs import (build_grid, spot_height_groups, volume_trapezoidal,
                   volume_simpson, average_end_area, prismoidal_volume,
                   shoelace_area, spot_height_volume, triangular_prism,
                   rectangular_prism)
from scipy.spatial import Delaunay

class Path:
    """Minimal drop-in for matplotlib.path.Path — point-in-polygon only."""
    def __init__(self, vertices):
        self.tri = Delaunay(np.asarray(vertices, float))

    def contains_points(self, points):
        return self.tri.find_simplex(np.asarray(points, float)) >= 0

st.set_page_config(page_title="CAD Terrain Analyser", layout="wide")
st.title("CAD Terrain Analyser")


@st.cache_data
def read_dxf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dxf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        msp = ezdxf.readfile(tmp_path).modelspace()
        out = []
        for e in msp:
            if e.dxftype() == "LWPOLYLINE":
                pts = [(p[0], p[1]) for p in e.get_points()]
                z = float(e.dxf.elevation)
            elif e.dxftype() == "POLYLINE":
                vs = list(e.vertices)
                if not vs:
                    continue
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in vs]
                z = float(vs[0].dxf.location.z)
            else:
                continue
            if len(pts) >= 2:
                out.append((e.dxf.layer, z, pts))
        return out
    finally:
        os.unlink(tmp_path)


def detect_contour_layers(polylines):
    stats = defaultdict(lambda: {"n": 0, "elevs": set()})
    for layer, z, pts in polylines:
        stats[layer]["n"] += 1
        stats[layer]["elevs"].add(round(z, 3))
    found = []
    for layer, s in stats.items():
        nz = {z for z in s["elevs"] if abs(z) > 1e-9}
        if len(nz) >= 2:
            found.append((layer, s["n"], len(nz), min(nz), max(nz)))
    found.sort(key=lambda r: -r[2])
    return found


EQUATIONS = ["Compare all", "1 - Trapezoidal Rule", "2 - Average End Area",
             "3 - Shoelace Formula", "4 - Simpson's Rule", "5 - Spot Height Volume",
             "6 - Triangular Prism", "7 - Rectangular Prism", "8 - Prismoidal Formula"]

EXPLANATIONS = {
    "1": ("Estimates volume by slicing the area into strips, each with a straight-line top, and adding them up.",
          "A = d·[(h₁+hₙ)/2 + h₂ + h₃ + … + hₙ₋₁]"),
    "2": ("Volume between cross-sections: average each pair of neighbouring section areas, times the distance between them.",
          "V = l·[(A₁+Aₙ)/2 + A₂ + … + Aₙ₋₁]"),
    "3": ("Plan area of a polygon computed directly from its corner coordinates.",
          "2A = (x₁y₂ + x₂y₃ + …) − (y₁x₂ + y₂x₃ + …)"),
    "4": ("Like the Trapezoidal Rule but fits curved (parabolic) tops, so it's usually more accurate. Needs an odd number of points.",
          "A = (d/3)·[h₁ + hₙ + 4(h₂+h₄+…) + 2(h₃+h₅+…)]"),
    "5": ("Grid method: each grid-corner height is weighted by how many squares share it.",
          "V = (A/4)·(Σh₁ + 2Σh₂ + 3Σh₃ + 4Σh₄)"),
    "6": ("Volume of a 3-corner prism: average of the 3 corner heights times the base area.",
          "V = (A/3)·(h₁ + h₂ + h₃)"),
    "7": ("Volume of a 4-corner prism: average of the 4 corner heights times the base area.",
          "V = A·(h₁ + h₂ + h₃ + h₄)/4"),
    "8": ("Uses two end sections plus a real measured middle section — more accurate than average-end-area.",
          "V = (l/3)·(A₁ + 4Aₘ + A₂)"),
}


def unit_for(kind):
    return "m²" if kind == "plan area" else "m³"

uploaded = st.file_uploader("Upload a DXF file", type=["dxf"])
if uploaded is None:
    st.info("Upload a DXF file to begin.")
    st.stop()

polylines = read_dxf(uploaded.getvalue())
candidates = detect_contour_layers(polylines)
if not candidates:
    st.error("No contour layer found - no lines in this file carry elevations.")
    st.stop()

labels = [f"{l}  ({n} lines, {k} levels, {lo:g}-{hi:g})"
          for l, n, k, lo, hi in candidates]
pick = st.selectbox("Contour layer (auto-detected)", range(len(labels)),
                    format_func=lambda i: labels[i])
chosen = candidates[pick][0]
contours = [(z, pts) for l, z, pts in polylines if l == chosen]
st.success(f"Loaded {len(contours)} contour lines from layer {chosen}")

xs, ys, zs = [], [], []
for z, pts in contours:
    for (x, y) in pts:
        xs.append(x); ys.append(y); zs.append(z)
xs = np.array(xs); ys = np.array(ys); zs = np.array(zs)

st.subheader("How do you want to select the area?")
method = st.radio("Selection method",
                  ["Draw on map (lasso)", "Click points on map", "Type N, E coordinates"],
                  horizontal=True)

sel = None
poly_xy = None

left, right = st.columns([3, 2])

if method == "Draw on map (lasso)":
    with left:
        st.caption("Pick the lasso or box tool in the chart toolbar, then drag.")
        fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="markers",
            marker=dict(size=3, color=zs, colorscale="Viridis", reversescale=True,
                        colorbar=dict(title="elev"))))
        fig.update_layout(height=600, dragmode="lasso",
                          margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="E (Easting)", yaxis_title="N (Northing)")
        fig.update_xaxes(tickformat=",d", exponentformat="none", hoverformat=",.2f")
        fig.update_yaxes(scaleanchor="x", scaleratio=1,
                         tickformat=",d", exponentformat="none", hoverformat=",.2f")
        ev = st.plotly_chart(fig, on_select="rerun",
                             selection_mode=["lasso", "box"], key="mapsel")
    picked = ev.selection.points if ev and ev.selection else []
    if picked:
        sel = np.array([(p["x"], p["y"], zs[p["point_index"]]) for p in picked])
elif method == "Click points on map":
    with left:
        st.caption("Click one contour point at a time to add a corner. "
                   "Each click adds one corner; build up at least 3.")

        if "clicked_corners" not in st.session_state:
            st.session_state.clicked_corners = []

        cc1, cc2 = st.columns(2)
        if cc1.button("Clear corners"):
            st.session_state.clicked_corners = []
        if cc2.button("Remove last") and st.session_state.clicked_corners:
            st.session_state.clicked_corners.pop()

        fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="markers",
            marker=dict(size=3, color=zs, colorscale="Viridis", reversescale=True,
                        colorbar=dict(title="elev"))))

        corners = st.session_state.clicked_corners
        if corners:
            cx = [c[0] for c in corners]
            cy = [c[1] for c in corners]
            if len(corners) >= 3:            # close the polygon visually
                cx = cx + [corners[0][0]]
                cy = cy + [corners[0][1]]
            fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines+markers",
                                     line=dict(color="orange", width=3),
                                     marker=dict(size=10, color="orange")))

        fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="E (Easting)", yaxis_title="N (Northing)",
                          showlegend=False)
        fig.update_xaxes(tickformat=",d", exponentformat="none", hoverformat=",.2f")
        fig.update_yaxes(scaleanchor="x", scaleratio=1,
                         tickformat=",d", exponentformat="none", hoverformat=",.2f")

        ev = st.plotly_chart(fig, on_select="rerun",
                             selection_mode="points", key="clicksel")

        # add the newest clicked point as a corner
        pts_now = ev.selection.points if ev and ev.selection else []
        if pts_now:
            newest = (pts_now[-1]["x"], pts_now[-1]["y"])
            if not corners or corners[-1] != newest:
                st.session_state.clicked_corners.append(newest)
                st.rerun()

        st.write(f"Corners so far: {len(st.session_state.clicked_corners)}")

        if len(st.session_state.clicked_corners) >= 3:
            poly_xy = st.session_state.clicked_corners
            inside = Path(poly_xy).contains_points(np.column_stack([xs, ys]))
            if inside.sum() >= 4:
                sel = np.column_stack([xs[inside], ys[inside], zs[inside]])
        else:
            st.info("Click at least 3 corners to form a polygon.")
else:
    with left:
        st.caption("Enter each corner as Northing (vertical) and Easting (horizontal), "
                   "in order around the polygon. At least 3 corners.")
        default = pd.DataFrame({
            "N (Northing)": [float(ys.min()), float(ys.min()), float(ys.max())],
            "E (Easting)":  [float(xs.min()), float(xs.max()), float(xs.max())],
        })
        edited = st.data_editor(default, num_rows="dynamic", key="corners",
                                use_container_width=True)
        corners = edited.dropna()
        if len(corners) >= 3:
            poly_xy = [(float(r["E (Easting)"]), float(r["N (Northing)"]))
                       for _, r in corners.iterrows()]
            inside = Path(poly_xy).contains_points(np.column_stack([xs, ys]))
            if inside.sum() >= 4:
                sel = np.column_stack([xs[inside], ys[inside], zs[inside]])

        fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="markers",
            marker=dict(size=3, color=zs, colorscale="Viridis", reversescale=True,
                        colorbar=dict(title="elev"))))
        if poly_xy:
            px = [p[0] for p in poly_xy] + [poly_xy[0][0]]
            py = [p[1] for p in poly_xy] + [poly_xy[0][1]]
            fig.add_trace(go.Scatter(x=px, y=py, mode="lines+markers",
                                     line=dict(color="orange", width=3),
                                     marker=dict(size=8, color="orange"),
                                     name="polygon"))
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis_title="E (Easting)", yaxis_title="N (Northing)",
                          showlegend=False)
        fig.update_xaxes(tickformat=",d", exponentformat="none", hoverformat=",.2f")
        fig.update_yaxes(scaleanchor="x", scaleratio=1,
                         tickformat=",d", exponentformat="none", hoverformat=",.2f")
        st.plotly_chart(fig, key="mapdraw")

with right:
    st.subheader("Calculation")
    eq = st.selectbox("Equation", EQUATIONS)
    n_grid = st.slider("Grid resolution (n x n)", 11, 81, 41, step=2)
    if sel is None:
        st.info("Select an area first (lasso, or type at least 3 corners).")
        st.stop()
    datum = st.number_input("Datum (base level)", value=float(sel[:, 2].min()))

st.divider()
st.write(f"**Selection:** {len(sel):,} points, "
         f"elevation {sel[:, 2].min():g} to {sel[:, 2].max():g}")

if len(sel) < 10:
        st.warning("Too few points selected - pick a bigger area.")
        st.stop()

    # Lasso mode gives us points but no boundary. Derive one from the
    # selection's outer edge so masking and plan area use the same region.
lasso_hull = False
if poly_xy is None:
        try:
            hull_idx = ConvexHull(sel[:, :2]).vertices
            poly_xy = [(float(sel[i, 0]), float(sel[i, 1])) for i in hull_idx]
            lasso_hull = True
        except Exception:
            st.warning("Could not work out a boundary for this selection - "
                       "the points may be in a straight line. Try a wider area.")
            st.stop()

if lasso_hull:
        st.caption("Lasso mode uses the outer boundary of your selected points. "
                   "For an exact shape, use click or coordinate mode.")

gx, gy, GZ, dx, dy = build_grid(sel, n_grid)

    # mask grid cells that fall OUTSIDE the selected polygon
if poly_xy and len(poly_xy) >= 3:
        GXc, GYc = np.meshgrid(gx, gy)
        inside_cell = Path(poly_xy).contains_points(
            np.column_stack([GXc.ravel(), GYc.ravel()])).reshape(GXc.shape)
else:
        inside_cell = np.ones(GZ.shape, dtype=bool)

H = np.where(inside_cell, np.clip(GZ - datum, 0, None), 0.0)
GZ_display = np.where(inside_cell, GZ, np.nan)
cell = dx * dy
st.caption(f"Grid spacing d = {dx:,.2f} x {dy:,.2f}   (cell area {cell:,.1f})")


def compute(name, poly_xy):
    if name.startswith("1"):
        v, _ = volume_trapezoidal(gx, gy, GZ, dx, dy, datum); return v, "volume"
    if name.startswith("2"):
        _, rows = volume_trapezoidal(gx, gy, GZ, dx, dy, datum)
        return average_end_area(rows, dy), "volume"
    if name.startswith("3"):
        return shoelace_area(poly_xy), "plan area"
    if name.startswith("4"):
        v, _ = volume_simpson(gx, gy, GZ, dx, dy, datum); return v, "volume"
    if name.startswith("5"):
        g1, g2, g3, g4 = spot_height_groups(H)
        return spot_height_volume(g1, g2, g3, g4, cell), "volume"
    if name.startswith("6"):
        tot = 0.0
        for i in range(H.shape[0] - 1):
            for j in range(H.shape[1] - 1):
                tot += triangular_prism([H[i, j], H[i, j+1], H[i+1, j]], cell / 2)
                tot += triangular_prism([H[i+1, j+1], H[i, j+1], H[i+1, j]], cell / 2)
        return tot, "volume"
    if name.startswith("7"):
        tot = sum(rectangular_prism([H[i, j], H[i, j+1], H[i+1, j], H[i+1, j+1]], cell)
                  for i in range(H.shape[0] - 1) for j in range(H.shape[1] - 1))
        return tot, "volume"
    if name.startswith("8"):
        _, rows = volume_trapezoidal(gx, gy, GZ, dx, dy, datum)
        m = len(rows) // 2
        return prismoidal_volume(rows[0], rows[m], rows[-1], (gy[-1] - gy[0]) / 2), "volume"


if eq == "Compare all":
    table = []
    for name in EQUATIONS[1:]:
        try:
            v, kind = compute(name, poly_xy)
            unit = unit_for(kind)
            table.append({"Equation": name,
                          "Result": f"{v:,.1f} {unit}" if v is not None else "n/a",
                          "Type": kind})
        except Exception as ex:
            table.append({"Equation": name, "Result": f"error: {ex}", "Type": "-"})
    st.table(table)
else:
    v, kind = compute(eq, poly_xy)
    if v is None:
        st.error("Simpson's Rule needs an odd number of grid points.")
    else:
        plan_area = shoelace_area(poly_xy)

        if kind == "plan area":
            # Shoelace itself is an area - just show the area
            st.metric("Area", f"{v:,.1f} m²")
        else:
            # a volume - show Volume, Area, and average height H = V / A
            c1, c2, c3 = st.columns(3)
            c1.metric("Volume", f"{v:,.1f} m³")
            c2.metric("Area", f"{plan_area:,.1f} m²")
            height = v / plan_area if plan_area else 0
            c3.metric("Height (H = V / A)", f"{height:,.2f} m")

        num = eq.split(" ")[0]
        if num in EXPLANATIONS:
            desc, formula = EXPLANATIONS[num]
            with st.expander("ℹ️ About this equation"):
                st.write(desc)
                st.code(formula)

st.subheader("Selected surface")
s3 = go.Figure(go.Surface(x=gx, y=gy, z=GZ_display, colorscale="Viridis", reversescale=True))
s3.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0),
                 scene=dict(
                     xaxis=dict(title="E (Easting)", tickformat=",d", exponentformat="none"),
                     yaxis=dict(title="N (Northing)", tickformat=",d", exponentformat="none"),
                     zaxis=dict(title="Elevation")))
st.plotly_chart(s3)