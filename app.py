import streamlit as st
import ezdxf, tempfile, os
from collections import defaultdict
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from calcs import (build_grid, spot_height_groups, volume_trapezoidal,
                   volume_simpson, average_end_area, prismoidal_volume,
                   shoelace_area, spot_height_volume, triangular_prism,
                   rectangular_prism)

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

left, right = st.columns([3, 2])

with left:
    st.subheader("Draw around the area you want")
    st.caption("Pick the lasso or box tool in the chart toolbar, then drag.")
    fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="markers",
        marker=dict(size=3, color=zs, colorscale="Viridis",
                    colorbar=dict(title="elev"))))
    fig.update_layout(height=600, dragmode="lasso", margin=dict(l=0, r=0, t=0, b=0))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    ev = st.plotly_chart(fig, on_select="rerun",
                         selection_mode=["lasso", "box"], key="mapsel")

picked = ev.selection.points if ev and ev.selection else []

with right:
    st.subheader("Calculation")
    eq = st.selectbox("Equation", EQUATIONS)
    n_grid = st.slider("Grid resolution (n x n)", 11, 81, 41, step=2)
    if not picked:
        st.info("Draw a shape on the map to select an area.")
        st.stop()
    sel = np.array([(p["x"], p["y"], zs[p["point_index"]]) for p in picked])
    datum = st.number_input("Datum (base level)", value=float(sel[:, 2].min()))

st.divider()
st.write(f"**Selection:** {len(sel):,} points, "
         f"elevation {sel[:, 2].min():g} to {sel[:, 2].max():g}")

if len(sel) < 10:
    st.warning("Too few points selected - draw a bigger area.")
    st.stop()

gx, gy, GZ, dx, dy = build_grid(sel, n_grid)
H = np.clip(GZ - datum, 0, None)
cell = dx * dy
st.caption(f"Grid spacing d = {dx:,.2f} x {dy:,.2f}   (cell area {cell:,.1f})")


def compute(name):
    if name.startswith("1"):
        v, _ = volume_trapezoidal(gx, gy, GZ, dx, dy, datum)
        return v, "volume"
    if name.startswith("2"):
        _, rows = volume_trapezoidal(gx, gy, GZ, dx, dy, datum)
        return average_end_area(rows, dy), "volume"
    if name.startswith("3"):
        hull = [(sel[i, 0], sel[i, 1]) for i in ConvexHull(sel[:, :2]).vertices]
        return shoelace_area(hull), "plan area"
    if name.startswith("4"):
        v, _ = volume_simpson(gx, gy, GZ, dx, dy, datum)
        return v, "volume"
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
            v, kind = compute(name)
            table.append({"Equation": name,
                          "Result": f"{v:,.1f}" if v is not None else "n/a",
                          "Type": kind})
        except Exception as ex:
            table.append({"Equation": name, "Result": f"error: {ex}", "Type": "-"})
    st.table(table)
else:
    v, kind = compute(eq)
    if v is None:
        st.error("Simpson's Rule needs an odd number of grid points.")
    else:
        st.metric(eq, f"{v:,.1f}", help=kind)

st.subheader("Selected surface")
s3 = go.Figure(go.Surface(x=gx, y=gy, z=GZ, colorscale="Viridis"))
s3.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(s3)