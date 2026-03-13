# pip install playwright reportlab
# playwright install chromium

from pathlib import Path
from contextlib import contextmanager
from playwright.sync_api import sync_playwright
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
import tempfile
from PIL import Image
import math
import os

from PIL import Image, ImageChops
from io import BytesIO

@contextmanager
def _launch_browser(width, height, device_scale_factor):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--headless=new",
                "--enable-gpu",
                "--ignore-gpu-blocklist",
                "--use-gl=angle",
                "--use-angle=swiftshader",  # software GL fallback; remove if you have GPU
            ],
        )
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()
        try:
            yield page, context, browser
        finally:
            context.close()
            browser.close()

def _set_camera_js(page, library, selector, azimuth_deg, elevation_deg, distance=None):
    """Attempt a direct camera set for known libraries. Returns True if applied."""
    lib = (library or "").lower().strip()
    args = {
        "selector": selector,
        "azimuth": float(azimuth_deg),
        "elevation": float(elevation_deg),
        "distance": None if distance is None else float(distance),
    }

    if lib == "plotly":
        # Robust, repeatable "front-corner" view for Plotly (Z-up), orthographic.
        ok = page.evaluate("""
        async (args) => {
          const { rootSelector, distance, zScale, sceneIndex, azimuth, elevation } = args;

          const root = document.querySelector(rootSelector);
          // Find the actual plotly graph element
          const plot = root
            ? (root.classList && root.classList.contains('js-plotly-plot') ? root
               : root.querySelector('.js-plotly-plot, .plotly-graph-div'))
            : null;

          if (!plot || !window.Plotly) return false;

          // Which 3D scene to target (scene, scene2, ...)
          const ids = (plot._fullLayout && plot._fullLayout._subplots && plot._fullLayout._subplots.gl3d)
                      || ['scene'];
          const id = ids[Math.min(Math.max(sceneIndex|0, 0), ids.length - 1)];

          const r = (distance && distance > 0) ? distance : 1.8;
          const k = (zScale && zScale > 0) ? zScale : 0.55;
          
          const theta = 0 * Math.PI / 180;   // clockwise = negative angle

          const x0 = r;
          const y0 = r;
          const z0 = r * k;
          
          const x = x0 * Math.cos(theta) - y0 * Math.sin(theta);
          const y = x0 * Math.sin(theta) + y0 * Math.cos(theta);
          const z = z0;
          
          const camera = {
            eye:    { x, y, z },
            center: { x: 0, y: 0, z: 0 },
            up:     { x: 0, y: 0, z: 1 }
          };

          const update = {};
          update[`${id}.camera`]     = camera;
          update[`${id}.aspectmode`] = 'cube';
          update[`${id}.projection`] = { type: 'orthographic' };

          // Some Plotly versions need two relayouts after changing projection
          await Plotly.relayout(plot, update);
          await new Promise(rq => requestAnimationFrame(rq));
          await Plotly.relayout(plot, update);

          return true;
        }
        """, {
            "rootSelector": selector,
            "distance": distance,
            "zScale": 0.55,  # 0.55
            "sceneIndex": 0
        })
        return bool(ok)

    if lib in ("three", "threejs"):
        ok = page.evaluate("""
        (args) => {
          const {selector, azimuth, elevation, distance} = args;
          const toRad = d => d * Math.PI / 180;

          // Robust camera finder that never assumes Array.prototype.find exists
          function pickCamera() {
            // 1) Obvious globals
            if (window.camera && window.camera.isCamera) return window.camera;

            // 2) Anything else global that looks like a camera
            try {
              for (const k in window) {
                const v = window[k];
                if (v && v.isCamera) return v;
              }
            } catch (e) { /* ignore cross-origin weirdness */ }

            // 3) From a global scene, by traversal or by iterating children
            const scene = window.scene;
            if (scene) {
              if (typeof scene.traverse === "function") {
                let found = null;
                scene.traverse(o => { if (!found && o && o.isCamera) found = o; });
                if (found) return found;
              }
              const ch = scene.children;
              if (ch && typeof ch.length === "number") {
                for (let i = 0; i < ch.length; i++) {
                  const o = ch[i];
                  if (o && o.isCamera) return o;
                }
              }
            }
            return null;
          }

          const cam = pickCamera();
          if (!cam) return false;

          // Distance: keep current radius if not provided
          const curR = cam.position ? Math.hypot(cam.position.x||0, cam.position.y||0, cam.position.z||0) : 5;
          const r = (distance != null && distance > 0) ? distance : (curR || 5);

          // Spherical → Cartesian (Y up): azimuth around Y, elevation up from horizon
          const x =  r * Math.cos(toRad(elevation)) * Math.cos(toRad(azimuth));
          const y =  r * Math.sin(toRad(elevation));
          const z =  r * Math.cos(toRad(elevation)) * Math.sin(toRad(azimuth));

          if (cam.position && typeof cam.position.set === "function") {
            cam.position.set(x, y, z);
          } else {
            cam.position = {x, y, z};
          }

          if (cam.lookAt) cam.lookAt(0, 0, 0);

          // Nudge controls if present
          const controls = window.controls || window.orbitControls || window.OrbitControls || null;
          if (controls && typeof controls.update === "function") {
            try { controls.update(); } catch(e) {}
          }

          // Force a render if we can
          const renderer = window.renderer || window.webglRenderer || null;
          const scene = window.scene || null;
          if (renderer && scene && typeof renderer.render === "function") {
            try { renderer.render(scene, cam); } catch(e) {}
          } else if (window.requestAnimationFrame) {
            requestAnimationFrame(()=>{});
          }
          return true;
        }
        """, args)
        return bool(ok)

    if lib in ("deck", "deckgl"):
        ok = page.evaluate("""
        (args) => {
          const {selector, azimuth, elevation} = args;
          if (!window.deckgl) return false;
          const bearing = azimuth;                  // clockwise from north
          const pitch = Math.max(0, Math.min(60, 90 - elevation)); // rough map
          try {
            window.deckgl.setProps({
              viewState: {...(window.deckgl.props.viewState||{}), bearing, pitch}
            });
            return true;
          } catch(e) { return false; }
        }
        """, args)
        return bool(ok)

    if lib in ("echarts", "echarts-gl", "echartsgl"):
        ok = page.evaluate("""
        (args) => {
          const {selector, azimuth, elevation} = args;
          const el = document.querySelector(selector);
          if (!el || !window.echarts) return false;
          const chart = echarts.getInstanceByDom(el);
          if (!chart) return false;
          const alpha = 90 - elevation;  // tilt
          const beta  = azimuth;         // azimuth
          try {
            chart.setOption({grid3D: {viewControl: {alpha, beta}}});
            return true;
          } catch(e) { return false; }
        }
        """, args)
        return bool(ok)

    return False

def _drag_rotate(page, selector, azimuth_deg, elevation_deg):
    """Generic fallback: simulate a mouse drag on the canvas to rotate the view."""
    handle = page.query_selector(selector)
    if not handle:
        raise RuntimeError(f"Could not find element: {selector}")
    box = handle.bounding_box()
    if not box:
        raise RuntimeError("Could not read element bounding box for drag.")

    cx = box["x"] + box["width"] * 0.5
    cy = box["y"] + box["height"] * 0.5
    page.mouse.move(cx, cy)

    # Heuristic: ~5 px per degree horizontally; ~4 px per degree vertically
    dx = azimuth_deg * 5
    dy = -elevation_deg * 4

    page.mouse.down()
    steps = 20
    page.mouse.move(cx + dx, cy + dy, steps=steps)
    page.mouse.up()

from io import BytesIO
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader

def _screenshot_element_to_pdf(page, selector, pdf_path):
    handle = page.query_selector(selector)
    if not handle:
        raise RuntimeError(f"Could not find element: {selector}")
    box = handle.bounding_box()
    if not box:
        raise RuntimeError("Could not read element bounding box for screenshot.")

    # Get PNG as bytes (no filesystem writes)
    png_bytes = page.screenshot(clip=box, type="png")

    # Build PDF from in-memory image
    bio = BytesIO(png_bytes)
    img_reader = ImageReader(bio)
    w, h = img_reader.getSize()

    c = rl_canvas.Canvas(pdf_path, pagesize=(w, h))
    c.drawImage(img_reader, 0, 0, width=w, height=h, mask='auto')
    c.showPage()
    c.save()

def html_3d_to_pdf(
    output_pdf: str,
    *,
    # Where to load the visualization from (pick exactly one)
    url: str | None = None,
    html_path: str | os.PathLike | None = None,
    html_string: str | None = None,
    base_dir: str | os.PathLike | None = None,  # used when html_string has relative assets
    # What to capture
    selector: str = "canvas,div.plotly,div#container",
    width: int = 1280,
    height: int = 960,
    device_scale_factor: float = 2.0,  # for crispness
    wait_for_selector: str | None = None,  # e.g., the canvas/div you need
    wait_ms_after_load: int = 500,
    # View / camera
    library: str | None = None,  # 'plotly'|'three'|'deckgl'|'echarts-gl' or None
    azimuth_deg: float = 45.0,   # rotate around vertical axis (0 = +X, 90 = +Y)
    elevation_deg: float = math.degrees(math.asin(0.55)), # tilt up from horizon (0 = horizon, 90 = top-down)
    distance: float | None = None,  # optional for libs that support it
    # Timeouts
    load_timeout_ms: int = 60000,
) -> str:
    """
    Render a rotatable HTML/WebGL visualization from a fixed angle into a 1-page PDF.

    Returns the absolute path to the created PDF.
    """

    if sum(bool(x) for x in (url, html_path, html_string)) != 1:
        raise ValueError("Provide exactly one of url, html_path, or html_string.")

    if html_string and not base_dir:
        # helps resolve relative assets in HTML strings
        base_dir = os.getcwd()

    output_pdf = str(Path(output_pdf).resolve())

    with _launch_browser(width, height, device_scale_factor) as (page, ctx, br):
        if url:
            page.goto(url, wait_until="load", timeout=load_timeout_ms)
        elif html_path:
            page.goto(Path(html_path).resolve().as_uri(), wait_until="load", timeout=load_timeout_ms)
        else:
            # Load raw HTML string; base_url helps with relative links to scripts/assets
            page.set_content(html_string, wait_until="load", base_url=str(Path(base_dir).resolve().as_uri()))

        # Wait for your visualization element
        target_selector = wait_for_selector or selector
        page.wait_for_selector(target_selector, state="visible", timeout=load_timeout_ms)

        if (library or "").lower() == "plotly":
            try:
                page.wait_for_function(
                    """
                    (sel) => {
                      const root = document.querySelector(sel);
                      const plot = root
                        ? (root.classList && root.classList.contains('js-plotly-plot') ? root
                           : root.querySelector('.js-plotly-plot, .plotly-graph-div'))
                        : null;
                      return !!(plot && window.Plotly && plot._fullLayout);
                    }
                    """,
                    arg=target_selector,
                    timeout=load_timeout_ms,
                )
            except Exception:
                pass

        # Let the scene settle
        page.wait_for_timeout(wait_ms_after_load)

        # Try direct camera control; fall back to simulated drag
        applied = _set_camera_js(page, library, target_selector, azimuth_deg, elevation_deg, distance)
        if not applied:
            if library:  # you explicitly asked for a deterministic view; fail loudly
                raise RuntimeError(f"Could not set camera for library='{library}'. "
                                   f"Check that selector '{target_selector}' points to the Plotly graph element.")
            _drag_rotate(page, target_selector, azimuth_deg, elevation_deg)
        # Give the renderer a moment to draw the new frame
        page.wait_for_timeout(250)

        # Screenshot and build the PDF
        _screenshot_element_to_pdf(page, target_selector, output_pdf)

    return output_pdf


# --------------------
# Example usages
# --------------------
# 1) Plotly figure saved to 'figure.html' (3D scatter/surface/mesh):
# pdf_path = html_3d_to_pdf(
#     output_pdf="figure_view.pdf",
#     html_path="figure.html",
#     selector="div.plotly",   # Plotly renders into a <div class="plotly ...">
#     library="plotly",
#     azimuth_deg=120,
#     elevation_deg=25
# )
# print("Saved:", pdf_path)
#
# 2) three.js app that exposes window.camera/renderer/scene:
# pdf_path = html_3d_to_pdf(
#     output_pdf="scene_view.pdf",
#     url="http://localhost:8080/",
#     selector="canvas",       # your renderer DOM element
#     library="three",
#     azimuth_deg=210,
#     elevation_deg=35,
#     distance=8
# )
#
# 3) Unknown library: fall back to a drag simulation on the canvas:
# pdf_path = html_3d_to_pdf(
#     output_pdf="snapshot.pdf",
#     html_path="viewer.html",
#     selector="canvas",
#     library=None,            # will simulate drag
#     azimuth_deg=60,          # ~rotate right
#     elevation_deg=15         # ~tilt up
# )


if __name__ == "__main__":

    paths = [r".\Bagging_for_LID\interactive_datasets\affine3to5.html"
r".\Bagging_for_LID\interactive_datasets\helix.html"
r".\Bagging_for_LID\interactive_datasets\m1_sphere3.html"
r".\Bagging_for_LID\interactive_datasets\m9affine.html"
r".\Bagging_for_LID\interactive_datasets\M10_Cubic.html"
r".\Bagging_for_LID\interactive_datasets\M12_Norm.html"
r".\Bagging_for_LID\interactive_datasets\mn_nonlinear_3d.html"
r".\Bagging_for_LID\interactive_datasets\moebius.html"
r".\Bagging_for_LID\interactive_datasets\nonlinear4_6_8.html"
r".\Bagging_for_LID\interactive_datasets\nonlinear4to6.html"
r".\Bagging_for_LID\interactive_datasets\roll.html"
r".\Bagging_for_LID\interactive_datasets\scurve.html"
r".\Bagging_for_LID\interactive_datasets\uniform_3d.html"]

    names = ["data_pdfs/affine3to5.pdf",
    "data_pdfs/helix.pdf",
    "data_pdfs/m1_sphere.pdf",
    "data_pdfs/m9affine.pdf",
    "data_pdfs/m10cubic.pdf",
    "data_pdfs/m12norm.pdf",
    "data_pdfs/mn_nonlinear_3d.pdf",
    "data_pdfs/moebius.pdf",
    "data_pdfs/nonlinear4_6_8.pdf",
    "data_pdfs/nonlinear4to6.pdf",
    "data_pdfs/roll.pdf",
    "data_pdfs/scurve.pdf",
    "data_pdfs/uniform_3d.pdf"]

    for i in range(len(paths)):
        path = paths[i]
        name = names[i]

        pdf_path = html_3d_to_pdf(
            output_pdf=name,
            html_path=path,
            selector=".js-plotly-plot",  # <- important
            library="plotly",
            distance=1.8  # try 1.6..2.2
        )