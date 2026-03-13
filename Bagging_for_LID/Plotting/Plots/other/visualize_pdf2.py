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

from io import BytesIO
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageChops


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


def _plotly_pack_colorbar_next_to_scene(page, selector: str, *, gap_px: int = 4):
    """
    After Plotly renders, measure the WebGL scene and SVG colorbar positions
    and translate the colorbar left so it sits gap_px from the scene.
    """
    page.evaluate(
        """
        async (args) => {
          const { rootSelector, gapPx } = args;

          const root = document.querySelector(rootSelector);
          const plot = root
            ? (root.classList && root.classList.contains('js-plotly-plot') ? root
               : root.querySelector('.js-plotly-plot, .plotly-graph-div'))
            : null;
          if (!plot) return {ok:false, reason:"no plot"};

          // Wait a couple frames so layout is stable
          await new Promise(r => requestAnimationFrame(r));
          await new Promise(r => requestAnimationFrame(r));

          const gl = plot.querySelector('.gl-container canvas, .gl-container');
          const svg = plot.querySelector('svg.main-svg');
          if (!gl || !svg) return {ok:false, reason:"missing gl or svg"};

          const glRect = gl.getBoundingClientRect();

          // Grab all plausible colorbar groups; choose the rightmost one
          const bars = svg.querySelectorAll('g.colorbar, g[class*="colorbar"], g.cb');
          if (!bars || bars.length === 0) return {ok:false, reason:"no colorbar"};

          let best = null;
          let bestRight = -Infinity;
          for (const g of bars) {
            const r = g.getBoundingClientRect();
            if (r.width < 2 || r.height < 10) continue; // ignore tiny junk
            if (r.right > bestRight) { bestRight = r.right; best = g; }
          }
          if (!best) return {ok:false, reason:"no usable colorbar"};

          const cbRect = best.getBoundingClientRect();

          // We want: cbRect.left == glRect.right + gap
          const desiredLeft = glRect.right + (gapPx ?? 4);
          const dx = desiredLeft - cbRect.left;   // negative => move left

          // Update the SVG transform translate(x,y) if present; otherwise prepend one
          const tf = best.getAttribute('transform') || '';
          const re = /translate\\(\\s*([-\\d.]+)[ ,]([-\\d.]+)\\s*\\)/;
          const m = tf.match(re);

          let newTf;
          if (m) {
            const x = parseFloat(m[1] || '0') + dx;
            const y = parseFloat(m[2] || '0');
            newTf = tf.replace(re, `translate(${x},${y})`);
          } else {
            newTf = `translate(${dx},0) ` + tf;
          }
          best.setAttribute('transform', newTf);

          return {ok:true, dx};
        }
        """,
        {"rootSelector": selector, "gapPx": int(gap_px)},
    )


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


def _screenshot_element_to_pdf(
    page,
    selector,
    pdf_path,
    *,
    trim=True,
    pack_colorbar=False,
    # NEW:
    cut_band: tuple[float, float] | None = None,  # (x1_pct, x2_pct)
    cut_gap_px: int = 0,
    cut_seam_trim_px: int = 1,
):
    handle = page.query_selector(selector)
    if not handle:
        raise RuntimeError(f"Could not find element: {selector}")
    box = handle.bounding_box()
    if not box:
        raise RuntimeError("Could not read element bounding box for screenshot.")

    png_bytes = page.screenshot(clip=box, type="png")

    # If you still want to keep the old packer, you can,
    # but usually you'll set pack_colorbar=False and use cut_band instead.
    if pack_colorbar:
        png_bytes = _pack_right_block_next_to_left(png_bytes, gap_px=4, threshold=10, pad=2)

    # NEW: deterministic cut+stitch
    if cut_band is not None:
        x1, x2 = cut_band
        png_bytes = cut_out_xband_and_stitch(
            png_bytes,
            x1_pct=float(x1),
            x2_pct=float(x2),
            gap_px=int(cut_gap_px),
            seam_trim_px=int(cut_seam_trim_px),
            flatten_white=True,
        )

    if trim:
        png_bytes = _trim_png_bytes(png_bytes, threshold=10, pad=2)

    bio = BytesIO(png_bytes)
    img_reader = ImageReader(bio)
    w, h = img_reader.getSize()

    c = rl_canvas.Canvas(pdf_path, pagesize=(w, h))
    c.drawImage(img_reader, 0, 0, width=w, height=h, mask='auto')
    c.showPage()
    c.save()

def _trim_png_bytes(png_bytes: bytes, *, threshold: int = 10, pad: int = 2) -> bytes:
    """
    Trim uniform border around an image (works even if background is opaque),
    keeping a small padding. First tries alpha-based trim; falls back to
    background-color-based trim.
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size

    # 1) Alpha-based trim (best when background is transparent)
    alpha = img.split()[-1]
    alpha_mask = alpha.point(lambda a: 255 if a > 0 else 0)
    bbox = alpha_mask.getbbox()
    if bbox and bbox != (0, 0, w, h):
        l, t, r, b = bbox
    else:
        # 2) Background-color-based trim (robust when alpha is uniform)
        corners = [
            img.getpixel((0, 0)),
            img.getpixel((w - 1, 0)),
            img.getpixel((0, h - 1)),
            img.getpixel((w - 1, h - 1)),
        ]
        # median RGB from corners
        bg_rgb = tuple(sorted([c[i] for c in corners])[len(corners) // 2] for i in range(3))

        rgb = img.convert("RGB")
        bg_img = Image.new("RGB", (w, h), bg_rgb)
        diff = ImageChops.difference(rgb, bg_img).convert("L")

        # threshold small noise
        mask = diff.point(lambda p: 255 if p > threshold else 0)
        bbox = mask.getbbox()
        if not bbox:
            return png_bytes
        l, t, r, b = bbox

    l = max(l - pad, 0)
    t = max(t - pad, 0)
    r = min(r + pad, w)
    b = min(b + pad, h)

    cropped = img.crop((l, t, r, b))
    out = BytesIO()
    cropped.save(out, format="PNG")
    return out.getvalue()


def _plotly_compact_styling(page, selector: str, *, colorbar_len: float = 0.55, colorbar_thickness: int = 12):
    page.evaluate(
        """
        async (args) => {
          const { rootSelector, colorbarLen, colorbarThickness } = args;

          const root = document.querySelector(rootSelector);
          const plot = root
            ? (root.classList && root.classList.contains('js-plotly-plot') ? root
               : root.querySelector('.js-plotly-plot, .plotly-graph-div'))
            : null;

          if (!plot || !window.Plotly) return false;

          // Hide UI chrome + transparent background (helps trimming)
          const styleId = '__html3d_to_pdf_plotly_style__';
          if (!document.getElementById(styleId)) {
            const st = document.createElement('style');
            st.id = styleId;
            st.textContent = `
              .modebar{display:none!important;}
              .plotly-notifier{display:none!important;}
              body{background:rgba(0,0,0,0)!important;}
            `;
            document.head.appendChild(st);
          }

          const fl = plot._fullLayout || {};
          const W = (fl.width || plot.clientWidth || 1280);

          // Put colorbar INSIDE the plot paper area, right-aligned by pixel math.
          // padPx controls how tight it is to the scene.
          const padPx = 2; // try 0..6 (smaller = closer)
          const cbPx  = Math.max(6, (colorbarThickness|0) + padPx);
          const xCb   = Math.max(0.0, Math.min(0.995, 1 - (cbPx / W)));

          // Scene should end just before the colorbar starts
          const gap = 0.003;                      // paper-units gap between scene and cb (tiny)
          const xSceneMax = Math.max(0.70, xCb - gap);

          // Which 3D scenes exist
          const ids = (fl._subplots && fl._subplots.gl3d) || ['scene'];

          const layoutUpdate = {
            'title.text': '',
            'margin.l': 0, 'margin.r': 120, 'margin.t': 0, 'margin.b': 0,
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
          };

          // Force each 3D scene to fill up to the colorbar
          for (const id of ids) {
            layoutUpdate[`${id}.bgcolor`] = 'rgba(0,0,0,0)';
            layoutUpdate[`${id}.domain.x`] = [0.0, xSceneMax];
            layoutUpdate[`${id}.domain.y`] = [0.0, 1.0];
          }

          // Apply layout changes first
          await Plotly.relayout(plot, layoutUpdate);

          // Common colorbar settings
          const cb = {
            'len': colorbarLen,
            'lenmode': 'fraction',
            'thickness': colorbarThickness,
            'x': xCb,
            'xanchor': 'left',
            'xpad': 0,
            'y': 0.5,
            'yanchor': 'middle',
            'title.text': '',
            'outlinewidth': 0,
          };

          // If layout uses coloraxis, update that too
          for (const k of Object.keys(fl)) {
            if (k.startsWith('coloraxis')) {
              const p = `${k}.colorbar`;
              await Plotly.relayout(plot, {
                [`${p}.len`]: cb.len,
                [`${p}.lenmode`]: cb.lenmode,
                [`${p}.thickness`]: cb.thickness,
                [`${p}.x`]: cb.x,
                [`${p}.xanchor`]: cb.xanchor,
                [`${p}.xpad`]: cb.xpad,
                [`${p}.y`]: cb.y,
                [`${p}.yanchor`]: cb.yanchor,
                [`${p}.title.text`]: cb['title.text'],
                [`${p}.outlinewidth`]: cb.outlinewidth,
              });
            }
          }

          // Traces may have colorbar at root, marker, or line
          const full = plot._fullData || plot.data || [];
          const idxRoot = [], idxMarker = [], idxLine = [];
          for (let i = 0; i < full.length; i++) {
            const tr = full[i] || {};
            if (tr.colorbar) idxRoot.push(i);
            if (tr.marker && tr.marker.colorbar) idxMarker.push(i);
            if (tr.line && tr.line.colorbar) idxLine.push(i);
          }

          const rootUpd = {
            'colorbar.len': cb.len,
            'colorbar.lenmode': cb.lenmode,
            'colorbar.thickness': cb.thickness,
            'colorbar.x': cb.x,
            'colorbar.xanchor': cb.xanchor,
            'colorbar.xpad': cb.xpad,
            'colorbar.y': cb.y,
            'colorbar.yanchor': cb.yanchor,
            'colorbar.title.text': cb['title.text'],
            'colorbar.outlinewidth': cb.outlinewidth,
          };

          const markerUpd = {};
          for (const [k,v] of Object.entries(rootUpd)) markerUpd[`marker.${k}`] = v;

          const lineUpd = {};
          for (const [k,v] of Object.entries(rootUpd)) lineUpd[`line.${k}`] = v;

          if (idxRoot.length)   await Plotly.restyle(plot, rootUpd, idxRoot);
          if (idxMarker.length) await Plotly.restyle(plot, markerUpd, idxMarker);
          if (idxLine.length)   await Plotly.restyle(plot, lineUpd, idxLine);

          await new Promise(r => requestAnimationFrame(r));
          await Plotly.relayout(plot, {}); // small nudge to reflow

          return true;
        }
        """,
        {
            "rootSelector": selector,
            "colorbarLen": float(colorbar_len),
            "colorbarThickness": int(colorbar_thickness),
        },
    )


def _pack_right_block_next_to_left(png_bytes: bytes, *, gap_px: int = 4, threshold: int = 10, pad: int = 2) -> bytes:
    """
    Heuristic packer for Plotly screenshots:
    - detects a 'right block' (colorbar + its labels) separated by a vertical whitespace gap
    - crops left content and right content and recomposes them adjacent with gap_px

    Safe: does not rescale/rotate; only translates the right block left.
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size

    # Estimate background color from corners (median RGB)
    corners = [img.getpixel((0, 0)), img.getpixel((w - 1, 0)), img.getpixel((0, h - 1)), img.getpixel((w - 1, h - 1))]
    bg_rgb = tuple(sorted([c[i] for c in corners])[len(corners) // 2] for i in range(3))

    rgb = img.convert("RGB")
    bg = Image.new("RGB", (w, h), bg_rgb)
    diff = ImageChops.difference(rgb, bg).convert("L")

    # Binary mask of "ink" pixels
    mask = diff.point(lambda p: 255 if p > threshold else 0)

    # Include alpha ink too (useful if background is transparent-ish)
    alpha = img.split()[-1]
    amask = alpha.point(lambda a: 255 if a > 0 else 0)
    mask = ImageChops.lighter(mask, amask)

    # Try numpy for fast column projection; fall back if unavailable
    try:
        import numpy as np
        proj = (np.array(mask) > 0).sum(axis=0)  # ink pixels per column
    except Exception:
        # fallback: slower pure-PIL column scan
        m = mask.load()
        proj = []
        for x in range(w):
            s = 0
            for y in range(h):
                if m[x, y] != 0:
                    s += 1
            proj.append(s)

    # Find rightmost "ink" (should include colorbar)
    min_ink = max(10, int(0.01 * h))
    gap_ink = max(2, int(0.001 * h))
    gap_run = max(6, int(0.006 * w))  # require a small run of near-empty columns

    r = w - 1
    while r > 0 and proj[r] <= min_ink:
        r -= 1
    if r <= 0:
        return png_bytes

    # Scan left to find a vertical whitespace gap separating main plot and colorbar
    run = 0
    split = None
    x = r
    while x > 0:
        if proj[x] <= gap_ink:
            run += 1
            if run >= gap_run:
                split = x + run  # first column after the gap-run (i.e., start of right block)
                break
        else:
            run = 0
        x -= 1

    if split is None or split <= 0 or split >= w:
        return png_bytes

    # Bounding boxes for left and right content (using mask)
    left_mask = mask.crop((0, 0, split, h))
    right_mask = mask.crop((split, 0, w, h))
    lb = left_mask.getbbox()
    rb = right_mask.getbbox()
    if not lb or not rb:
        return png_bytes

    # Convert to full-image coordinates + pad
    l1, t1, r1, b1 = lb
    l2, t2, r2, b2 = rb
    l1 = max(l1 - pad, 0);
    t1 = max(t1 - pad, 0);
    r1 = min(r1 + pad, split);
    b1 = min(b1 + pad, h)
    l2 = max(split + l2 - pad, 0);
    t2 = max(t2 - pad, 0);
    r2 = min(split + r2 + pad, w);
    b2 = min(b2 + pad, h)

    left = img.crop((l1, t1, r1, b1))
    right = img.crop((l2, t2, r2, b2))

    lw, lh = left.size
    rw, rh = right.size

    # Keep original vertical alignment between blocks
    left_center = (t1 + b1) / 2
    right_center = (t2 + b2) / 2
    delta = right_center - left_center

    new_h = max(lh, rh)
    new_w = lw + gap_px + rw
    out = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))

    left_y = (new_h - lh) // 2
    left_center_new = left_y + lh / 2
    right_center_new = left_center_new + delta
    right_y = int(round(right_center_new - rh / 2))
    right_y = max(0, min(new_h - rh, right_y))

    out.paste(left, (0, left_y))
    out.paste(right, (lw + gap_px, right_y))

    bio = BytesIO()
    out.save(bio, format="PNG")
    return bio.getvalue()


def _plotly_place_colorbar_next_to_gl(
        page,
        selector: str,
        *,
        gap_px: int = 6,  # how close the bar is to the plot
        colorbar_len: float = 0.45,  # shorter bar
        thickness: int = 12,
        right_margin_px: int = 80,  # prevents tick labels from clipping
):
    page.evaluate(
        """
        async (args) => {
          const { rootSelector, gapPx, cbLen, cbThickness, rightMarginPx } = args;

          const root = document.querySelector(rootSelector);
          const gd = root
            ? (root.classList && root.classList.contains('js-plotly-plot') ? root
               : root.querySelector('.js-plotly-plot, .plotly-graph-div'))
            : null;

          if (!gd || !window.Plotly) return false;

          // Wait for layout to stabilize (important after camera changes)
          await new Promise(r => requestAnimationFrame(r));
          await new Promise(r => requestAnimationFrame(r));

          const gl = gd.querySelector('.gl-container canvas, .gl-container');
          if (!gl) return false;

          const gdRect = gd.getBoundingClientRect();
          const glRect = gl.getBoundingClientRect();

          // Desired left edge of the colorbar in page pixels
          const desiredLeftPx = glRect.right + (gapPx ?? 6);

          // Convert to Plotly paper coordinate (0..1 relative to the graph div)
          let xPaper = (desiredLeftPx - gdRect.left) / gdRect.width;
          xPaper = Math.max(0, Math.min(0.98, xPaper));

          // Make layout compact (title/margins) but leave right margin for labels
          await Plotly.relayout(gd, {
            'title.text': '',
            'margin.l': 0,
            'margin.t': 0,
            'margin.b': 0,
            'margin.r': rightMarginPx ?? 80
          });

          // Apply to any layout coloraxis colorbars
          const fl = gd._fullLayout || {};
          for (const k of Object.keys(fl)) {
            if (k === 'coloraxis' || /^coloraxis\\d+$/.test(k)) {
              const p = `${k}.colorbar`;
              const upd = {};
              upd[`${p}.x`] = xPaper;
              upd[`${p}.xanchor`] = 'left';
              upd[`${p}.xpad`] = 0;
              upd[`${p}.y`] = 0.5;
              upd[`${p}.yanchor`] = 'middle';
              upd[`${p}.len`] = cbLen;
              upd[`${p}.lenmode`] = 'fraction';
              upd[`${p}.thickness`] = cbThickness;
              upd[`${p}.title.text`] = '';
              upd[`${p}.outlinewidth`] = 0;
              await Plotly.relayout(gd, upd);
            }
          }

          // Apply to traces (marker colorbar, root colorbar, line colorbar)
          const full = gd._fullData || gd.data || [];
          const idxMarker = [];
          const idxRoot = [];
          const idxLine = [];

          for (let i = 0; i < full.length; i++) {
            const tr = full[i] || {};

            // Most 3D scatters: marker.color exists; showscale defaults true when colorscale used
            const hasMarkerColor = tr.marker && (tr.marker.color != null);
            if (hasMarkerColor && tr.marker.showscale !== false) idxMarker.push(i);

            // Surfaces/meshes sometimes use root-level showscale/colorbar
            if (tr.showscale && tr.showscale !== false) idxRoot.push(i);

            // Some traces use line colorbars
            if (tr.line && tr.line.color != null && tr.line.showscale !== false) idxLine.push(i);
          }

          const markerUpd = {
            'marker.colorbar.x': xPaper,
            'marker.colorbar.xanchor': 'left',
            'marker.colorbar.xpad': 0,
            'marker.colorbar.y': 0.5,
            'marker.colorbar.yanchor': 'middle',
            'marker.colorbar.len': cbLen,
            'marker.colorbar.lenmode': 'fraction',
            'marker.colorbar.thickness': cbThickness,
            'marker.colorbar.title.text': '',
            'marker.colorbar.outlinewidth': 0,
          };

          const rootUpd = {
            'colorbar.x': xPaper,
            'colorbar.xanchor': 'left',
            'colorbar.xpad': 0,
            'colorbar.y': 0.5,
            'colorbar.yanchor': 'middle',
            'colorbar.len': cbLen,
            'colorbar.lenmode': 'fraction',
            'colorbar.thickness': cbThickness,
            'colorbar.title.text': '',
            'colorbar.outlinewidth': 0,
          };

          const lineUpd = {
            'line.colorbar.x': xPaper,
            'line.colorbar.xanchor': 'left',
            'line.colorbar.xpad': 0,
            'line.colorbar.y': 0.5,
            'line.colorbar.yanchor': 'middle',
            'line.colorbar.len': cbLen,
            'line.colorbar.lenmode': 'fraction',
            'line.colorbar.thickness': cbThickness,
            'line.colorbar.title.text': '',
            'line.colorbar.outlinewidth': 0,
          };

          if (idxMarker.length) await Plotly.restyle(gd, markerUpd, idxMarker);
          if (idxRoot.length)   await Plotly.restyle(gd, rootUpd, idxRoot);
          if (idxLine.length)   await Plotly.restyle(gd, lineUpd, idxLine);

          await new Promise(r => requestAnimationFrame(r));
          return true;
        }
        """,
        {
            "rootSelector": selector,
            "gapPx": int(gap_px),
            "cbLen": float(colorbar_len),
            "cbThickness": int(thickness),
            "rightMarginPx": int(right_margin_px),
        },
    )

def cut_out_xband_and_stitch(
    png_bytes: bytes,
    *,
    x1_pct: float,
    x2_pct: float,
    gap_px: int = 0,
    seam_trim_px: int = 1,
    flatten_white: bool = True,
) -> bytes:
    """
    Cut out the vertical band between [x1_pct, x2_pct] of image width and stitch the sides.
    - x1_pct, x2_pct in [0, 100]
    - gap_px: optional pixels between the stitched halves (usually 0..4)
    - seam_trim_px: trims 1px off both touching edges to avoid a seam line
    - flatten_white: paste onto white background to avoid alpha->black seams in PDF
    """
    if not (0 <= x1_pct < x2_pct <= 100):
        raise ValueError("x1_pct/x2_pct must satisfy 0 <= x1_pct < x2_pct <= 100")

    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    w, h = img.size

    x1 = int(round((x1_pct / 100.0) * w))
    x2 = int(round((x2_pct / 100.0) * w))
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    if x2 <= x1:
        return png_bytes

    left = img.crop((0, 0, x1, h))
    right = img.crop((x2, 0, w, h))

    # Optional seam trimming (prevents double line at join)
    if seam_trim_px and left.size[0] > seam_trim_px + 2 and right.size[0] > seam_trim_px + 2:
        left = left.crop((0, 0, left.size[0] - seam_trim_px, h))
        right = right.crop((seam_trim_px, 0, right.size[0], h))

    out_w = left.size[0] + gap_px + right.size[0]
    out_h = h

    if flatten_white:
        out = Image.new("RGB", (out_w, out_h), (255, 255, 255))
        # paste left/right with alpha
        out.paste(left, (0, 0), left.split()[-1])
        out.paste(right, (left.size[0] + gap_px, 0), right.split()[-1])
        out_rgba = out.convert("RGBA")
        bio = BytesIO()
        out_rgba.convert("RGB").save(bio, format="PNG")
        return bio.getvalue()
    else:
        out = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))
        out.paste(left, (0, 0))
        out.paste(right, (left.size[0] + gap_px, 0))
        bio = BytesIO()
        out.save(bio, format="PNG")
        return bio.getvalue()

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
        azimuth_deg: float = 45.0,  # rotate around vertical axis (0 = +X, 90 = +Y)
        elevation_deg: float = math.degrees(math.asin(0.55)),  # tilt up from horizon (0 = horizon, 90 = top-down)
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

        if (library or "").lower() == "plotly":
            _plotly_place_colorbar_next_to_gl(
                page,
                target_selector,
                gap_px=4,  # tighter
                colorbar_len=0.45,  # shorter
                thickness=12,
                right_margin_px=80  # avoids clipping tick labels
            )
            page.wait_for_timeout(80)

        _screenshot_element_to_pdf(
            page, target_selector, output_pdf,
            trim=True,
            pack_colorbar=False,
            cut_band=(72.0, 88.0),
            cut_gap_px=2,
            cut_seam_trim_px=1,
        )

    return output_pdf


if __name__ == "__main__":

    paths = [r".\interactive_datasets\affine3to5.html",
             r".\interactive_datasets\helix.html",
             r".\interactive_datasets\m1_sphere3.html",
             r".\interactive_datasets\m9affine.html",
             r".\interactive_datasets\M10_Cubic.html",
             r".\interactive_datasets\M12_Norm.html",
             r".\interactive_datasets\mn_nonlinear_3d.html",
             r".\interactive_datasets\moebius.html",
             r".\interactive_datasets\nonlinear4_6_8.html",
             r".\interactive_datasets\nonlinear4to6.html",
             r".\interactive_datasets\roll.html",
             r".\interactive_datasets\scurve.html",
             r".\interactive_datasets\uniform_3d.html"]

    names = [r".\Bagging_for_LID\data_pdfs\affine3to5.pdf",
             r".\Bagging_for_LID\data_pdfs\helix.pdf",
             r".\Bagging_for_LID\data_pdfs\m1_sphere.pdf",
             r".\Bagging_for_LID\data_pdfs\m9affine.pdf",
             r".\Bagging_for_LID\data_pdfs\m10cubic.pdf",
             r".\Bagging_for_LID\data_pdfs\m12norm.pdf",
             r".\Bagging_for_LID\data_pdfs\mn_nonlinear_3d.pdf",
             r".\Bagging_for_LID\data_pdfs\moebius.pdf",
             r".\Bagging_for_LID\data_pdfs\nonlinear4_6_8.pdf",
             r".\Bagging_for_LID\data_pdfs\nonlinear4to6.pdf",
             r".\Bagging_for_LID\data_pdfs\roll.pdf",
             r".\Bagging_for_LID\data_pdfs\scurve.pdf",
             r".\Bagging_for_LID\data_pdfs\uniform_3d.pdf"]

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