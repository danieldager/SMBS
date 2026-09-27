"""Write pipeline.svg (light) and pipeline-dark.svg.   python docs/figures/src/make_pipeline.py"""
# Sizes are in viewBox units; the README shows the SVG at full column width (about 920px, 77%).
from pathlib import Path
OUT = Path(__file__).parent.parent
# (title, sub-label lines, command that runs this step or None)
STAGES = [("Audio", ["speech recordings,", "any length"], "smbs scan"),
          ("Encoder", ["HuBERT-500, mHuBERT", "or SpidR"], "smbs encode"),
          ("Unit shards", ["WebDataset .tar,", "streamed to GPUs"], None),
          ("LSTM or GPT-2", ["next-unit language", "model, multi-GPU"], "smbs train"),
          ("sWuggy score", ["real word vs", "pseudo-word"], "smbs evaluate")]
THEMES = {  # tuned for GitHub's #ffffff and #0d1117 page grounds
    "pipeline.svg": dict(fill="#eef1f5", stroke="#57606a", title="#1f2328", sub="#424a53", num="#57606a", arrow="#57606a",
                         chip=("#e3eefb", "#a9c8ef", "#1c5cab")),
    "pipeline-dark.svg": dict(fill="#21262d", stroke="#8b949e", title="#f0f6fc", sub="#c9d1d9", num="#9198a1", arrow="#8b949e",
                              chip=("#132a45", "#35649c", "#a9cdf5"))}
W, H, BW, BH, GAP, X0, Y0 = 1200, 180, 206, 60, 40.5, 4, 8
CW, CH = 150, 30
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
LABEL = ("Pipeline in five steps: Audio (smbs scan) is encoded into discrete units by HuBERT-500, mHuBERT or SpidR "
         "(smbs encode); the units are written as WebDataset shards and streamed to the GPUs; an LSTM or GPT-2 "
         "learns to predict the next unit (smbs train); the model is scored on sWuggy, real word versus "
         "pseudo-word (smbs evaluate).")
for name, c in THEMES.items():
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{LABEL}">',
         f'<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
         f'<path d="M0,0 L10,5 L0,10 z" fill="{c["arrow"]}"/></marker></defs>']
    for i, (t, subs, cmd) in enumerate(STAGES):
        x = X0 + i * (BW + GAP)
        cx = x + BW / 2
        dash = ' stroke-dasharray="5 4"' if cmd is None else ""
        o.append(f'<rect x="{x}" y="{Y0}" width="{BW}" height="{BH}" rx="10" fill="{c["fill"]}" stroke="{c["stroke"]}" stroke-width="1.5"{dash}/>')
        o.append(f'<text x="{cx}" y="{Y0 + BH / 2 + 7}" text-anchor="middle" font-family="{FONT}" font-size="21" '
                 f'font-weight="600" fill="{c["title"]}">{t}</text>')
        for j, s in enumerate(subs):
            o.append(f'<text x="{cx}" y="{Y0 + BH + 28 + j * 23}" text-anchor="middle" font-family="{FONT}" '
                     f'font-size="18" fill="{c["sub"]}">{s}</text>')
        if cmd:
            fl, st, tx = c["chip"]
            cy = Y0 + BH + 72
            o.append(f'<rect x="{cx - CW / 2}" y="{cy}" width="{CW}" height="{CH}" rx="{CH / 2}" fill="{fl}" stroke="{st}" stroke-width="1"/>')
            o.append(f'<text x="{cx}" y="{cy + CH / 2 + 5}" text-anchor="middle" font-family="{MONO}" font-size="15" '
                     f'fill="{tx}">{cmd}</text>')
        if i < len(STAGES) - 1:
            o.append(f'<line x1="{x + BW + 4}" y1="{Y0 + BH / 2}" x2="{x + BW + GAP - 4}" y2="{Y0 + BH / 2}" '
                     f'stroke="{c["arrow"]}" stroke-width="2" marker-end="url(#ah)"/>')
    o.append("</svg>")
    (OUT / name).write_text("\n".join(o) + "\n")
