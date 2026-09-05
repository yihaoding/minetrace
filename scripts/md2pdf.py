#!/usr/bin/env python
"""Minimal Markdown -> PDF renderer (CJK-safe, no pandoc/LaTeX needed).

Supports the subset used in reports/: headings, paragraphs, bullet/ordered
lists, blockquotes, fenced code blocks, pipe tables, horizontal rules, and
inline **bold** / `code` / [text](link).

Usage:  python scripts/md2pdf.py <input.md> <output.pdf> [--title "..."]
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "pylibs"))

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, HRFlowable, KeepTogether, ListFlowable, ListItem,
    PageTemplate, Paragraph, Spacer, Table, TableStyle, XPreformatted,
)

# ── fonts ────────────────────────────────────────────────────────────────────
# No single system font covers both CJK and Latin/maths here, so text is routed
# per character: DejaVu for ASCII + arrows/maths (it has a real bold), Droid
# Sans Fallback for CJK, DejaVu Sans Mono for code.
DROID = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"
DEJA  = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
DEJAB = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"
MONOF = "/usr/share/fonts/dejavu-sans-mono-fonts/DejaVuSansMono.ttf"

CJK, LAT, MONO = "CJK", "Lat", "Mono"
pdfmetrics.registerFont(TTFont(CJK, DROID))
pdfmetrics.registerFont(TTFont(CJK + "-Bold", DROID))      # no CJK bold on this box
pdfmetrics.registerFont(TTFont(LAT, DEJA))
pdfmetrics.registerFont(TTFont(LAT + "-Bold", DEJAB))
pdfmetrics.registerFont(TTFont(MONO, MONOF))
pdfmetrics.registerFont(TTFont(MONO + "-Bold", MONOF))
for fam in (CJK, LAT, MONO):
    pdfmetrics.registerFontFamily(fam, normal=fam, bold=fam + "-Bold",
                                  italic=fam, boldItalic=fam + "-Bold")


def _coverage(path: str) -> set[int]:
    from fontTools.ttLib import TTFont as _TTF
    f = _TTF(path, fontNumber=0)
    return {c for tb in f["cmap"].tables for c in tb.cmap}


_COV_LAT = _coverage(DEJA)
_COV_CJK = _coverage(DROID)


def _face(ch: str, latin: str) -> str:
    o = ord(ch)
    if o < 0x2E80 and o in _COV_LAT:      # ASCII, arrows, maths, dashes
        return latin
    if o in _COV_CJK:                     # CJK ideographs and punctuation
        return CJK
    return latin if o in _COV_LAT else CJK


def fontify(text: str, base: str, latin: str = LAT) -> str:
    """Wrap runs whose covering font differs from the paragraph's base font.

    Existing markup tags are passed through untouched.
    """
    out = []
    for part in re.split(r"(<[^>]*>)", text):
        if not part:
            continue
        if part.startswith("<"):
            out.append(part)
            continue
        run, run_face = "", None
        for ch in part:
            f = _face(ch, latin)
            if f != run_face:
                if run:
                    out.append(run if run_face == base else
                               '<font face="%s">%s</font>' % (run_face, run))
                run, run_face = ch, f
            else:
                run += ch
        if run:
            out.append(run if run_face == base else
                       '<font face="%s">%s</font>' % (run_face, run))
    return "".join(out)


INK    = colors.HexColor("#1f2937")
ACCENT = colors.HexColor("#14507d")
MUTED  = colors.HexColor("#6b7280")
RULE   = colors.HexColor("#d7dee6")
CODEBG = colors.HexColor("#f2f5f8")
HEADBG = colors.HexColor("#eaf0f6")

S = {
    "title": ParagraphStyle("title", fontName=CJK, fontSize=19, leading=26,
                            textColor=ACCENT, spaceAfter=2),
    "h2":    ParagraphStyle("h2", fontName=CJK, fontSize=13.5, leading=19,
                            textColor=ACCENT, spaceBefore=13, spaceAfter=5),
    "h3":    ParagraphStyle("h3", fontName=CJK, fontSize=11.5, leading=17,
                            textColor=INK, spaceBefore=9, spaceAfter=3),
    "body":  ParagraphStyle("body", fontName=CJK, fontSize=9.8, leading=16.2,
                            textColor=INK, alignment=TA_LEFT, spaceAfter=6),
    "quote": ParagraphStyle("quote", fontName=CJK, fontSize=9.3, leading=15.4,
                            textColor=MUTED, leftIndent=9, borderPadding=(5, 5, 5, 9),
                            spaceBefore=3, spaceAfter=7),
    "code":  ParagraphStyle("code", fontName=MONO, fontSize=8.6, leading=14.5,
                            textColor=INK, backColor=CODEBG, borderPadding=7,
                            leftIndent=2, spaceBefore=3, spaceAfter=8),
    "cell":  ParagraphStyle("cell", fontName=CJK, fontSize=8.8, leading=13.4,
                            textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName=CJK, fontSize=8.8, leading=13.4,
                            textColor=ACCENT),
    "foot":  ParagraphStyle("foot", fontName=CJK, fontSize=7.5, leading=10,
                            textColor=MUTED),
}


# ── inline markup ────────────────────────────────────────────────────────────
def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)                  # links -> text
    text = re.sub(r"`([^`]+)`",
                  r'<font backColor="#eaeef2" color="#0f3a5f"> \1 </font>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r'<font color="#0b3352"><b>\1</b></font>', text)
    return fontify(text, base=CJK)


def para(text: str, style: str = "body") -> Paragraph:
    return Paragraph(inline(text), S[style])


# ── block parsing ────────────────────────────────────────────────────────────
def build_table(rows: list[list[str]], width: float) -> Table:
    header, body = rows[0], rows[1:]
    ncol = len(header)
    weights = []
    for c in range(ncol):
        longest = max(len(r[c]) for r in rows)
        weights.append(max(longest, 6) ** 0.72)
    total = sum(weights)
    col_w = [width * w / total for w in weights]

    data = [[Paragraph(inline(c), S["cellh"]) for c in header]]
    data += [[Paragraph(inline(c), S["cell"]) for c in r] for r in body]

    t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADBG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("INNERGRID", (0, 1), (-1, -1), 0.35, RULE),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def parse(md: str, width: float) -> list:
    lines = md.split("\n")
    flow: list = []
    i, n = 0, len(lines)
    bullets: list[str] = []

    def flush_bullets():
        nonlocal bullets
        if not bullets:
            return
        flow.append(ListFlowable(
            [ListItem(para(b), leftIndent=15) for b in bullets],
            bulletType="bullet", start="\u2022", bulletFontName=LAT,
            bulletFontSize=8, bulletColor=ACCENT, leftIndent=15, spaceAfter=6,
        ))
        bullets = []

    while i < n:
        ln = lines[i]
        stripped = ln.strip()

        if not stripped:
            flush_bullets(); i += 1; continue

        if stripped.startswith("```"):                                   # fenced code
            flush_bullets()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            body = "\n".join(buf).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            flow.append(XPreformatted(fontify(body, base=MONO), S["code"]))
            continue

        if re.match(r"^(---+|\*\*\*+)$", stripped):                      # hr
            flush_bullets()
            flow.append(Spacer(1, 4))
            flow.append(HRFlowable(width="100%", color=RULE, thickness=0.7))
            flow.append(Spacer(1, 6))
            i += 1; continue

        if stripped.startswith("|") and i + 1 < n and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            flush_bullets()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "|".join(cells)):
                    rows.append(cells)
                i += 1
            flow.append(build_table(rows, width))
            flow.append(Spacer(1, 8))
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)                     # heading
        if m:
            flush_bullets()
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                flow.append(para(text, "title"))
                flow.append(HRFlowable(width="100%", color=ACCENT, thickness=1.1,
                                       spaceBefore=4, spaceAfter=10))
            else:
                flow.append(KeepTogether([para(text, "h2" if level == 2 else "h3")]))
            i += 1; continue

        if stripped.startswith(">"):                                     # blockquote
            flush_bullets()
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip()); i += 1
            q = Paragraph(inline(" ".join(buf)), S["quote"])
            flow.append(Table([[q]], colWidths=[width], hAlign="LEFT", style=TableStyle([
                ("LINEBEFORE", (0, 0), (0, -1), 2.2, ACCENT),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7f9fb")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])))
            flow.append(Spacer(1, 7))
            continue

        m = re.match(r"^[-*]\s+(.*)$", stripped)                         # bullet
        if m:
            bullets.append(m.group(1)); i += 1; continue

        m = re.match(r"^(\d+)\.\s+(.*)$", stripped)                      # ordered item
        if m:
            flush_bullets()
            flow.append(Paragraph(
                inline(m.group(2)),
                ParagraphStyle("num", parent=S["body"], leftIndent=16,
                               bulletIndent=2, bulletFontName=LAT,
                               bulletFontSize=9.5, bulletColor=ACCENT, spaceAfter=4),
                bulletText=m.group(1) + "."))
            i += 1; continue

        flush_bullets()                                                  # paragraph
        buf = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(
                r"^(#{1,4}\s|[-*]\s|\d+\.\s|>|\||```|---+$)", lines[i].strip()):
            buf.append(lines[i].strip()); i += 1
        flow.append(para(" ".join(buf)))

    flush_bullets()
    return flow


# ── document ─────────────────────────────────────────────────────────────────
def render(src: str, dst: str, footer: str = "") -> None:
    with open(src, encoding="utf-8") as fh:
        md = fh.read()

    page_w, page_h = A4
    lm = rm = 20 * mm
    tm, bm = 18 * mm, 20 * mm
    frame_w = page_w - lm - rm

    doc = BaseDocTemplate(dst, pagesize=A4,
                          leftMargin=lm, rightMargin=rm, topMargin=tm, bottomMargin=bm,
                          title=os.path.basename(src), author="gad_reasoning")

    def draw_mixed(canvas, x, y, text, size=7.5):
        """drawString for text that mixes CJK and Latin (no font covers both)."""
        for ch in text:
            face = _face(ch, LAT)
            canvas.setFont(face, size)
            canvas.drawString(x, y, ch)
            x += pdfmetrics.stringWidth(ch, face, size)

    def decorate(canvas, _doc):
        canvas.saveState()
        canvas.setFillColor(MUTED)
        if footer:
            draw_mixed(canvas, lm, bm - 11, footer)
        canvas.setFont(LAT, 7.5)
        canvas.drawRightString(page_w - rm, bm - 11, str(canvas.getPageNumber()))
        canvas.restoreState()

    doc.addPageTemplates([PageTemplate(
        id="main",
        frames=[Frame(lm, bm, frame_w, page_h - tm - bm, id="body",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)],
        onPage=decorate,
    )])
    doc.build(parse(md, frame_w))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    foot = ""
    if "--footer" in sys.argv:
        foot = sys.argv[sys.argv.index("--footer") + 1]
    render(sys.argv[1], sys.argv[2], foot)
    print(f"wrote {sys.argv[2]}")
