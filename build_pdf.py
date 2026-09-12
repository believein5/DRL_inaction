"""Convert conclusion_notes.md to conclusion_notes.pdf using reportlab.

Basic markdown handling: # ## ### headers, code blocks ```lang / ``` , inline
`code`, **bold**, *italic*, bullet lists starting with - or *, numbered lists.
Not a full commonmark parser — just enough for our note.
"""
from __future__ import annotations
import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted,
    ListFlowable, ListItem, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT

SRC = Path("conclusion_notes.md")
DST = Path("conclusion_notes.pdf")


# Styles
styles = getSampleStyleSheet()
body_style = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontName="Helvetica",
    fontSize=10.5, leading=15, alignment=TA_LEFT, spaceAfter=6,
)
h1_style = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontName="Helvetica-Bold",
    fontSize=22, leading=26, spaceBefore=14, spaceAfter=10,
    textColor=HexColor("#1a1a1a"),
)
h2_style = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontName="Helvetica-Bold",
    fontSize=15, leading=19, spaceBefore=16, spaceAfter=6,
    textColor=HexColor("#2a2a2a"),
)
h3_style = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontName="Helvetica-Bold",
    fontSize=12, leading=16, spaceBefore=10, spaceAfter=4,
    textColor=HexColor("#333333"),
)
code_style = ParagraphStyle(
    "Code", parent=styles["Code"], fontName="Courier",
    fontSize=8.8, leading=11, leftIndent=12, rightIndent=8,
    backColor=HexColor("#f2f2f2"), borderColor=HexColor("#dddddd"),
    borderWidth=0.5, borderPadding=6, spaceBefore=6, spaceAfter=8,
)
italic_style = ParagraphStyle(
    "Italic", parent=body_style, fontName="Helvetica-Oblique",
)
list_body_style = ParagraphStyle(
    "ListBody", parent=body_style, leftIndent=0, spaceAfter=2,
)


def esc(text: str) -> str:
    """Escape HTML/XML special chars for reportlab Paragraph, then re-enable
    inline markdown (**bold**, *italic*, `code`)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # inline code: `foo`
    text = re.sub(r"`([^`]+)`",
                  r'<font face="Courier" backColor="#f2f2f2" color="#7a1a1a">\1</font>',
                  text)
    # bold: **foo**
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    # italic: *foo*
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    return text


def parse_markdown(text: str):
    """Yield reportlab flowables."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            j = i + 1
            code_lines = []
            while j < len(lines) and not lines[j].startswith("```"):
                code_lines.append(lines[j])
                j += 1
            yield Preformatted("\n".join(code_lines), code_style)
            i = j + 1
            continue

        # Horizontal rule
        if line.strip() == "---":
            yield Spacer(1, 4)
            yield HRFlowable(width="100%", thickness=0.5,
                             color=HexColor("#888888"),
                             spaceBefore=4, spaceAfter=8)
            i += 1
            continue

        # Headings
        if line.startswith("### "):
            yield Paragraph(esc(line[4:]), h3_style)
            i += 1
            continue
        if line.startswith("## "):
            yield Paragraph(esc(line[3:]), h2_style)
            i += 1
            continue
        if line.startswith("# "):
            yield Paragraph(esc(line[2:]), h1_style)
            i += 1
            continue

        # Numbered list
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m:
            items = []
            while i < len(lines):
                m2 = re.match(r"^(\d+)\.\s+(.*)$", lines[i])
                if not m2:
                    break
                items.append(ListItem(
                    Paragraph(esc(m2.group(2)), list_body_style),
                    leftIndent=18, bulletColor=HexColor("#444444"),
                ))
                i += 1
            yield ListFlowable(items, bulletType="1", start="1",
                               leftIndent=18, bulletFontSize=10.5,
                               bulletFontName="Helvetica")
            continue

        # Bullet list
        if line.startswith("- ") or line.startswith("* "):
            items = []
            while i < len(lines) and (lines[i].startswith("- ") or lines[i].startswith("* ")):
                items.append(ListItem(
                    Paragraph(esc(lines[i][2:]), list_body_style),
                    leftIndent=18, bulletColor=HexColor("#444444"),
                ))
                i += 1
            yield ListFlowable(items, bulletType="bullet",
                               leftIndent=18, bulletFontSize=10.5,
                               bulletFontName="Helvetica")
            continue

        # Italic-only paragraph (e.g., subtitle line starting with *)
        if line.startswith("*") and line.endswith("*") and line.count("*") == 2:
            yield Paragraph(esc(line[1:-1]), italic_style)
            i += 1
            continue

        # Blank line
        if line.strip() == "":
            yield Spacer(1, 4)
            i += 1
            continue

        # Regular paragraph — accumulate contiguous non-empty non-markup lines
        para_lines = [line]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if nxt.strip() == "":
                break
            if nxt.startswith(("#", "```", "- ", "* ", "---")):
                break
            if re.match(r"^\d+\.\s+", nxt):
                break
            para_lines.append(nxt)
            j += 1
        yield Paragraph(esc(" ".join(para_lines)), body_style)
        i = j


def main():
    text = SRC.read_text()
    doc = SimpleDocTemplate(
        str(DST), pagesize=A4,
        leftMargin=2.2*cm, rightMargin=2.2*cm,
        topMargin=2.0*cm, bottomMargin=2.0*cm,
        title="Deep RL in Action — Learning Journal",
        author="believein5",
    )
    story = list(parse_markdown(text))
    doc.build(story)
    print(f"wrote {DST} ({DST.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
