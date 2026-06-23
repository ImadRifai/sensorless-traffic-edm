"""Tiny self-contained Markdown -> styled HTML -> PDF (via Playwright/Chromium).

Usage: python md_to_pdf.py input.md output.pdf "Document title"
No external Markdown library needed — handles the subset used in our scripts.
"""
from __future__ import annotations

import html
import re
import sys
from pathlib import Path


def inline(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r'<a href="\2">\1</a>', text)
    return text


def md_to_body(md: str) -> str:
    out, in_ul, in_quote = [], False, False

    def close_ul():
        nonlocal in_ul
        if in_ul:
            out.append("</ul>")
            in_ul = False

    def close_quote():
        nonlocal in_quote
        if in_quote:
            out.append("</blockquote>")
            in_quote = False

    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            close_ul()
            close_quote()
            continue
        if re.match(r"^---+$", line):
            close_ul(); close_quote()
            out.append("<hr/>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            close_ul(); close_quote()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            continue
        if line.startswith(">"):
            close_ul()
            if not in_quote:
                out.append("<blockquote>"); in_quote = True
            content = line.lstrip(">").strip()
            out.append(f"<p>{inline(content)}</p>" if content else "")
            continue
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            close_quote()
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{inline(m.group(1))}</li>")
            continue
        close_ul(); close_quote()
        out.append(f"<p>{inline(line)}</p>")
    close_ul(); close_quote()
    return "\n".join(out)


CSS = """
@page { size: A4; margin: 16mm 15mm; }
* { box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  color: #16323F; line-height: 1.5; font-size: 12.5px; max-width: 720px; margin: 0 auto; }
h1 { font-size: 22px; border-bottom: 3px solid #0F6E82; padding-bottom: 8px; margin: 0 0 14px; }
h2 { font-size: 15px; color: #0F6E82; margin: 22px 0 8px; }
h3 { font-size: 13.5px; background: #F2F6F8; border-left: 3px solid #0F6E82;
  padding: 6px 10px; border-radius: 5px; margin: 18px 0 8px; page-break-after: avoid; }
hr { border: 0; border-top: 1px solid #E2E9EC; margin: 14px 0; }
p { margin: 6px 0; }
ul { margin: 6px 0 6px 4px; padding-left: 18px; }
li { margin: 3px 0; }
blockquote { margin: 8px 0; padding: 9px 13px; background: #FBFAF4;
  border-left: 3px solid #E8B210; border-radius: 5px; page-break-inside: avoid; }
blockquote p { margin: 3px 0; }
code { background: #EEF3F5; padding: 1px 5px; border-radius: 4px;
  font-family: 'SF Mono', Menlo, Consolas, monospace; font-size: 0.9em; }
a { color: #0F6E82; text-decoration: none; }
strong { color: #0F2630; }
"""


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else src.stem
    body = md_to_body(src.read_text(encoding="utf-8"))
    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
           f"<body>{body}</body></html>")
    tmp = dst.with_suffix(".html")
    tmp.write_text(doc, encoding="utf-8")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--no-sandbox"])
        pg = b.new_page()
        pg.goto(tmp.resolve().as_uri(), wait_until="networkidle")
        pg.pdf(path=str(dst), format="A4", print_background=True,
               margin={"top": "16mm", "bottom": "16mm", "left": "15mm", "right": "15mm"})
        b.close()
    tmp.unlink(missing_ok=True)
    print(f"PDF written: {dst} ({dst.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
