"""Build the bilingual engineering papers from the checked-in Markdown sources."""
import argparse
from pathlib import Path
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/paper'


def inline(text):
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#2160a0">\1</link>', text)
    return text


def build(lang, font, bold):
    source = PAPER / ('paper.' + lang + '.md')
    output = PAPER / ('H3-TP-Experiments.' + lang + '.pdf')
    zh = lang == 'zh-CN'
    body = ParagraphStyle('body', fontName=font, fontSize=10.4 if zh else 10.6,
                          leading=16.6 if zh else 15.2, spaceAfter=10, wordWrap='CJK' if zh else None)
    styles = {
        'body': body,
        'title': ParagraphStyle('title', parent=body, fontName=bold, fontSize=25, leading=32, textColor=colors.HexColor('#19334b'), spaceAfter=20),
        'h2': ParagraphStyle('h2', parent=body, fontName=bold, fontSize=16, leading=22, spaceBefore=8, spaceAfter=13, keepWithNext=True),
        'h3': ParagraphStyle('h3', parent=body, fontName=bold, fontSize=12.4, leading=17, spaceBefore=10, spaceAfter=9, keepWithNext=True),
        'cell': ParagraphStyle('cell', parent=body, fontSize=9.1 if not zh else 9.4, leading=12.7 if not zh else 14.2, spaceAfter=0),
        'headcell': ParagraphStyle('headcell', parent=body, fontName=bold, fontSize=9.2, leading=13, textColor=colors.white, spaceAfter=0),
        'cap': ParagraphStyle('cap', parent=body, fontSize=9, leading=13, textColor=colors.HexColor('#52616d')),
    }
    width = A4[0] - 104
    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=52, rightMargin=52, topMargin=50, bottomMargin=46,
                            title='双Turing GPU上的MiniMax H3推理优化' if zh else 'MiniMax H3 on Dual Turing GPUs',
                            author='Luna-AI-Infra', subject='Engineering measurements, precision trade-offs and reproducibility')
    story = []
    lines = source.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak())
            continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['title']))
            continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]), styles['h3']))
            continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['h2']))
            continue
        if line.startswith('|'):
            rows = [line]
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i].strip())
                i += 1
            rows = [r for r in rows if not re.fullmatch(r'[|:\s-]+', r)]
            cells = [[Paragraph(inline(c.strip()), styles['headcell'] if row_id == 0 else styles['cell'])
                      for c in r.strip('|').split('|')] for row_id, r in enumerate(rows)]
            n = len(cells[0])
            weights = [1.05, 2.95] if n == 2 else [1.15, 1.7, 1.25] if n == 3 else [1.55, 1.05, 1.05, 1.05]
            widths = [width * w / sum(weights) for w in weights] if n <= 4 else [width/n]*n
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#193b53')),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f5f7f9')]),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 6), ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('TOPPADDING', (0,0), (-1,-1), 7), ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('LINEBELOW', (0,0), (-1,0), .6, colors.HexColor('#b5c6d2')),
                ('LINEBELOW', (0,-1), (-1,-1), .5, colors.HexColor('#d5dfe6')),
            ]))
            story += [table, Spacer(1,9)]
            continue
        m = re.match(r'!\[([^\]]*)\]\(([^)]+)\)', line)
        if m:
            p = (source.parent / m[2]).resolve()
            image = Image(str(p))
            native_width, native_height = image.imageWidth, image.imageHeight
            image.drawWidth = width
            image.drawHeight = width * native_height/native_width
            story += [image, Spacer(1,7)]
            continue
        text = line
        while i < len(lines) and lines[i].strip() and not lines[i].lstrip().startswith(('#','|','![','<!--','- ')):
            text += ' ' + lines[i].strip()
            i += 1
        if text.startswith('- '):
            story.append(Paragraph('- ' + inline(text[2:]), body))
        else:
            style = styles['cap'] if text.startswith(('Figure ', '图1', '图2', '图3')) else body
            story.append(Paragraph(inline(text), style))

    def page(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d8e1e8'))
        canvas.line(46, A4[1]-32, A4[0]-46, A4[1]-32)
        canvas.setFont(font, 8)
        canvas.setFillColor(colors.HexColor('#657786'))
        canvas.drawString(46, A4[1]-25, 'Luna-AI-Infra | H3 TP | Engineering Report')
        canvas.drawString(46, 24, 'Experimental results - scopes and limitations apply')
        canvas.drawRightString(A4[0]-46, 24, str(doc.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=page, onLaterPages=page)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cjk-font', required=True, type=Path, help='TrueType/TTC font containing Simplified Chinese')
    ap.add_argument('--cjk-bold', type=Path)
    args = ap.parse_args()
    pdfmetrics.registerFont(TTFont('PaperCJK', str(args.cjk_font), subfontIndex=0))
    pdfmetrics.registerFont(TTFont('PaperCJKBold', str(args.cjk_bold or args.cjk_font), subfontIndex=0))
    pdfmetrics.registerFontFamily('PaperCJK', normal='PaperCJK', bold='PaperCJKBold', italic='PaperCJK', boldItalic='PaperCJKBold')
    print(build('en', 'Times-Roman', 'Times-Bold'))
    print(build('zh-CN', 'PaperCJK', 'PaperCJKBold'))


if __name__ == '__main__':
    main()
