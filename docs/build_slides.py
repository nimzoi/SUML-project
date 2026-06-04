"""Build the SUML presentation deck (slides.pptx). Run: python docs/build_slides.py

Business-focused deck: what it does, ease of use, retraining, use cases.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

NAVY = RGBColor(0x1E, 0x27, 0x61)
TEAL = RGBColor(0x02, 0x80, 0x90)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
LIGHT = RGBColor(0xF5, 0xF7, 0xFB)
DARK = RGBColor(0x22, 0x22, 0x33)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0x5A, 0x63, 0x78)
HEAD = "Trebuchet MS"
BODY = "Calibri"
IMG = Path("docs/img")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def slide(bg=LIGHT, motif=True):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = bg
    if motif:
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), SH)
        bar.fill.solid()
        bar.fill.fore_color.rgb = TEAL
        bar.line.fill.background()
        bar.shadow.inherit = False
    return s


def text(
    s,
    left,
    top,
    width,
    height,
    lines,
    size=16,
    color=DARK,
    bold=False,
    font=BODY,
    align=PP_ALIGN.LEFT,
    anchor=MSO_ANCHOR.TOP,
    bullet=False,
    space=10,
):
    box = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space)
        run = para.add_run()
        run.text = ("•  " + line) if bullet else line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = font
        run.font.color.rgb = color
    return box


def title(s, label, color=NAVY):
    text(s, 0.7, 0.5, 12.0, 1.0, label, size=32, color=color, bold=True, font=HEAD)


def rrect(s, left, top, width, height, fill, line=None):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height)
    )
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1)
    shp.shadow.inherit = False
    return shp


def card(s, left, top, width, height, big, label):
    box = rrect(s, left, top, width, height, WHITE, line=ICE)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = big
    r.font.size = Pt(38)
    r.font.bold = True
    r.font.name = HEAD
    r.font.color.rgb = TEAL
    p2 = tf.add_paragraph()
    p2.alignment = PP_ALIGN.CENTER
    r2 = p2.add_run()
    r2.text = label
    r2.font.size = Pt(13)
    r2.font.name = BODY
    r2.font.color.rgb = MUTED


def panel(s, left, top, width, height, header, lines):
    box = rrect(s, left, top, width, height, WHITE, line=ICE)
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Pt(14)
    tf.margin_right = Pt(14)
    tf.margin_top = Pt(12)
    tf.margin_bottom = Pt(12)
    p = tf.paragraphs[0]
    rh = p.add_run()
    rh.text = header
    rh.font.size = Pt(20)
    rh.font.bold = True
    rh.font.name = HEAD
    rh.font.color.rgb = NAVY
    p.space_after = Pt(8)
    for line in lines:
        para = tf.add_paragraph()
        para.space_after = Pt(6)
        run = para.add_run()
        run.text = "•  " + line
        run.font.size = Pt(15)
        run.font.name = BODY
        run.font.color.rgb = DARK


def banner(s, left, top, width, height, label, fill=TEAL, color=WHITE, size=22):
    box = rrect(s, left, top, width, height, fill)
    box.text_frame.word_wrap = True
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    box.text_frame.margin_left = Pt(14)
    box.text_frame.margin_right = Pt(14)
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.name = HEAD
    r.font.color.rgb = color


def picture(s, name, left, top, width):
    path = IMG / name
    if path.exists():
        s.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


# --- Slide 1: title ---
s = slide(NAVY, motif=False)
rrect(s, 0, 0, 0.32, 7.5, TEAL)
text(s, 0.9, 2.3, 11.5, 1.4, "Wycena laptopa", size=54, color=WHITE, bold=True, font=HEAD)
text(
    s,
    0.9,
    3.7,
    11.5,
    0.9,
    "Aplikacja, która szacuje cenę laptopa na podstawie jego specyfikacji",
    size=22,
    color=ICE,
    font=BODY,
)
text(
    s,
    0.9,
    5.5,
    11.5,
    0.5,
    "SUML — Środowiska uruchomieniowe ML · PJATK",
    size=16,
    color=WHITE,
    bold=True,
    font=BODY,
)
text(s, 0.9, 6.0, 11.5, 0.5, "Grupa: [numery indeksów]", size=15, color=ICE, font=BODY)

# --- Slide 2: what & why ---
s = slide()
title(s, "Co to robi i po co")
text(
    s,
    0.7,
    1.9,
    6.6,
    4.6,
    [
        "Szacuje cenę laptopa na podstawie specyfikacji: marki, typu, RAM, dysków, ekranu, CPU/GPU.",
        "Wartość: trafna cena = szybsza sprzedaż i lepsza marża.",
        "Dla sklepów, platform marketplace i skupu sprzętu używanego.",
    ],
    size=18,
    bullet=True,
    space=16,
)
picture(s, "feature_scatter.png", 7.7, 2.1, 5.0)

# --- Slide 3: ease of use ---
s = slide()
title(s, "Proste w użyciu")
text(
    s,
    0.7,
    1.9,
    5.6,
    4.6,
    [
        "Otwierasz stronę, wpisujesz specyfikację, klikasz — cena w sekundę.",
        "Bez instalacji i bez znajomości kodu.",
        "Dla integracji z innymi systemami: gotowe API.",
    ],
    size=18,
    bullet=True,
    space=16,
)
picture(s, "ui.png", 6.7, 1.6, 6.0)

# --- Slide 4: accuracy ---
s = slide()
title(s, "Na ile trafne?")
card(s, 0.9, 2.4, 3.4, 2.2, "~85%", "zmienności ceny\nwyjaśnia model")
text(
    s,
    0.9,
    4.9,
    5.2,
    1.4,
    "Gwarancja: więcej RAM/SSD/lepszy CPU → nigdy niższa cena. Najważniejsze cechy: RAM, SSD, typ.",
    size=15,
    color=MUTED,
)
picture(s, "feat_importance.png", 6.6, 2.1, 6.2)

# --- Slide 5: retraining ---
s = slide()
title(s, "Model nadąża za rynkiem")
text(
    s,
    0.7,
    1.9,
    11.8,
    2.6,
    [
        "Ceny się zmieniają (nowe modele, sezon, promocje) — model można dotrenować.",
        "Dotrenowanie = podmiana pliku z danymi i jedno uruchomienie. Bez zmian w kodzie.",
        "Dzięki temu wycena pozostaje aktualna.",
    ],
    size=18,
    bullet=True,
    space=16,
)
banner(s, 0.9, 5.2, 11.5, 1.1, "Dotrenowanie = nowe dane + 1 komenda → świeży model")

# --- Slide 6: serving / integration ---
s = slide()
title(s, "Dla ludzi i dla systemów")
panel(
    s,
    0.8,
    2.0,
    5.7,
    4.0,
    "Strona (dla ludzi)",
    [
        "Formularz w przeglądarce → cena",
        "Dla pracowników i klientów",
        "Działa też samodzielnie w chmurze",
    ],
)
panel(
    s,
    6.8,
    2.0,
    5.7,
    4.0,
    "API (dla systemów)",
    [
        "Integracja z e-commerce / ERP",
        "Zwraca cenę w formacie JSON",
        "Interaktywna dokumentacja online",
    ],
)

# --- Slide 7: deployment / portability ---
s = slide()
title(s, "Uruchomienie: jedna komenda")
text(
    s,
    0.7,
    1.9,
    11.8,
    2.6,
    [
        "docker compose up — i działa, na każdym komputerze.",
        "Zero konfiguracji systemu; zależności instalują się same.",
        "Łatwe przeniesienie i odtworzenie środowiska.",
    ],
    size=18,
    bullet=True,
    space=16,
)
banner(s, 0.9, 5.2, 11.5, 1.1, "Jedna komenda uruchamia całość")

# --- Slide 8: use cases ---
s = slide()
title(s, "Zastosowania")
panel(
    s,
    0.8,
    2.0,
    5.7,
    4.2,
    "Sprzedaż",
    [
        "Szybka wstępna wycena oferty",
        "Weryfikacja, czy cena jest rynkowa",
        "Spójne ceny w całym zespole",
    ],
)
panel(
    s,
    6.8,
    2.0,
    5.7,
    4.2,
    "Skup i integracja",
    [
        "Wycena sprzętu używanego",
        "Wsparcie negocjacji",
        "Automatyczna wycena przez API",
    ],
)

# --- Slide 9: quality / trust ---
s = slide()
title(s, "Solidne i gotowe do utrzymania")
card(s, 1.4, 2.4, 3.4, 2.0, "10/10", "jakość kodu (pylint)")
card(s, 5.0, 2.4, 3.4, 2.0, "28", "testów automatycznych")
card(s, 8.6, 2.4, 3.4, 2.0, "CI", "automatyczna kontrola")
text(
    s,
    0.7,
    4.8,
    12.0,
    1.6,
    [
        "Czysty, modularny kod (data | model | app) i pełna dokumentacja.",
        "Powtarzalne środowisko — łatwe w rozwoju i utrzymaniu.",
    ],
    size=16,
    bullet=True,
    space=10,
)

# --- Slide 10: summary + demo ---
s = slide(NAVY, motif=False)
rrect(s, 0, 0, 0.32, 7.5, TEAL)
title(s, "Podsumowanie i demo", color=WHITE)
text(
    s,
    0.9,
    1.9,
    11.4,
    3.6,
    [
        "Demo na żywo: wpisujemy specyfikację → otrzymujemy cenę.",
        "Łatwe w użyciu, trafne (~85%), gotowe do dotrenowania i integracji.",
        "Repo: github.com/nimzoi/SUML-project",
    ],
    size=19,
    color=ICE,
    bullet=True,
    space=16,
)
text(s, 0.9, 6.0, 11.4, 0.8, "Dziękujemy — pytania?", size=24, color=WHITE, bold=True, font=HEAD)

prs.save("slides.pptx")
print("saved slides.pptx with", len(prs.slides), "slides")
