"""
transliterate_tote.py — Cyrillic Kazakh -> Töte jazu (Kazakh Arabic script).

Follows documented orthography (Wikipedia "Kazakh alphabets"; Library of Congress
Kazakh-Arabic romanization table). The research-relevant properties:

  * Front/back vowel PAIRS share one base glyph:
        а/ә -> ا   о/ө -> و   ұ/ү -> ۇ   ы/і -> ى
  * и and й -> the same glyph ي ; у -> ۋ              (many-to-one)
  * A word-initial high-hamza ٴ (dáiekshe) marks the whole word as FRONT,
    OMITTED when the word already contains е, or к/г (imply front), or
    when it contains қ/ғ (force back).

=> The inverse (our learning task) needs WORD-LEVEL context to recover Cyrillic:
   the same glyph ۇ is ұ in a back word but ү in a front word. A per-character
   lookup table cannot do this; a sequence model can.

NOTE: This is a programmatic transliteration of REAL Kazakh vocabulary following
documented rules (the same nature as any rule-based converter, incl. nvrislam.net).
It is NOT human-scanned parallel text. A native reviewer should spot-check the
table; loanword/exception handling here is deliberately simplified.
"""

FRONT_V = set("әөүіе")          # front (soft) vowels
BACK_V  = set("аоұы")           # back (hard) vowels
# и, у are treated as neutral for harmony detection.

CONS = {
    "б":"ب","в":"ۆ","г":"گ","ғ":"ع","д":"د","ж":"ج","з":"ز","к":"ك","қ":"ق",
    "л":"ل","м":"م","н":"ن","ң":"ڭ","п":"پ","р":"ر","с":"س","т":"ت","ф":"ف",
    "х":"ح","һ":"ھ","ц":"تس","ч":"چ","ш":"ش","щ":"شش","й":"ي",
}
# vowel base glyphs (front and back of a pair collapse to one glyph)
VOWEL = {
    "а":"ا","ә":"ا", "о":"و","ө":"و", "ұ":"ۇ","ү":"ۇ", "ы":"ى","і":"ى",
    "е":"ە", "и":"ي", "у":"ۋ",
}
HAMZA = "\u0674"

def is_front_word(w):
    if any(c in BACK_V for c in w) and not any(c in FRONT_V for c in w):
        return False
    if any(c in FRONT_V for c in w):
        return True
    return False  # only neutral vowels -> treat as back (no hamza)

def needs_hamza(w):
    if not is_front_word(w):
        return False
    if "е" in w:            # е already signals front
        return False
    if "к" in w or "г" in w:  # к/г only occur in front words
        return False
    if "қ" in w or "ғ" in w:  # uvulars force back; shouldn't co-occur, but guard
        return False
    return True

def to_tote(w):
    out = []
    for c in w:
        if c in CONS:   out.append(CONS[c])
        elif c in VOWEL: out.append(VOWEL[c])
        else:           out.append(c)   # leave unknown as-is
    s = "".join(out)
    if needs_hamza(w):
        s = HAMZA + s
    return s

if __name__ == "__main__":
    # Validation against documented examples.
    tests = [
        ("түйіс",    "\u0674تۇيىس"),     # front, no е/к/г -> hamza
        ("түйістер", "تۇيىستەر"),         # has е -> NO hamza
        ("барлық",   "بارلىق"),           # back -> no hamza
        ("адамдар",  "ادامدار"),          # back -> no hamza
    ]
    ok = True
    for cyr, gold in tests:
        got = to_tote(cyr)
        mark = "OK " if got == gold else "XX "
        if got != gold: ok = False
        print(f"{mark}{cyr:10s} -> {got}   (expected {gold})")
    print("ALL PASS" if ok else "MISMATCH — rules need adjustment")
