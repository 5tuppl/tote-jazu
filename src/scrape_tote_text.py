"""
scrape_tote_text.py — fetch REAL human-written Töte jazu text (OCR/LM side).

HONEST SCOPE: there is no clean, ready-made Töte-jazu <-> Cyrillic *parallel*
corpus to scrape (nvrislam.net is a client-side converter, not a corpus; "Kazakh
Quran with Arabic" is original-Arabic + Cyrillic, the wrong script pairing). So
this does NOT build parallel data. It collects authentic Arabic-script Kazakh
*text* (e.g. Chinese-Kazakh news) — useful as (a) a language-model prior for the
post-corrector and (b) realistic strings to render for the Stage-2 OCR study.

Run LOCALLY (your machine can reach these sites; the research sandbox cannot).
Be a good citizen: it checks robots.txt and rate-limits. Set SOURCE_URLS and the
CSS selector for your chosen source, and VERIFY the selector in your browser.
"""
import time, sys, urllib.robotparser, urllib.parse
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("pip install requests beautifulsoup4")

# ---- configure these for your chosen Arabic-script Kazakh source -------------
SOURCE_URLS = [
    # e.g. article URLs from a Chinese-Kazakh (Töte jazu) news site
]
ARTICLE_SELECTOR = "p"     # CSS selector for body text; verify in your browser
RATE_LIMIT_SEC = 2.0
OUT = "tote_text.txt"
# Arabic-script Kazakh letter range to keep lines that are actually Töte jazu
ARABIC = set(range(0x0600, 0x06FF + 1)) | {0x0674, 0x06C7, 0x06CB, 0x06D5, 0x06AD, 0x06BE}

def allowed(url):
    rp = urllib.robotparser.RobotFileParser()
    p = urllib.parse.urlparse(url)
    rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
    try: rp.read()
    except Exception: return True
    return rp.can_fetch("*", url)

def is_tote(line):
    chars = [c for c in line if not c.isspace()]
    if len(chars) < 4: return False
    ar = sum(ord(c) in ARABIC for c in chars)
    return ar / len(chars) > 0.6

def main():
    if not SOURCE_URLS:
        print("Set SOURCE_URLS to article URLs of an Arabic-script Kazakh site,")
        print("verify ARTICLE_SELECTOR in your browser, then re-run.")
        return
    seen = set()
    with open(OUT, "w", encoding="utf-8") as f:
        for url in SOURCE_URLS:
            if not allowed(url):
                print("robots.txt disallows", url); continue
            try:
                html = requests.get(url, timeout=20,
                                    headers={"User-Agent": "tote-research/0.1"}).text
            except Exception as e:
                print("fetch failed", url, e); continue
            soup = BeautifulSoup(html, "html.parser")
            for node in soup.select(ARTICLE_SELECTOR):
                for line in node.get_text("\n").splitlines():
                    line = line.strip()
                    if is_tote(line) and line not in seen:
                        seen.add(line); f.write(line + "\n")
            print("done", url, "->", len(seen), "lines so far")
            time.sleep(RATE_LIMIT_SEC)
    print(f"wrote {OUT} ({len(seen)} unique Töte lines)")

if __name__ == "__main__":
    main()
