"""Text normalisation for business names / addresses.

Everything here is pure-python per-string; callers parallelise with
multiprocessing (see prepare.py).
"""
import re
from unidecode import unidecode

# ---------------------------------------------------------------- constants
TLD_RE = re.compile(r"(?:^|\s)(?:https?://)?(?:www\.)?([a-z0-9\-]+)\.(?:com|net|org|biz|info|co\.in|in|us|co)\b")
NON_ALNUM = re.compile(r"[^a-z0-9]+")
NON_ALNUM_KEEP_SEP = re.compile(r"[^a-z0-9,]+")
DIGITS = re.compile(r"\d+")

# Legal forms / honorifics / generic filler. Extended at runtime with the
# noise vocabulary learned from ground truth (vocab.py).
BASE_NAME_STOP = set("""
inc incorporated incorporation llc l l c ltd limited pvt private corp corporation co company
plc llp lp pllc pc pa the and of smt shri sri m s ms mr dr
""".split())

ADDR_ABBR = {
    "street": "st", "str": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "boulevard": "blvd", "lane": "ln", "court": "ct", "place": "pl", "circle": "cir",
    "highway": "hwy", "parkway": "pkwy", "trail": "trl", "terrace": "ter", "square": "sq",
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw", "suite": "ste", "apartment": "apt", "floor": "fl",
    "building": "bldg", "mount": "mt", "saint": "st", "fort": "ft", "center": "ctr", "centre": "ctr",
    "point": "pt", "heights": "hts", "junction": "jct", "expressway": "expy", "freeway": "fwy",
    "way": "wy", "crossing": "xing", "cross": "crs", "main": "main", "nagar": "ngr",
    "colony": "col", "sector": "sec", "phase": "ph", "near": "nr", "opposite": "opp", "opp": "opp",
}
# tokens that carry no identity inside an address
ADDR_STOP = set("""
no num number door plot hno h unit apt ste fl po box pmb p o b bldg flat shop site survey
the of and at in near nr opp behind
""".split())

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
    "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma",
    "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo", "montana": "mt",
    "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
    "district of columbia": "dc", "puerto rico": "pr",
}
IN_STATES = {
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr", "himachal pradesh": "hp",
    "jharkhand": "jh", "karnataka": "ka", "kerala": "kl", "madhya pradesh": "mp",
    "maharashtra": "mh", "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj", "sikkim": "sk",
    "tamil nadu": "tn", "telangana": "tg", "tripura": "tr", "uttar pradesh": "up",
    "uttarakhand": "uk", "west bengal": "wb", "delhi": "dl", "new delhi": "dl",
    "jammu and kashmir": "jk", "ladakh": "la", "puducherry": "py", "pondicherry": "py",
    "chandigarh": "ch", "dadra and nagar haveli": "dn", "daman and diu": "dd", "lakshadweep": "ld",
    "andaman and nicobar islands": "an",
}
# common alternate codes seen in the data (TG/TS, OR/OD ...)
IN_STATE_ALIASES = {"ts": "tg", "or": "od", "ct": "cg", "uk": "uk", "ut": "uk", "tl": "tg"}


def _skel_map():
    return str.maketrans({"c": "k", "q": "k", "w": "v", "z": "j"})


_SKEL_TR = _skel_map()
_VOWELS_H = re.compile(r"[aeiouyh]")
_REPEAT = re.compile(r"(.)\1+")
_LEET_TR = str.maketrans("0134568", "oleasgb")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")


# ---------------------------------------------------------------- helpers
def to_ascii(s: str) -> str:
    if not s:
        return ""
    return s if s.isascii() else unidecode(s)


def skeleton(tok: str) -> str:
    """Phonetic consonant skeleton; robust to Indic transliteration noise.
    'maarkettiNg' -> 'mrktng', 'marketing' -> 'mrktng', 'dillii' -> 'dl'."""
    if not tok:
        return ""
    t = tok.lower().translate(_SKEL_TR).replace("ph", "f").replace("x", "ks")
    head, rest = t[0], t[1:]
    rest = _VOWELS_H.sub("", rest)
    return _REPEAT.sub(r"\1", head + rest)


# ---------------------------------------------------------------- name
def norm_name(raw: str):
    """-> (name_norm, is_domain, is_native_script)."""
    if not raw:
        return "", 0, 0
    native = 0 if raw.isascii() else int(sum(ord(c) > 0x2FF for c in raw) > 2)
    s = to_ascii(raw).lower().replace("&", " and ").replace("'", "")
    dom = 0
    m = TLD_RE.search(s)
    if m:
        dom = 1
        s = TLD_RE.sub(lambda mm: " " + mm.group(1).replace("-", " ") + " ", s)
    s = NON_ALNUM.sub(" ", s).strip()
    s = " ".join(deleet(t) for t in s.split())
    return s, dom, native


def deleet(t: str) -> str:
    """Undo OCR/leet substitutions: 'c1inic'->'clinic', '5ervices'->'services',
    'lnc'->'inc', 'lndia'->'india'."""
    if not t.isalpha():
        if t.isdigit() or _ORDINAL.match(t) or sum(c.isalpha() for c in t) < 2:
            return t
        t = t.translate(_LEET_TR)
    if t.startswith("ln") and len(t) > 2:
        t = "in" + t[2:]
    return t


def core_tokens(name_norm: str, stop: set):
    toks = name_norm.split()
    core = [t for t in toks if t not in stop]
    return core if core else toks


# ---------------------------------------------------------------- address
def norm_addr(raw: str):
    """-> (addr_norm, nums, state_code).

    addr_norm: abbreviation-canonicalised tokens, components separated by ','.
    nums: space separated digit groups in order of appearance.
    """
    if not raw:
        return "", "", ""
    s = to_ascii(raw).lower().replace("'", "")
    s = NON_ALNUM_KEEP_SEP.sub(" ", s)
    comps = [c.strip() for c in s.split(",") if c.strip()]
    state = ""
    out = []
    for c in comps:
        if c in US_STATES:
            state = US_STATES[c]; continue
        if c in IN_STATES:
            # 'delhi' is both a city and a state; keep it as a token too
            state = IN_STATES[c]
            if c not in ("delhi",):
                continue
        toks = [ADDR_ABBR.get(t, t) for t in c.split()]
        out.append(" ".join(toks))
    if not state and out:
        last = out[-1]
        if len(last) == 2 and last.isalpha():
            state = IN_STATE_ALIASES.get(last, last)
            out = out[:-1]
    addr = ", ".join(out)
    nums = " ".join(DIGITS.findall(addr))
    return addr, nums, state


def addr_tokens(addr_norm: str):
    return [t for t in NON_ALNUM.sub(" ", addr_norm).split() if t not in ADDR_STOP]
