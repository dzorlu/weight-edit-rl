"""Answer grading shared by the sweep and the search. Pure Python, no dependencies."""
import re


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(s):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
        k += 1
    return None


def _strip_text(a):
    b = re.sub(r"\\(?:text|mbox|mathrm|textbf)\{[^{}]*\}", "", a)
    if re.sub(r"[\s\\,;!.]", "", b):
        return b
    return re.sub(r"\\(?:text|mbox|mathrm|textbf)\{([^{}]*)\}", r"\1", a)


def norm_math(a):
    a = a.strip()
    a = a.replace("\\displaystyle", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    a = _strip_text(a)
    for x in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad", "^\\circ", "^{\\circ}", "\\%", "%", "\\$", "$", "\n", " ", "~"]:
        a = a.replace(x, "")
    a = a.rstrip(".")
    if "=" in a and "\\begin" not in a:
        lhs, rhs = a.rsplit("=", 1)
        if rhs and not re.search(r"[<>]", lhs):
            a = rhs
    a = re.sub(r"^[a-zA-Z]\\in", "", a)
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\frac\{([^{}]+)\}(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\frac(\d)\{([^{}]+)\}", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", a)
    a = re.sub(r"\^\{(\w)\}", r"^\1", a)
    a = re.sub(r"_\{(\w)\}", r"_\1", a)
    a = re.sub(r"\\begin\{pmatrix\}", "[", a); a = re.sub(r"\\end\{pmatrix\}", "]", a)
    a = re.sub(r"\\\\\[\d+pt\]", r"\\\\", a)
    a = re.sub(r"\\frac\{(-?\d+)\}\{(\d+)\}", r"\1/\2", a) if ("[" in a) else a
    m = re.fullmatch(r"\(?([A-Ja-j])\)?", a)
    if m:
        return m.group(1).upper()
    if a.startswith("\\{") and a.endswith("\\}"):
        a = a[2:-2]
    return a


def to_num(a):
    a = a.replace(",", "")
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", a)
    if m and int(m.group(3)) != 0:
        v = int(m.group(2)) / int(m.group(3))
        return -v if m.group(1) else v
    m = re.fullmatch(r"(-?\d+)/(-?\d+)", a)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    try:
        return float(a)
    except Exception:
        return None


def math_equal(pred, gold):
    x, g = norm_math(pred), norm_math(gold)
    if x == g:
        return True
    fx, fg = to_num(x), to_num(g)
    if fx is not None and fg is not None:
        return abs(fx - fg) < 1e-9
    if "(" not in x and "[" not in x and "(" not in g and "[" not in g and "," in x and "," in g:
        return sorted(x.split(",")) == sorted(g.split(","))
    return False


def grade_math(text, gold):
    """Returns (extracted, correct). Uses only the text after the last </think> when present."""
    tail = text.split("</think>")[-1] if "</think>" in text else text
    b = last_boxed(tail)
    if b is None:
        return None, False
    return b, math_equal(b, gold)


def sci_number(b):
    """Last number in a boxed expression, with a x 10^n factor if present. Returns (value, is_bare_number)."""
    s = b.replace("\\,", "").replace("\\!", "").replace("{,}", "").replace(" ", "")
    bare = re.fullmatch(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?|[-+]?\.\d+|[-+]?\d[\d,]*\.?\d*(?:\\times|\\cdot|\*)10\^\{?[-+]?\d+\}?", s) is not None
    last = None
    for m in re.finditer(r"([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)(?:(?:\\times|\\cdot|\*|x)10\^\{?([-+]?\d+)\}?)?", s):
        try:
            v = float(m.group(1).replace(",", ""))
        except Exception:
            continue
        if m.group(2):
            v *= 10 ** int(m.group(2))
        last = (v, m.start())
    return (last[0] if last else None), bare


def grade_sci(text, gold, tol=0.05):
    b = last_boxed(text)
    if b is None:
        return None, False, False
    v, bare = sci_number(b)
    try:
        g = float(gold.replace("+", "").replace(",", ""))
    except Exception:
        return b, False, bare
    if v is None:
        return b, False, bare
    ok = abs(v - g) <= tol * abs(g) if g != 0 else abs(v) < 1e-6
    return b, ok, bare
