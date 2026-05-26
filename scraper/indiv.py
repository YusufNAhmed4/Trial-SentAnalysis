'''
Functions which compile, clean, and collate all case data
'''

import re
import unicodedata

def append_all(doc) :
    """
    Appends all pages from the open document into one string
    """
    all_text = []
    for page in doc :
        text = page.get_text()
        all_text.append(text)

    text = "\n".join(all_text)
    return text

def clean_pdf_text(text):
    """
    Cleans out most unprintable characters
    """
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }

    for old, new in ligatures.items():
        text = text.replace(old, new)

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"[-‐-–—\u00AD]\s+\n?\s*", "", text)

    text = text.replace("\xa0", " ")
    text = ''.join(
        c for c in text
        if c.isprintable() or c == '\n'
    )
    text = text.replace("\u2019", "'")
    text = text.replace("\u00a7", "S. ")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u00b7", " ")
    text = text.replace("affrmed", "affirmed")
    return text

def cut_opening(text) :
    """
    Removes opening of volume, grabbing publication year and skipping to justice names.
    """

    match = re.search(
        r"T\s*E\s*R\s*M,\s+(\d{1,4})",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        print("no opening found")
        return None, text

    year = match.group(1)

    # ^ = beginning of doc
    # .*? = match everything
    # (?=...) stop at ...
    text = re.sub(
        r"^.*?(?=d\s*u\s*r\s*i\s*n\s*g "
        r"t\s*h\s*e t\s*i\s*m\s*e "
        r"o\s*f t\s*h\s*e\s*s\s*e "
        r"r\s*e\s*p\s*o\s*r\s*t\s*s)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return year, text

def spaced(word):
    """
    Allows for spacing within words
    """
    return r"\s*".join(map(re.escape, word))

def grab_names(text) :
    """
    Returns justice names
    """
    match = re.search(
        r"d\s*u\s*r\s*i\s*n\s*g t\s*h\s*e "
        r"t\s*i\s*m\s*e o\s*f t\s*h\s*e\s*s\s*e "
        r"r\s*e\s*p\s*o\s*r\s*t\s*s"
        r"\**\s*(.*?)(?=r\s*e\s*t\s*i\s*r\s*e\s*d|officers)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )


    names = []

    if match:
        names_block = match.group(1)

        names_block = unicodedata.normalize("NFKC", names_block)
        names_block = re.sub(
            r'(?<=[.,])\d+',
            '',
            names_block
        )

        names = [
            line.strip()
            for line in names_block.splitlines()
            if line.strip()
        ]
    to_ret = []
    # print(names)
    for name in names :
        if len(name) < 15 :
            continue
        # print("name: ", name)
        name_title = name.split(",")
        # print("nametitle: ", name_title)
        name_split = name_title[0].split()
        # print("namesplit: ", name_split)
        to_ret.append(name_split[-1].lower())
    if len(to_ret) > 9 :
        return to_ret[:9]
    if len(to_ret) == 0 :
        print("No names found")
        return None
    # print(to_ret)
    return to_ret

def skip_to_cases(text, year) :
    """
    Jumps to volume body of cases
    """
    pattern = (
            rf"^.*?"
            rf"(?= OF THE UNITED STATES\s+"
            rf"AT\s+"
            rf"OCTOBER TERM,\s*{re.escape(str(year))})"
        )

    text = re.sub(
        pattern,
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    return text

def skip_to_next_term(text, year):
    """
    Skips to next instance of "Term, year" available
    Next case in modern volumes, next page in old volumes
    """
    matches = list(re.finditer(
        rf"Term,\s*{re.escape(str(year))}\s*\n",
        text,
        flags=re.IGNORECASE
    ))

    # Need at least two occurrences:
    # first = current case
    # second = next case
    if len(matches) >= 2:
        return text[matches[1].start():]

    return None

def skip_to_title(text, raw_title) :
    """
    Skips to nearest case defined by title
    """
    if text is None or raw_title is None:
        return None

    match = re.search(
        re.escape(raw_title),
        text,
        flags=re.IGNORECASE
    )

    if match:
        return text[match.start():]

    return None


def get_first_title(text, year):
    """
    Gets title of first case in volume (special case)
    """
    match = re.search(
        rf"{re.escape(str(year))}\s*"
        rf"(.*?)(?=\s*(?:Certiorari\s+to\s+the|Appeal\s+from\s+the|On\s+petition\s+for)"
        rf"(?!\s+same\b))",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not match:
        return None, None

    raw_title = match.group(1).strip()

    # Clean up multi-line title into one line
    to_ret = " ".join(
        line.strip()
        for line in raw_title.splitlines()
        if line.strip()
    )
    title = unicodedata.normalize("NFKD", to_ret)

    # Convert to lowercase
    title = title.lower()

    # Remove apostrophes and periods
    title = re.sub(r"[.'’]", "", title)

    # Replace punctuation with spaces
    title = re.sub(r"[^a-z0-9\s]", " ", title)

    # Collapse repeated spaces
    title = re.sub(r"\s+", " ", title)

    return raw_title, title.strip()

# () = get the thing inside here
# [^\n]+ = match one or more chars which aren't a newline
#     [] = parentheses
#     ^ = not
#     \n = newline
#     + = more than one

# \s* = there may be whitespace
# \n = on the next line
# \s* = there may be whitespace
# Certiorari\s+to\s+the = match "Certiorari to the" with >=one whitespace between each word
# (?!) = text cannot match this


def get_title(text):
    """
    Gets the next available title (normal case)
    """
    match = re.search(
        r"(?:Syllabus|Per\s+Curiam)\s*(?:\d{1,4}\s+U\.\s*S\.\s*)?(?:\d{1,4}\s+U\s*S\s*)?"
        r"(.*?)(?=\s*(?:Certiorari\s+to\s+the|Appeal\s+from\s+the|On\s+petition\s+for)"
        r"(?!\s+same\b))",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # print(match)
    if not match:
        return None, None

    raw_title = match.group(1).strip()

    # Clean up multi-line title into one line
    if len(raw_title) >= 200:
        # print("Title hit 1000 characters; likely bad match. Skipping case.")
        return "raw_too_long", "Too Long"

    title = unicodedata.normalize("NFKD", " ".join(raw_title.split()))

    # Convert to lowercase
    title = title.lower()

    # Remove apostrophes and periods
    title = re.sub(r"[.'’]", "", title)

    # Replace punctuation with spaces
    title = re.sub(r"[^a-z0-9\s]", " ", title)

    # Collapse repeated spaces
    title = re.sub(r"\s+", " ", title)

    title = re.sub(r"\d{1,4} u s ", "", title)

    return raw_title, title.strip()


# .*? = match all chars
# (.*?) = get all chars until the ?

def get_excerpt(text):
    """
    Gets excerpt of case
    """
    # print(text[:4000])
    result_pattern = (
        r"\n\s*"
        r"(?:Certiorari\s+granted;\s*)?"
        r"(?:"
            r"\b\d{1,4}\s+"
            r"[A-Za-z]\.\s+"
            r"\d+[A-Za-z]\s+"
            r"\d{1,4},\s*"
            r"([A-Za-z ,]+?)"
        r"|"
            r"\b\d{1,4}\s+Fed\.\s+Appx\.\s+"
            r"\d{1,4},\s*"
            r"(?:\d{1,4},\s*)*"
            r"([A-Za-z ,\n]+?)"
        r"|"
            r"\b\d{1,4}\s+F\.\s+Supp\.\s+"
            r"\d{1,4},\s*"
            r"(?:\d{1,4},\s*)*"
            r"([A-Za-z ,\n]+?)"
        r"|"
            r"Certiorari\s*granted;"
        r")"
        r"\s*\."
    )
    decided_match = re.search(
        r"\s*D\s*e\s*c\s*i\s*d\s*e\s*d\b.*?\n",
        text,
        flags=re.IGNORECASE
    )

    if not decided_match:
        return None

    start = decided_match.end()

    held_match = re.search(
        rf"\s*H\s*e\s*l\s*d\s*:|{result_pattern}",
        text[start:],
        flags=re.IGNORECASE
    )

    if not held_match:
        return None

    end = start + held_match.start()
    excerpt = text[start:end]

    # Normalize weird PDF unicode
    excerpt = unicodedata.normalize("NFKD", excerpt)

    # Remove footnote markers like:
    # "Marshall.1" or "decision[2]"
    excerpt = re.sub(r"\[\d+\]", " ", excerpt)
    excerpt = re.sub(r"(?<!\()\b\d+\b(?!\))", " ", excerpt)

    # Replace line breaks/tabs with spaces
    excerpt = re.sub(r"[\n\t]+", " ", excerpt)

    # Remove repeated punctuation artifacts
    excerpt = re.sub(r"[•■□]+", " ", excerpt)

    # Collapse multiple spaces
    excerpt = re.sub(r"\s+", " ", excerpt)

    if len(excerpt) > 4000 :
        print("excerpt too long")
        # print(excerpt[:4000])
        return "null"
    if len(excerpt) == 0:
        print("excerpt too short")
        return "null"

    return excerpt.strip()


# \d{1,4}        # 1-4 ints
# \s+            # space
# [A-Za-z]\.     # Letter and period
# \s+            # space
# \d+[A-Za-z]    # number and letter
# \s+            # space
# \d{1-4}        # 1-4 ints
# ,              # comma
# \s*            # optional spaces
# [A-Za-z]+      # keyword

def get_case_result(text):
    """
    Finds result of case
    """
    pattern = (
        r"\n\s*"
        r"(?:Certiorari\s+granted;\s*)?"
        r"(?:"
            r"\b\d{1,4}\s+"
            r"[A-Za-z]\.\s+"
            r"\d+[A-Za-z]\s+"
            r"\d{1,4},\s*"
            r"([A-Za-z ,]+?)"
        r"|"
            r"\b\d{1,4}\s+Fed\.\s+Appx\.\s+"
            r"\d{1,4},\s*"
            r"(?:\d{1,4},\s*)*"
            r"([A-Za-z ,\n]+?)"
        r"|"
            r"\b\d{1,4}\s+F\.\s+Supp\.\s+"
            r"\d{1,4},\s*"
            r"(?:\d{1,4},\s*)*"
            r"([A-Za-z ,\n]+?)"
        r")"
        r"\s*\."
    )

    # print(repr(text[:2000]))
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match :
        # print("TO CONSIDER: ", match)
        # print(repr(text[54300:54355]))
        # print("GROUP ONE: ", match.group(1))
        # print("GROUP TWO: ", match.group(2))
        to_ret = to_ret = match.group(1) or match.group(2) or match.group(3)
        # print(to_ret)
        to_ret = to_ret.strip().replace("\n", "")
        words = to_ret.split()
        if words[0] != "affirmed" and words[0] != "reversed" and words[0] != "vacated" :
            return None
        return words[0]

    # print("couldn't find result")
    return None
