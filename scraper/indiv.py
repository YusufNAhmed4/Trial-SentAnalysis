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
        r"TERM,\s+(\d{4})",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None, text

    year = match.group(1)

    # ^ = beginning of doc
    # .*? = match everything
    # (?=...) stop at ...
    text = re.sub(
        r"^.*?(?=during the time of these reports)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return year, text

def grab_names(text) :
    """
    Returns justice names
    """
    match = re.search(
        r"during the time of these reports\**\s*(.*?)(?=retired)",
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
    for name in names :
        name_title = name.split(",")
        name_split = name_title[0].split(" ")
        to_ret.append(name_split[-1].lower())
    return to_ret

def skip_to_cases(text, year) :
    """
    Jumps to volume body of cases
    """
    pattern = (
            rf"^.*?"
            rf"(?=CASES ADJUDGED\s+"
            rf"IN THE\s+"
            rf"SUPREME COURT OF THE UNITED STATES\s+"
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
        r"(?:Syllabus|Per\s+Curiam)\s*"
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
    if len(raw_title) >= 1000:
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

    return raw_title, title.strip()


# .*? = match all chars
# (.*?) = get all chars until the ?

def get_excerpt(text):
    """
    Gets excerpt of case
    """
    match = re.search(
        r"Decided.*?\n\s*(.*?)(?=\bHeld\s*:)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not match:
        return None

    excerpt = match.group(1)

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
            r"([A-Za-z ]+?)"
        r"|"
            r"\b\d{1,4}\s+Fed\.\s+Appx\.\s+"
            r"\d{1,4},\s*"
            r"(?:\d{1,4},\s*)*"
            r"([A-Za-z ,\n]+?)"
        r")"
        r"\s*\."
    )

    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match :
        # print("TO CONSIDER: ", match)
        # print("GROUP ONE: ", match.group(1))
        # print("GROUP TWO: ", match.group(2))
        to_ret = to_ret = match.group(1) or match.group(2)
        to_ret = to_ret.strip().replace("\n", "")
        words = to_ret.split()
        return words[0]

    print("couldn't find result")
    return None
