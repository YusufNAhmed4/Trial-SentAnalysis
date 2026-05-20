import fitz
import re
import unicodedata

def append_all(doc) :
    all_text = []
    for page in doc :
        text = page.get_text()
        all_text.append(text)

    text = "\n".join(all_text)
    return text

def clean_pdf_text(text):
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
    return text

def cut_opening(text) :
    # ^ = beginning of doc
    # .*? = match everything
    # (?=...) stop at ...

    match = re.search(
        r"TERM,\s+(\d{4})",
        text,
        flags=re.IGNORECASE
    )

    if not match:
        return None, text

    year = match.group(1)

    # Remove everything before "during the time..."
    text = re.sub(
        r"^.*?(?=during the time of these reports)",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    return year, text

def grab_names(text) :
    match = re.search(
        r"during the time of these reports\*\s*(.*?)(?=retired)",
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

    
    return names

def skip_to_cases(text, year) :
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



def get_first_title(text, year):
    match = re.search(
        rf"{re.escape(str(year))}\s*(.*?)(?=\s*(?:Certiorari\s+to\s+the|Appeal\s+from\s+the|On\s+petition\s+for)(?!\s+same\b))",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if not match:
        return None

    title = match.group(1).strip()

    # Clean up multi-line title into one line
    title = " ".join(
        line.strip()
        for line in title.splitlines()
        if line.strip()
    )
    # print(repr(title))

    return title

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
    match = re.search(
        r"(?:Syllabus|Per\s+Curiam)\s*(.*?)(?=\s*(?:Certiorari\s+to\s+the|Appeal\s+from\s+the|On\s+petition\s+for)(?!\s+same\b))",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # print(match)
    if not match:
        return None

    title = match.group(1).strip()

    # Clean up multi-line title into one line
    if len(title) >= 1000:
        # print("Title hit 1000 characters; likely bad match. Skipping case.")
        return "None"

    return " ".join(title.split())


# .*? = match all chars
# (.*?) = get all chars until the ?

def get_excerpt(text):
    match = re.search(
        r"Decided.*?\n\s*(.*?)(?=\bHeld\s*:)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )
    if match :
        to_ret = match.group(1).strip()
        to_ret = to_ret.replace("\n", " ")
        return to_ret
    else :
        return None
    

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
        return to_ret.strip().replace("\n", "")
    else :
        print("couldn't find result")
        return None
