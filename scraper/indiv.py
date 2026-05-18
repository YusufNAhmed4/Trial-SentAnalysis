import fitz
import re
import unicodedata

def append_all(doc) :
    all_text = []
    i = 0
    for page in doc :
        text = page.get_text()
        all_text.append(text)
        if i == 49 :
            break
        i += 1

    text = "\n".join(all_text)
    return text

def cut_opening(text) :
    """
    ^ = beginning of doc
    .*? = match everything
    (?=...) stop at ...
    """

    text = re.sub(
        r"^.*?(?=TERM*)",
        "",
        text,
        flags=re.DOTALL
    )
    year = text[5:9]

    text = re.sub(
        r"^.*?(?=during the time of these reports*)",
        "",
        text,
        flags=re.DOTALL
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

def skip_to_cases(year, text) :
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