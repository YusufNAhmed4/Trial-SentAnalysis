'''
The main scraper, taking volume pdfs and turning them into a json for data.
'''

import json
from pathlib import Path
import pymupdf as fitz
import indiv
from tqdm import tqdm

def scrape_all_pdfs():
    """
    Scrapes all PDFs for data and outputs JSONL.
    """
    all_files = [
        str(f.relative_to("Trial Data PDFs/new PDFs"))
        for f in Path("Trial Data PDFs/new PDFs").rglob("*")
        if f.is_file()
    ]
    print(all_files)
    #all_files = ["420bv.pdf"]

    with open("output.jsonl", "a", encoding="utf-8") as out:
        with tqdm(desc="Scraping cases", unit="case") as pbar:
            for file in all_files:
                cases = scrape_one_pdf(file, pbar)

                for case in cases:
                    out.write(json.dumps(case, ensure_ascii=False) + "\n")


def load_pdf_text(path):
    """
    loads in all pdf text into one string
    """
    pdf_path = "Trial Data PDFs/new PDFs/" + path
    # print(pdf_path)

    with fitz.open(pdf_path) as doc:
        if doc.page_count == 0 :
            print("Error opening PDF")
        text = indiv.append_all(doc)

    return indiv.clean_pdf_text(text)


def extract_pdf_metadata(text):
    """
    extracts year, names, and text as a string
    """
    year, text = indiv.cut_opening(text)
    names = indiv.grab_names(text)
    text = indiv.skip_to_cases(text, year)

    # print("Names for ", year, ": ", names)

    return year, names, text

def get_next_valid_title(text, year, titles, first):
    """
    Searches for next valid title
    """
    if first:
        raw, title = indiv.get_first_title(text, year)
    else:
        raw, title = indiv.get_title(text)

    while title == "Too Long" or title in titles:
        reason = "too-long title" if title == "Too Long" else "already-seen title"

        text = indiv.skip_to_next_term(text, year)
        if text is None:
            print(f"text broke after {reason}")
            return None, None, None

        raw, title = indiv.get_title(text)

        if title is None:
            return None, None, None

    return text, raw, title


def extract_case(text, raw, title, year, names):
    """
    wraps up info into a case JSON
    """
    text = indiv.skip_to_title(text, raw)

    if text is None:
        return None, None

    excerpt = indiv.get_excerpt(text)
    result = indiv.get_case_result(text)

    if result is None:
        # print("RESULT WAS NONE, TITLE: ", title)
        return None, None

    case = {
        "year": year,
        "justices": names,
        "title": title,
        "excerpt": excerpt,
        "result": result
    }

    return case, text

def scrape_one_pdf(path, pbar=None):
    """
    Scrapes one PDF and converts needed info into JSON.
    Read README for data format.
    """
    text = load_pdf_text(path)
    year, names, text = extract_pdf_metadata(text)

    cases = []
    titles = set()
    first = True

    while True:
        text, raw, title = get_next_valid_title(text, year, titles, first)
        first = False

        if text is None or title is None:
            break

        titles.add(title)

        case, text = extract_case(text, raw, title, year, names)

        if case is None:
            if year is None or text is None:
                break

            text = indiv.skip_to_next_term(text, year)

            if text is None:
                print("Could not recover after bad case:", title)
                break

            if pbar is not None:
                pbar.set_postfix({
                    "file": str(path)[:25],
                    "skipped": title[:30],
                    "cases": len(cases)
                })

            continue

        cases.append(case)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix({
                "file": str(path)[:25],
                "latest": title[:30],
                "cases": len(cases)
            })

        text = indiv.skip_to_next_term(text, year)
        if text is None:
            break

    return cases

# print("avg excerpt length: ", excerpt_length / len(cases))

if __name__ == "__main__":
    scrape_all_pdfs()
