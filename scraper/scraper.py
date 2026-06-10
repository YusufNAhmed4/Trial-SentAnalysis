'''
The main scraper, taking volume pdfs and turning them into a json for data.
'''

import sys
import json
from pathlib import Path
import pymupdf as fitz
import indiv
from tqdm import tqdm

def scrape_all_pdfs(pdf_dir, output_file, output_type, max_files):
    """
    Scrapes all PDFs for data and outputs JSONL.
    """
    all_files = [
        str(f.relative_to(pdf_dir))
        for f in Path(pdf_dir).rglob("*")
        if f.is_file()
    ]
    print(all_files)
    # all_files = ["551bv.pdf"]
    # num_cases = 0

    with open(output_file, output_type, encoding="utf-8") as out:
        with tqdm(desc="Scraping cases", unit="case") as pbar:
            for file in all_files:
                cases = scrape_one_pdf(file, pdf_dir, max_files, pbar)
                # num_cases += len(cases)
                for case in cases:
                    out.write(json.dumps(case, ensure_ascii=False) + "\n")
    # print("Avg cases per file: ", num_cases / len(all_files))


def load_pdf_text(path, pdf_dir):
    """
    loads in all pdf text into one string
    """
    pdf_path = pdf_dir + "/" + path
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
        # print("title not found, searching again")
        reason = "too-long title" if title == "Too Long" else "already-seen title"

        # print(f"messed up title because of {reason}")
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
        print("text was none after skipping to title")
        return None, None

    excerpt, text = indiv.get_excerpt(text)
    result = indiv.get_case_result(text)


    if result is None:
        # print("RESULT WAS NONE, TITLE: ", title)
        return None, text

    case = {
        "year": year,
        "justices": names,
        "title": title,
        "excerpt": excerpt,
        "result": result
    }

    return case, text

def scrape_one_pdf(path, pdf_dir, max_files, pbar=None):
    """
    Scrapes one PDF and converts needed info into JSON.
    Read README for data format.
    """
    # print("MAX FILES: ", max_files)
    text = load_pdf_text(path, pdf_dir)
    year, names, text = extract_pdf_metadata(text)

    cases = []
    titles = set()
    first = True

    i = 0
    while True:
        text, raw, title = get_next_valid_title(text, year, titles, first)
        first = False
        # print("considering title:", title)

        if text is None or title is None:
            # print("couldn't find next title")
            break
        if "Certiorari dismissed" in text[text.index(raw):text.find(raw) + 500] :
            # print("fake case found")
            text = indiv.skip_to_next_term(text, year)
            continue

        titles.add(title)


        case, text = extract_case(text, raw, title, year, names)

        if case is None:
            if year is None:
                print("couldn't find next title year case")
                break
            if text is None:
                print("couldn't find next title text case")
                break

            # print("skipping after case failure")
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

        # print("skipping after normal case flow")
        text = indiv.skip_to_next_term(text, year)

        if text is None:
            # print("couldn't find next term after normal case work")
            break

        i += 1
        if i == int(max_files) :
            print("broke for max_files")
            break

    return cases

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python scrape.py <pdf_directory> <output_file> <output_type> <max_files>")
        sys.exit(1)

    scrape_all_pdfs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
