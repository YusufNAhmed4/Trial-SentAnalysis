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
    all_files = ["429bv.pdf"]

    with open(output_file, output_type, encoding="utf-8") as out:
        with tqdm(desc="Scraping cases", unit="case") as pbar:
            for file in all_files:
                cases = scrape_one_pdf(file, pdf_dir, max_files, pbar)

                for case in cases:
                    out.write(json.dumps(case, ensure_ascii=False) + "\n")


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

        text = indiv.skip_to_next_term(text, year)

        # if title == "united states v kopp" :
        #     print("found kopp")
        #     print(text[:2500])

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
    # if title == "moody v daggett":
    #     print("found moody:", text[:1000])
    text = indiv.skip_to_title(text, raw)
    # if title == "hutto arkansas department of correction commissioner v ross":
    #     print("found hutto:", repr(text[:2600]))

    if text is None:
        print("text was none after skipping to title")
        return None, None

    excerpt = indiv.get_excerpt(text)
    # if title == "moody v. daggett" :
    #     print("MOODY FOUND")
    result = indiv.get_case_result(text)
    # if title == "moody v. daggett" :
    #     print(result)

    if result is None:
        #print("RESULT WAS NONE, TITLE: ", title)
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
            #print("fake case found")
            text = indiv.skip_to_next_term(text, year)
            # print(text[:500])
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
            # if title == "massachusetts et al v feeney on" :
            #     print("found mass v feeny: ", repr(text[:2000]))
            text = indiv.skip_to_next_term(text, year)
            # print("Skipping text")
            # if title == "massachusetts et al v feeney on" :
            #     print("found mass v feeny after: ", repr(text[:2000]))

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

        # if title == "united states v kopp":
        #     print("BEFORE JUMP: ", repr(text[:2500]))
        #     print(text.find("DAVIS"))
        #     print("")

        text = indiv.skip_to_next_term(text, year)
        # if title == "united states v kopp":
        #     print(text[:2500])
        #     print(text.find("DAVIS"))
        if text is None:
            # print("couldn't find next term after normal case work")
            break

        i += 1
        if i == int(max_files) :
            print("broke for max_files")
            break

    return cases

# print("avg excerpt length: ", excerpt_length / len(cases))

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python scrape.py <pdf_directory> <output_file> <output_type> <max_files>")
        sys.exit(1)

    scrape_all_pdfs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
