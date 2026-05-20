'''
The main scraper, taking a volume pdf and turning it into a json for data.
'''

import json
from pathlib import Path
import fitz
import indiv
from tqdm import tqdm

def scrape_all_pdfs() :
    """
    Scrapes all PDFs for data and outputs a json into a txt file
    """

    all_files = [f.name for f in Path('Trial Data PDFs').iterdir() if f.is_file()]

    all_cases = []

    for file in all_files:
        cases = scrape_one_pdf(file)
        all_cases.extend(cases)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=4)
    print(len(all_cases))


def scrape_one_pdf(path) :
    """
    Scrapes one PDF and converts needed info into JSON
    Read README for data format
    """
    pdf_path = "Trial Data PDFs/" + path
    with fitz.open(pdf_path) as doc:
        text = indiv.append_all(doc)

    text = indiv.clean_pdf_text(text)

    year, text = indiv.cut_opening(text)
    names = indiv.grab_names(text)
    text = indiv.skip_to_cases(text, year)

    cases = []
    titles = set()
    excerpt_length = 0
    first = True

    pbar = tqdm(desc="Processing cases", unit="case")

    while True:
        if first:
            raw, title = indiv.get_first_title(text, year)
            first = False
        else:
            raw, title = indiv.get_title(text)

        if title is None:
            break

        # Skip bad/repeated titles until a valid one appears
        while title == "Too Long" or title in titles:
            reason = "too-long title" if title == "Too Long" else "already-seen title"

            text = indiv.skip_to_next_term(text, year)
            if text is None:
                print(f"text broke after {reason}")
                break

            raw, title = indiv.get_title(text)

            if title is None:
                break

        if text is None or title is None:
            break


        titles.add(title)
        text = indiv.skip_to_title(text, raw)
        # print(text[:200])
        excerpt = indiv.get_excerpt(text)

        result = indiv.get_case_result(text)
        if result is None:
            break

        cases.append({
            "year": year,
            "justices": names,
            "title": title,
            "excerpt": excerpt,
            "result": result
        })
        excerpt_length += len(excerpt)
        text = indiv.skip_to_next_term(text, year)
        if text is None :
            print("text broke regular")
            break

        pbar.update(1)
        pbar.set_postfix({
            "latest": title[:30],
            "total": len(cases)
        })

    pbar.close()
    return cases

# print("avg excerpt length: ", excerpt_length / len(cases))

if __name__ == "__main__":
    scrape_all_pdfs()
