import fitz
import re
import unicodedata
import indiv
import json
from tqdm import tqdm



pdf_path = "Trial Data PDFs/502bv.pdf"
doc = fitz.open(pdf_path)

text = indiv.append_all(doc)
text = indiv.clean_pdf_text(text)

year, text = indiv.cut_opening(text)
names = indiv.grab_names(text)
text = indiv.skip_to_cases(text, year)

# print(repr(year))

cases = []
titles = set()
excerpt_length = 0
first = True
i = 0

pbar = tqdm(desc="Processing cases", unit="case")

while True:
    if first :
        title = indiv.get_first_title(text, year)
        first = False
    else:
        title = indiv.get_title(text)

    if title == "None" :
        text = indiv.skip_to_next_term(text, year)
        if text is None :
            print("text broke 1")
        continue

    if title is None:
        break

    if title in titles :
        text = indiv.skip_to_next_term(text, year)
        if text is None :
            print("text broke 3")
        continue
    
    titles.add(title)
    excerpt = indiv.get_excerpt(text)

    result = indiv.get_case_result(text)

    # print(repr(title))
    # print(excerpt)
    # print(result)

    if result is None:
        break

    cases.append([
        year,
        names,
        title,
        excerpt,
        result
    ])
    excerpt_length += len(excerpt)
    text = indiv.skip_to_next_term(text, year)
    if text is None :
        print("text broke 2")
    i += 1
    if i == 50 :
        break

    pbar.update(1)
    pbar.set_postfix({
        "latest": title[:30],
        "total": len(cases)
    })

pbar.close()

with open("output.txt", "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=4)

# print("avg excerpt length: ", excerpt_length / len(cases))