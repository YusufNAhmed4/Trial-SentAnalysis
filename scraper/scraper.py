import fitz
import re
import unicodedata
import indiv
import json



pdf_path = "Trial Data PDFs/557bv.pdf"
doc = fitz.open(pdf_path)

text = indiv.append_all(doc)
text = indiv.clean_pdf_text(text)

year, text = indiv.cut_opening(text)

names = indiv.grab_names(text)

text = indiv.skip_to_cases(text, year)

cases = []
first = True
while True:
    if first :
        title = indiv.get_first_title(text, year)
        first = False
    else:
        title = indiv.get_title(text)
    # print(title)

    if title is None:
        break

    excerpt = indiv.get_excerpt(text)

    # indiv.debug_case_results(text)
    result = indiv.get_case_result(text)

    # print(title)
    # print(excerpt)
    # print("RESULT: ", result)
    # print("\n")

    if result is None:
        break

    cases.append([
        year,
        names,
        title,
        excerpt,
        result
    ])
    # Cut off everything up through the current case result
    text = indiv.skip_to_next_term(text, year)

with open("output.txt", "w", encoding="utf-8") as f:
    json.dump(cases, f, indent=4)