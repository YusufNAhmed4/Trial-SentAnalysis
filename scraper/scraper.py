import fitz
import re
import unicodedata
import indiv



pdf_path = "Trial Data PDFs/557bv.pdf"
doc = fitz.open(pdf_path)

text = indiv.append_all(doc)

year, text = indiv.cut_opening(text)

names = indiv.grab_names(text)

# print(names)


text = indiv.skip_to_cases(year, text)


cleaned = "".join(
    c for c in text
    if c.isprintable() or c in "\n\t"
)

with open("output.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(names))
    f.write("\n")
    f.write(cleaned)