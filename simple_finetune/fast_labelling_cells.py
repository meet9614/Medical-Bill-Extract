# =============================================================================
# FASTER LABELLING - two cells to replace the one-page-at-a-time loop
#
# Why: input() in Colab makes you click, type, press Enter and wait for the next
# image to render, 50 times. This instead shows every page in one scroll, then
# takes all 50 labels in a single string.
#
# Paste CELL A and CELL B as two new cells, after the "turn PDFs into images"
# cell. Do not run the old labelling cell.
# =============================================================================


# ---------------------------------------------------------------- CELL A -----
# Shows every page with a number next to it. Scroll through and write the
# digits down somewhere - a notes app, or on paper.

from IPython.display import display, Image as ShowImage

print("Write down one digit per page as you scroll.\n")
print("  1 = Bill Summary   only category totals, no individual rows")
print("  2 = Bill Detail    a table of individual services with amounts")
print("  3 = Pharmacy Bill  medicines / drugs, often with batch and expiry")
print("  4 = Lab Bill       only pathology / diagnostic tests")
print("  5 = Other          anything else (implant invoice, cover page)")
print()
print("If a page has BOTH category subtotals and individual rows, call it 2.")
print("Apply that rule every time - being consistent matters more than being")
print("right on genuinely ambiguous pages.")
print("=" * 60)

for index, page in enumerate(all_pages):
    print(f"\n>>> PAGE {index}   ({page['document']}, page {page['page_number']})")
    display(ShowImage(filename=page["image_path"], width=440))

print("\n" + "=" * 60)
print(f"That was {len(all_pages)} pages, numbered 0 to {len(all_pages) - 1}.")
print("Now fill in the string in the next cell.")


# ---------------------------------------------------------------- CELL B -----
# Put your digits between the triple quotes. Spaces and line breaks are fine,
# so you can group them per document to keep your place.

MY_LABELS = """
2 2
3 3 3
3
2 2
5 5 5
2 2 2
3
3
2 2 2
1 2 2
3
2 2 2 2 2 2
2 2 2 2 2
2 2 2 2 2 2
2 2 2 2 2 2 2 2 2 2
"""

# -----------------------------------------------------------------------------
import json

digits = MY_LABELS.split()

if len(digits) != len(all_pages):
    print(f"PROBLEM: you gave {len(digits)} labels but there are {len(all_pages)} pages.")
    print("Count again - the numbers must match exactly.\n")
    print("Pages in order:")
    for index, page in enumerate(all_pages):
        print(f"  {index:>3}  {page['document']} page {page['page_number']}")
else:
    for page, digit in zip(all_pages, digits):
        if digit not in ["1", "2", "3", "4", "5"]:
            raise ValueError(f"'{digit}' is not a valid label - use 1 to 5 only")
        page["label"] = LABEL_OPTIONS[int(digit) - 1]

    with open(LABELS_FILE, "w") as f:
        json.dump(all_pages, f, indent=2)

    print(f"Labelled all {len(all_pages)} pages and saved to {LABELS_FILE}\n")

    from collections import Counter
    counts = Counter(page["label"] for page in all_pages)
    for label, n in counts.most_common():
        print(f"  {label:<15} {n}")

    top_label, top_count = counts.most_common(1)[0]
    baseline_accuracy = top_count / len(all_pages)
    print(f"\nAlways guessing '{top_label}' would score {baseline_accuracy:.0%}.")
    print("Your trained model has to beat that number to be worth anything.")
