# What's actually in the 15 training samples

Findings from inspecting every source file in `attachments.zip`. These describe
the data the Gemini pipeline has to handle, and several of them affect how the
extractor should behave or how results should be reported.

---

## The corpus is 80% scanned images, not digital PDFs

| | files | pages |
|---|---|---|
| scanned, image-only | 12 | 41 |
| digital text layer | 3 | 9 |

This is why the multimodal approach is the right one — `_text_pdf_fallback()` in
`extractor.py` is dead code on 12 of 15 files.

It also means a text layer's *presence* is not evidence of its *quality*.
`train_sample_7` reports 1,244 chars/page, but the content is junk from a prior
OCR pass: `_ ;t'FG .. __:Jl?tr.h .!:tL`. Don't branch on "has text layer".

## Two files are the same document

`train_sample_9` and `train_sample_10` are contiguous slices of one 90-page bill:

```
train_sample_10.pdf : "Page 1 of 90" .. "Page 3 of 90"
train_sample_9.pdf  : "Page 4 of 90" .. "Page 6 of 90"
both: Bill No INT2043376 | IP No AMHLIP398580 | UHID AMHL.0002165665
```

Relevant if you ever split this data for evaluation: treating them as
independent documents leaks the same bill, hospital template, patient and
scanner across both sides of the split. Only 3 of 15 files have a text layer, so
similar pairs among the 12 scanned files can't be detected automatically — they
need a human pass over the letterheads.

## Class distribution is badly imbalanced

The overwhelming majority of the 50 pages are Bill Detail (itemised service
tables). Bill Summary appears roughly 2–4 times; handwritten pharmacy memos make
up most of the remainder.

If you ever quote a page-type accuracy figure, quote the majority-class baseline
next to it — always guessing "Bill Detail" scores roughly 60–75% on this data.

## The 5-class taxonomy doesn't cleanly fit

- **Hybrid summary/detail.** `train_sample_1` ("DETAIL FINAL BILL") and
  `train_sample_14` ("IP FINAL CREDIT BILL - BREAKUP") carry per-category
  subtotals *and* individual line items on the same page. Both labels are
  defensible, so some error here is structural rather than a model failure.
- **Two bills on one page.** `train_sample_11` is a photograph of *two* separate
  pharmacy cash bills (No. 383 and No. 257), each with its own total. One
  `page_type` and one flat `bill_items` list can't represent this correctly.
- **Not a hospital bill.** `train_sample_5` is an orthopaedic implant supplier
  invoice (MODULAR CUP 43MM, UNCEMENTED FEMORAL STEM 10MM) with a GST breakdown.
  It lands in "Other" by elimination.
- **Mid-document excerpts.** `train_sample_12` starts at "Page 5 of 33";
  `train_sample_15` at "Page 1 of 21". Pages begin mid-table with no header, so
  there is often no visual cue for page type at all.

## Hardest pages: handwritten, rotated, photographed

`train_sample_3` is a handwritten pharmacy invoice **rotated 90°**, shot on a
desk, with cursive drug names. `train_sample_8` and `train_sample_11` are
handwritten bills photographed on a *textured cloth background*, skewed, with
bleed-through.

`extractor.py` now auto-rotates via Tesseract OSD before both OCR and the API
call. Tesseract remains useless on the handwriting itself — those pages depend
entirely on the multimodal model reading the image.

## Redaction looks exactly like the fraud signal being prompted for

Every file is anonymised with **white boxes painted over patient names, bill
numbers and hospital letterheads**. The system prompt in `extractor.py` says:

> Fraud indicators to flag: mismatched fonts, **white-out / whitener over text**, ...

So the fraud detector fires on essentially every page in this corpus, and every
flag is a false positive caused by the anonymisation pipeline. On real
un-redacted bills the rule may be fine, but it cannot be demonstrated or
evaluated on this data. Either narrow the rule to *handwriting-over-print* and
*rate × quantity mismatch*, or drop the white-out clause.

## Repeated line items are real — never dedupe them

`train_sample_9` has 12+ consecutive identical rows (`IP CONSULTATION CHARGES`,
qty 1.00, ₹1,000.00). `train_sample_12` has ~20 identical
`BLOOD SUGAR BY GLUCOMETER` rows. These are legitimate repeated charges.

Any heuristic that collapses identical rows silently deletes tens of thousands
of rupees. The current `_deduplicate()` only suppresses Bill Summary pages, which
is correct — keep it that way.

---

## OCR findings (measured)

Against `pdftotext` ground truth on the pages that have a usable text layer:

| variant | token F1 | number F1 |
|---|---|---|
| **current** (contrast 2.0 + sharpness 2.0 + median 3, `--psm 6`) | **69.5%** | **80.7%** |
| no preprocessing, psm 6 | 58.9% | 75.1% |
| current preprocessing, psm 3 (default) | 41.9% | 6.2% |

The preprocessing earns ~11 points of token F1, and `--psm 6` over the default
is the difference between 81% and 6% on numeric tokens. Do not "fix" it to auto
page segmentation.

Tesseract confidence **cannot** be used to detect handwriting — measured mean
confidence: handwritten 59.3 / 49.5, printed 65.5 / 86.8 / 88.8 / 74.2. The
distributions overlap, so `OCR_MIN_CONFIDENCE` is set to 35 as a
catastrophic-failure guard only.

Parallel OCR requires `OMP_THREAD_LIMIT=1`. Tesseract 4.x is internally
multi-threaded, so concurrent instances oversubscribe the CPU — measured on 4
cores, 2 pages: serial 2.40s, 2 workers unrestricted **17.42s**, 3 workers with
the limit set 2.1× faster than serial. `extractor.py` sets this automatically.

---

## Handling note

These are real Indian hospital bills with patient names, addresses, phone
numbers and UHID/IP numbers. Redaction is **incomplete** — `train_sample_9` and
`train_sample_10` still carry a full name, address and mobile number in the PDF
text layer.

`attachments.zip` is committed to a public repository. Worth confirming the
datathon terms permit redistribution before the repo goes on a CV. Removing it
would require a history rewrite (`git filter-repo`), not just a delete commit.

Also: `.env` holds a live `GOOGLE_API_KEY`. It is gitignored and has never
entered git history — keep it that way.
