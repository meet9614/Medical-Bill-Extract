# What's actually in the 15 training samples

Findings from inspecting every source file. Several of these change the design,
and two of them would have invalidated the benchmark if left alone.

The uploaded PDFs are byte-identical (MD5) to the contents of `attachments.zip`,
so this describes the whole corpus.

---

## 1. Two "samples" are the same document — **fixed**

`train_sample_9` and `train_sample_10` are contiguous slices of one 90-page bill:

```
train_sample_10.pdf : "Page 1 of 90" .. "Page 3 of 90"
train_sample_9.pdf  : "Page 4 of 90" .. "Page 6 of 90"
both: Bill No INT2043376 | IP No AMHLIP398580 | UHID AMHL.0002165665
```

The original split put sample_9 in test and sample_10 in train — leaking the
same document, hospital template, patient, and scanner across the boundary.

Fixed via `config.DOCUMENT_GROUPS`, which the splitter now treats as one unit.
The split moved from 28/22 to **32 train / 18 held-out pages, 10 / 4 documents**,
verified to have zero group overlap.

`vlm/data/find_duplicates.py` re-runs this check. **It can only see 3 of 15
files** — the other 12 have no text layer. Undetected pairs among the scanned
files are the largest remaining threat to the held-out numbers, and clearing
that needs a human pass over the letterheads.

## 2. The corpus is 80% scanned images, not digital PDFs

| | files | pages |
|---|---|---|
| scanned, image-only | 12 | 41 |
| digital text layer | 3 | 9 |

This vindicates the multimodal approach — `extractor.py`'s `_text_pdf_fallback`
is dead code on 12 of 15 files. It also means `train_sample_7`'s "text layer" is
junk OCR output, not real text: 1,244 chars/page of garbage like
`_ ;t'FG .. __:Jl?tr.h`. Anything that trusts a text layer's *presence* as a
proxy for its *quality* will be wrong here.

## 3. Class distribution is badly imbalanced — **baseline added**

From inspection, the overwhelming majority of the 50 pages are Bill Detail
(itemised service tables). Bill Summary appears roughly 2–4 times, and the
handwritten pharmacy memos account for most of the rest.

At that skew, **a model that always predicts "Bill Detail" scores somewhere
around 60–75%.** An 85% accuracy result would then be nearly worthless, and
quoting it without the baseline would be the single most attackable claim in
the writeup.

`benchmark.py` now computes and reports a majority-class baseline row
automatically, with the note that any backend at or below it has learned
nothing.

## 4. The 5-class taxonomy doesn't cleanly fit the data

Concrete cases where `PAGE_TYPES` breaks down:

- **Hybrid summary/detail.** `train_sample_1` ("DETAIL FINAL BILL") and
  `train_sample_14` ("IP FINAL CREDIT BILL - BREAKUP") carry per-category
  subtotals *and* individual line items on the same page. Both "Bill Summary"
  and "Bill Detail" are defensible labels. Label noise here is structural, not
  annotator sloppiness — and it will show up as an irreducible error floor for
  both Gemini and the fine-tune.
- **Two bills on one page.** `train_sample_11` is a photograph of *two* separate
  pharmacy cash bills (No. 383 and No. 257), each with its own total. One
  `page_type` and one flat `bill_items` list cannot represent this correctly.
- **Not a hospital bill at all.** `train_sample_5` is an orthopaedic *implant
  supplier invoice* (MODULAR CUP 43MM, UNCEMENTED FEMORAL STEM 10MM) with a GST
  breakdown. It falls into "Other" by elimination rather than by fit.
- **Mid-document excerpts.** `train_sample_12` starts at "Page 5 of 33";
  `train_sample_15` at "Page 1 of 21"; `train_sample_13` is 5 of 7. Pages begin
  mid-table with no header, so there is often no visual cue for page type at all.

Worth deciding before labelling, because the review CSV bakes it in.

## 5. Hardest pages are handwritten, rotated, and photographed

`train_sample_3` is a handwritten pharmacy invoice, **rotated 90°**, shot on a
desk, with drug names like "Lezinate-MF" in cursive. `train_sample_8` and
`train_sample_11` are handwritten bills photographed on a *textured cloth
background*, skewed, with bleed-through.

Prediction worth writing down before you train, so it can be checked after:
**this is where the 2B model will lose to Gemini, and it will lose badly.**
Handwriting recognition is exactly the capability that scales with parameter
count. If the aggregate gap turns out to be driven almost entirely by the
handwritten pharmacy pages, that is a far more interesting result than a single
averaged accuracy number — it says "use the local model for the 80% of printed
pages, route handwriting to the API." Segment the results that way.

Practical consequence: add rotation to the augmentation set, and do **not**
assume upright input at inference.

## 6. Redaction looks exactly like the fraud signal you're prompting for

Every file is anonymised with **white boxes painted over patient names, bill
numbers and hospital letterheads**. `extractor.py`'s system prompt says:

> Fraud indicators to flag: mismatched fonts, **white-out / whitener over text**, ...

So the fraud detector will fire on essentially every page in the corpus, and
every flag will be a false positive caused by the anonymisation pipeline. On
real un-redacted bills the same rule may be fine — but you cannot demo or
evaluate it on this data, and a reviewer who spots this will assume the feature
was never tested. Either scope the rule to *handwriting-over-print* and
*rate x qty mismatch*, or drop the white-out clause and say why.

## 7. Line-item duplication is real and must not be "cleaned"

`train_sample_9` has 12+ consecutive identical rows (`IP CONSULTATION CHARGES`,
qty 1.00, ₹1,000.00). `train_sample_12` has ~20 identical
`BLOOD SUGAR BY GLUCOMETER` rows. These are legitimate repeated charges, not
duplicates.

Any dedup heuristic that collapses identical rows will silently delete tens of
thousands of rupees. The current `_deduplicate` only suppresses Summary pages,
so it's safe — worth keeping it that way, and worth stating explicitly in the
prompt so the fine-tuned model doesn't learn to compress runs.

---

## Handling note

These are real Indian hospital bills with patient names, addresses, phone
numbers and UHID/IP numbers — redacted, but incompletely (`train_sample_9` and
`train_sample_10` still carry a full name, address and mobile number in the text
layer).

Two things follow. First, the distillation step ships these pages to a
third-party API; that's presumably sanctioned for the datathon, but it is worth
knowing you're doing it. Second, this is the concrete version of the on-device
argument — *"the pages I was given still contained PHI in the text layer, which
is exactly why I wanted a model that doesn't need to send them anywhere."*
That's a better interview answer than a cost multiplier.

Also: `.env` with a live `GOOGLE_API_KEY` is sitting in the repo root. It's
gitignored, but confirm it never entered git history before this goes public.
