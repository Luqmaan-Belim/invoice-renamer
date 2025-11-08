# Code Review: `process_invoices.py`

## 1. OCR confidence averaging always zero
In `ocr_with_confidence`, the confidence values returned by Tesseract are string
representations of numbers. The current comprehension only accepts numeric
instances (`int`/`float`), so it drops every value, which forces `avg` to be `0`
and triggers the "low confidence" retry logic for every document. This turns the
adaptive preprocessing into the default path and makes the confidence metric
meaningless. Convert the confidence entries with `float(c)` (skipping `-1`) before
computing the average.

## 2. Minor cleanups
* `download_first_page` initialises `downloader = next_chunk = None` but `next_chunk`
is unused afterwards. Drop the assignment to avoid confusion.
* `unique_name_in_folder` always splits the original `base` instead of the
current `name` when adding suffixes. This works for the current pattern but would
break for names that already include a suffix (e.g. `NNNN_000001_2.pdf`). Split
the evolving `name` instead to keep the logic robust if the function is reused.
