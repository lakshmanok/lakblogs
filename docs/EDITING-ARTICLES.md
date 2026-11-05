# Adding and featuring articles

Edit **docs/articles.json**. It is the single source for article entries in both the featured section and the full archive. Yearly subject summaries live in **docs/article-years.json**. No HTML or CSS edits are needed.

## Add a new article

Copy this object to the **top of the list**, immediately after the opening `[`. Leave a comma between it and the next article:

```json
{
  "title": "Your new article title",
  "url": "https://medium.com/@lakshmanok/your-article",
  "published_year": 2026,
  "published_month": 11,
  "featured": false
},
```

The list order is the display order: keep it newest first. To add an older article, insert it in the appropriate position. The builder preserves this order, groups articles by `published_year`, and continues descending numbering across year sections down to 1. It rejects years placed out of order. `published_month` is an integer from 1 to 12, or `null` when not verified; do not invent a month. Saved metadata makes builds entirely offline.

## Choose featured articles

Set `"featured": true` for any article you want at the top. Set it to `false` to remove it from Featured without removing it from the archive. Featured articles follow the same newest-first order as the main list. Four to six featured articles works well. No separate list to maintain.

## Rebuild and preview

From the repository root, run:

```sh
python3 scripts/build_articles.py
python3 -m http.server 8000 --directory docs
```

Open http://localhost:8000/articles.html. Commit both `docs/articles.json` and the rebuilt `docs/articles.html` when ready to publish.

To check that the page matches the data without writing:

```sh
python3 scripts/build_articles.py --check
```

Python 3.9+ is sufficient; there are no packages to install. GitHub Pages serves the generated HTML directly. The live page does not need JavaScript, Python, or network calls to load its list.

## JSON editing tips

Use double quotes around titles and URLs. Escape a quote inside a title as `\"`. Use lowercase `true` and `false`. Separate objects with commas, but do not put a comma after the last object. The builder validates required fields and URLs before changing the page.

All 180 original entries were retained, including any repeated links in the original index. External destinations are preserved; availability of older third-party articles may change.

## Year headings and subjects

Edit `docs/article-years.json` to give each year one to three short subject areas. Add a new year there when adding its first article. Rebuild after changing either JSON file.

## Initial date backfill

Years were assigned using sampled publisher dates at year boundaries and the author's confirmation that the archive is in strict reverse chronological order. Months were retained only where found in publisher dates, dated URLs, or corroborating archival evidence; other months are `null`. Article order and all original links remain unchanged.

Boundary samples (newest / oldest articles around each transition):

- 2026 / 2025: Vertical AI platform (January 2026); Private Equity (December 2025).
- 2025 / 2024: Optimizing AI-Generated Content (February 2025); PydanticAI evaluation (December 2024).
- 2024 / 2023: Advice on using LLMs wisely (January 2024); GenAI tips (October 2023).
- 2023 / 2022: Trends in Data and AI (January 2023); LLMs as bullshit artists (December 2022).
- 2022 / 2021: ECMWF forecasts (January 2022); Vertex AI hyperparameter tuning (November 2021).
- 2021 / 2020: Loading CSV with Google Sheets (January 2021); Keras data augmentation (December 2020).
- 2020 / 2019: Reading BigQuery from TensorFlow (February 2020); raster queries, brute force ([December 2019 newsletter](https://www.gcpweekly.com/newsletter/168/), bounded by the next December 2019 article).
- 2019 / 2018: Text classification choices (January 2019); BigQuery precision/recall (December 2018).
- 2018 / 2017: Moving to core tf.estimator (January 2018); sensitive ML datasets (also published in the [December 2017 book](https://www.oreilly.com/library/view/data-science-on/9781491974551/app01.html); article month left unknown).
- 2017 / 2016: Automating project creation (January 2017); repeatable sampling (November 2016).
- 2016 / 2014: NOAA CRADA (January 2016); w2clustertable (July 2014), both dated in their URLs. There are no 2015 entries.
