# Editing the Research page

Edit the opening, contributions, and recognition directly in `research.html`, outside the generated markers.

Edit publication entries in `research.json`, then run from the repository root:

```sh
python3 scripts/build_research.py
python3 scripts/build_research.py --check
```

Each entry contains a stable `id`, `category` (`articles` or `chapters`), integer `year`, `title`, primary `url` (empty if none), `citation_and_summary`, and a `links` array of objects with `title` and `url`. Keep IDs stable because selected contributions link to them. Entries sort by descending year within their category; ties retain JSON order. Update the archive introduction and navigation counts if adding or removing entries.

The migration retains all 49 articles/reports and four book chapters from the original Research page, including their original citation text, summaries, and link destinations. Whitespace and title punctuation have been normalized. Publication years are those listed on the original page; bibliographic metadata has not been independently corrected. Two entries (WDSS-II and Multiscale storm identification and forecast) had no paper links on the original page and remain unlinked. Legacy external destinations have been preserved, not all individually verified.

Sources:
- https://aisoftwarellc.weebly.com/research.html
- https://aisoftwarellc.weebly.com/resumevitae.html
- https://www.ametsoc.org/AMs/about-ams/ams-awards-honors/fellows/
