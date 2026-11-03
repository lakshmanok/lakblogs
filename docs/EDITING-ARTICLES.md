# Adding and featuring articles

Edit **docs/articles.json**. It is the single source for both the featured section and the full archive. No HTML or CSS edits are needed.

## Add a new article

Copy this object to the **top of the list**, immediately after the opening `[`. Leave a comma between it and the next article:

```json
{
  "title": "Your new article title",
  "url": "https://medium.com/@lakshmanok/your-article",
  "featured": false
},
```

The list order is the display order: keep it newest first. To add an older article, insert it in the appropriate position. Existing entries retain their order from the original site; publication dates were not supplied there and have not been invented.

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
