# Valliappa Lakshmanan — personal website

Classic navy home page, built as plain HTML and CSS for GitHub Pages. No build step, package dependencies, JavaScript, or external fonts.

## Preview

From the repository root:

```sh
python3 -m http.server 8000 --directory docs
```

Visit http://localhost:8000.

## Publish to GitHub Pages

Commit and push `docs/`. In GitHub repository **Settings → Pages**, choose **Deploy from a branch**, select the branch containing these files, and choose **/docs**. The default project URL is https://lakshmanok.github.io/lakblogs/.

Only publish `docs/`, not the repository root. Relative assets support the project URL and a custom domain. `.nojekyll` disables Jekyll processing.

## Content and maintenance

- `index.html`: approved first-person home page copy; LinkedIn is the primary contact action.
- `assets/styles.css`: responsive Classic navy design.
- Portrait and both book covers are stored locally, sourced from the original Weebly site.
- Biography, Books, and Speaking link to the existing Weebly pages until those pages are migrated. Courses, OSS, and bridge are intentionally omitted from the home page.
- Google Cloud title: Director of Data Analytics and AI Solutions.
- Fellowship selectivity is annual: new Fellows represent no more than 0.2% of AMS members each year. The text links to the AMS criteria.

Sources: original home page, résumé and books page at https://aisoftwarellc.weebly.com/; https://www.obin.ai/; AMS Fellowship criteria linked in the page.

The original portrait is 118 × 159 pixels, displayed at its native size on desktop. A higher-resolution original would improve high-density display sharpness.

## Books page

`books.html` retains all 11 entries from the original Books page, including both editions of Data Science on the Google Cloud Platform. The four topics are Generative AI; Machine Learning; Data Platforms & Governance; and Climate & Geospatial Science. Entries are ordered by descending publication year within each topic, with equal-year entries retaining editorial order. All purchase links point to Amazon.

The homepage Books links now point to `books.html`. Biography and Speaking remain on Weebly.

Machine Learning Design Patterns uses O’Reilly’s publication date (October 2020), correcting the original site's 2021 copyright-year listing: https://www.oreilly.com/library/view/machine-learning-design/9781098115777/.

The climate proceedings' hardcover ISBN is 9783319172194 (Amazon ISBN-10 product identifier 3319172190), as confirmed by Springer: https://link.springer.com/book/10.1007/978-3-319-17220-0.

## Articles page

`articles.html` has a featured section and the full newest-first archive from the original index. Edit `articles.json` and run `python3 scripts/build_articles.py` from the repository root to regenerate it. See [EDITING-ARTICLES.md](EDITING-ARTICLES.md) for a copy-and-paste example and featuring instructions. The page is static; regeneration is needed only after editing article data.

## Speaker bio

`speaker-bio.html` presents the approved conversational introduction first, short bio second, and full bio in a collapsed native disclosure below. The two concise options have prominent copy buttons. Headshots link to the original gallery. All biographies remain readable without JavaScript; the full-bio disclosure also works without it.

To change wording, edit the paragraphs inside `conversational-bio`, `short-bio`, or `full-bio`. Copy buttons read those same paragraphs, so there is no duplicate copy to maintain. `assets/speaker-bio.js` uses the browser clipboard on HTTPS or localhost; if unavailable or denied, it selects the text for manual copying and explains the fallback.

## Talks & videos

`talks.html` groups all 37 recordings into four topics. Each topic shows up to three selected thumbnails, with its full chronological list in a collapsed “More videos” disclosure. Career & Conversations has two recordings. Dates sort newest first within each topic; unknown dates appear last. Thumbnails are lazy-loaded local assets and links open the original YouTube/Vimeo video without embedding a player. Edit `talks.json` and run `python3 scripts/build_talks.py`; see [EDITING-TALKS.md](EDITING-TALKS.md).

## Teaching & Education

`courses.html` features the Advanced Solutions Lab, four Google Cloud programs, and two University of Oklahoma courses. Edit the HTML directly. Links are included for the verified Advanced Solutions Lab, advanced ML, and data engineering program pages. The original introductory ML and data-to-insights Coursera links returned 404; the university material links could not be retrieved, so those entries currently have no resource links.

## Research page

`research.html` presents selected research contributions, AMS recognition, and all 53 entries from the original publication archive. Citations and summaries expand individually. Edit `research.json` and run `python3 scripts/build_research.py`; see [EDITING-RESEARCH.md](EDITING-RESEARCH.md).

## Biography page

`biography.html` contains the first-person career narrative, career conversation video, advisory roles, selected recognition, education, and expandable earlier experience and additional recognition. Edit the HTML directly. The résumé PDF link points to the existing file on Weebly. AMS Fellow leads Selected recognition; the University of Oklahoma Innovator Award is in Additional recognition.
