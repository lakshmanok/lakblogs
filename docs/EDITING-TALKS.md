# Adding talks and choosing featured videos

Edit `docs/talks.json`, then run `python3 scripts/build_talks.py` from the repository root. Commit the JSON and rebuilt `docs/talks.html` together. No dependencies are needed beyond Python 3.9+.

Add an object like this, separated from adjacent objects by a comma:

```json
{
  "title": "Your talk title",
  "topic": "agents",
  "date": "2026-11",
  "event": "Conference or podcast name",
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "thumbnail": "",
  "featured": true
}
```

Topics: `strategy` (AI Strategy & Leadership), `agents` (Agentic AI & Design Patterns), `platforms` (Data Platforms & Applied Machine Learning), or `career` (Career & Conversations).

Use `YYYY-MM-DD`, `YYYY-MM`, or `YYYY` for dates. Use `null` if unknown. The builder sorts newest first within each topic’s featured row and full list; undated videos go last. Within the same year, records with an unknown month follow records with known months; it does not invent precise dates.

Set `featured` to `true` for up to three videos per topic. These form the visible thumbnail row. If fewer than three are marked, the newest other videos fill the row. The full chronological list (including the featured selections) sits inside “More videos”, collapsed by default. Topics with fewer than three recordings show only the available videos.

For YouTube, leaving `thumbnail` empty uses the standard YouTube thumbnail URL. To avoid a third-party image request, save a thumbnail under `docs/assets/talk-thumbnails/` and put its relative path in `thumbnail`. Vimeo accepts a local image or HTTPS thumbnail URL; without one it uses the neutral video placeholder. All migrated thumbnails are local. Two original thumbnail downloads failed (Data Canteen and How all your data leads to Cloud Machine Learning); those entries use a neutral placeholder while retaining their video links.

Preview with `python3 -m http.server 8000 --directory docs`, then open http://localhost:8000/talks.html. Check consistency without writing with `python3 scripts/build_talks.py --check`.

## Migration notes

All 37 video links from the original Talks page are preserved. Dates and event names are taken from that page, except the Actian/Vimeo entry whose title and June 2021 upload date were verified using Vimeo metadata. One Cloud N Clear recording has no supplied date. External videos may become unavailable; migrated links are not a guarantee of current playback availability.
