#!/usr/bin/env python3
"""Render docs/articles.json into the marked sections of docs/articles.html.

Run from any directory: python3 /path/to/lakblogs/scripts/build_articles.py
Use --check to verify the committed page matches its data without writing.
"""
import argparse
import json
from itertools import groupby
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'docs/articles.json'
PAGE = ROOT / 'docs/articles.html'
SUBJECTS = ROOT / 'docs/article-years.json'
START = '<!-- BEGIN GENERATED ARTICLES -->'
END = '<!-- END GENERATED ARTICLES -->'


def render(articles, subjects):
    if not isinstance(articles, list) or not articles:
        raise ValueError('articles.json must contain a nonempty list of articles.')
    for i, article in enumerate(articles, 1):
        if not isinstance(article, dict):
            raise ValueError(f'Article {i} must be an object.')
        for field in ('title', 'url'):
            if not isinstance(article.get(field), str) or not article[field].strip():
                raise ValueError(f'Article {i} needs a nonempty {field}.')
        parsed = urlsplit(article['url'])
        if parsed.scheme not in ('https', 'http') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f'Article {i} needs a public HTTP(S) URL.')
        if not isinstance(article.get('featured', False), bool):
            raise ValueError(f'Article {i}: featured must be true or false.')
        year, month = article.get('published_year'), article.get('published_month')
        if type(year) is not int or not 1900 <= year <= 2099:
            raise ValueError(f'Article {i}: published_year must be a four-digit year.')
        if month is not None and (type(month) is not int or not 1 <= month <= 12):
            raise ValueError(f'Article {i}: published_month must be 1–12 or null.')
    years = [a['published_year'] for a in articles]
    if years != sorted(years, reverse=True):
        raise ValueError('Keep articles in newest-first year order.')
    featured = [a for a in articles if a.get('featured', False)]
    lines = []
    if featured:
        lines += ['<section class="featured-articles" aria-labelledby="featured-heading">',
                  '<h2 id="featured-heading">Featured</h2>', '<div class="featured-grid">']
        for a in featured:
            title, url = escape(a['title']), escape(a['url'], quote=True)
            host = escape(urlsplit(a['url']).hostname.removeprefix('www.'))
            lines += [f'<article class="featured-article"><p class="article-source">{host}</p>',
                      f'<h3><a href="{url}">{title}<span aria-hidden="true"> ↗</span></a></h3></article>']
        lines += ['</div></section>']
    lines += ['<section class="article-archive" id="all-articles" aria-labelledby="archive-heading">',
              '<div class="archive-heading"><h2 id="archive-heading">All articles</h2>',
              f'<p>{len(articles)} articles · Newest first</p></div>']
    number = len(articles)
    for year, entries in groupby(articles, key=lambda a: str(a['published_year'])):
        entries = list(entries)
        topics = subjects.get(year, [])
        if not isinstance(topics, list) or not 1 <= len(topics) <= 3 or any(not isinstance(t, str) or not t.strip() for t in topics):
            raise ValueError(f'{year}: supply one to three subject areas in article-years.json.')
        heading = year
        lines += [f'<section class="article-year" aria-labelledby="year-{year}">',
                  '<header class="article-year-heading">',
                  f'<h3 id="year-{year}">{heading}</h3>',
                  '<p class="article-year-subjects">Topics: ' + ', '.join(escape(topic) for topic in topics) + '.</p>']
        lines += ['</header>', f'<ol class="article-list" reversed start="{number}">']
        for a in entries:
            title, url = escape(a['title']), escape(a['url'], quote=True)
            host = escape(urlsplit(a['url']).hostname.removeprefix('www.'))
            lines += [f'<li><a href="{url}"><span class="article-title">{title}</span>',
                      f'<span class="article-source">{host} <span aria-hidden="true">↗</span></span></a></li>']
        number -= len(entries)
        lines += ['</ol></section>']
    lines += ['</section>']
    return '\n'.join(lines)


def build():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Check without changing the page')
    args = parser.parse_args()
    articles = json.loads(DATA.read_text(encoding='utf-8'))
    original = PAGE.read_text(encoding='utf-8')
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError('articles.html must contain one pair of generated-section markers.')
    before, rest = original.split(START)
    _, after = rest.split(END)
    updated = before + START + '\n' + render(articles, json.loads(SUBJECTS.read_text(encoding='utf-8'))) + '\n' + END + after
    if args.check:
        if updated != original:
            raise SystemExit('Articles page is out of date. Run python3 scripts/build_articles.py')
        print(f'PASS: {len(articles)} articles; generated page matches articles.json.')
    else:
        PAGE.write_text(updated, encoding='utf-8')
        print(f'Built {PAGE} with {len(articles)} articles.')


if __name__ == '__main__':
    build()
