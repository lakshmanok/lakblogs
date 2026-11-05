#!/usr/bin/env python3
"""Rebuild the static publication archive from docs/research.json."""
import argparse
import json
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEGIN = '<!-- BEGIN GENERATED RESEARCH -->'
END = '<!-- END GENERATED RESEARCH -->'

def render(entries):
    ids = set()
    for entry in entries:
        assert entry['id'] not in ids, 'Duplicate publication ID'
        ids.add(entry['id'])
        assert entry['category'] in ('articles', 'chapters')
        assert isinstance(entry['year'], int) and 1900 <= entry['year'] <= 2100
        for url in [entry['url']] + [link['url'] for link in entry['links']]:
            assert not url or url.startswith(('https://', 'http://')), 'Invalid URL'
    output = []
    for category, anchor, heading in [('articles', 'journal-articles', 'Articles & reports'), ('chapters', 'book-chapters', 'Book chapters')]:
        output.append(f'<section class="publication-section" aria-labelledby="{anchor}"><h2 id="{anchor}">{escape(heading)}</h2><ol class="publication-list">')
        for e in sorted((e for e in entries if e['category'] == category), key=lambda e: -e['year']):
            title = escape(e['title'])
            if e['url']:
                title = f'<a href="{escape(e["url"], quote=True)}">{title}</a>'
            output.append(f'<li class="publication" id="{escape(e["id"], quote=True)}"><p class="catalog-meta">{e["year"]}</p><h3>{title}</h3><details><summary>Full citation &amp; summary</summary><p class="citation">{escape(e["citation_and_summary"])}</p>')
            seen = set()
            links = []
            for link in e['links']:
                if link['url'] in seen: continue
                seen.add(link['url'])
                label = link['title'] if link['title'].startswith('[link') else 'Read paper / source'
                links.append(f'<a href="{escape(link["url"], quote=True)}">{escape(label.strip("[]"))} ↗</a>')
            if links: output.append('<div class="source-links">' + ''.join(links) + '</div>')
            output.append('</details></li>')
        output.append('</ol></section>')
    return '\n'.join(output)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    entries = json.loads((ROOT/'docs/research.json').read_text())
    path = ROOT/'docs/research.html'
    current = path.read_text()
    before, rest = current.split(BEGIN)
    _, after = rest.split(END)
    result = before + BEGIN + '\n' + render(entries) + '\n' + END + after
    if args.check:
        assert result == current, 'Page differs from research.json; rebuild it.'
        print(f'PASS: {len(entries)} publications; page matches data.')
    else:
        path.write_text(result)
        print(f'Built {len(entries)} publications.')

if __name__ == '__main__': main()
