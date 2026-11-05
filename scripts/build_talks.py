#!/usr/bin/env python3
"""Build the static Talks page from docs/talks.json; --check verifies without writing."""
import argparse
import calendar
import datetime
import json
import re
from html import escape as e
from pathlib import Path
from urllib.parse import urlsplit, parse_qs
ROOT = Path(__file__).resolve().parents[1]
TOPICS = [('strategy', 'AI Strategy & Leadership'), ('agents', 'Agentic AI & Design Patterns'), ('platforms', 'Data Platforms & Applied Machine Learning'), ('career', 'Career & Conversations')]
START, END = '<!-- BEGIN GENERATED TALKS -->', '<!-- END GENERATED TALKS -->'


def validate(items):
    if not isinstance(items, list) or not items:
        raise ValueError('talks.json must be a nonempty list.')
    seen = set()
    for item in items:
        if not isinstance(item, dict) or not all(isinstance(item.get(k), str) and item[k].strip() for k in ('title', 'url', 'topic')):
            raise ValueError('Every talk needs title, url and topic strings.')
        if item['topic'] not in dict(TOPICS):
            raise ValueError('Unknown topic: ' + item['topic'])
        url = urlsplit(item['url'])
        if url.scheme != 'https' or url.hostname not in ('www.youtube.com','youtube.com','youtu.be','vimeo.com','www.vimeo.com') or url.username or url.password:
            raise ValueError('Use an HTTPS YouTube or Vimeo URL.')
        if item['url'] in seen:
            raise ValueError('Duplicate video URL: ' + item['url'])
        seen.add(item['url'])
        date = item.get('date')
        if date:
            if not isinstance(date, str) or not re.fullmatch(r'\d{4}(-\d{2}){0,2}', date):
                raise ValueError('Dates must be YYYY, YYYY-MM or YYYY-MM-DD, or null.')
            datetime.date.fromisoformat(date + {4:'-01-01',7:'-01',10:''}[len(date)])
        if not isinstance(item.get('featured',False),bool):
            raise ValueError('featured must be true or false.')
        for key in ('event','thumbnail'):
            if key in item and not isinstance(item[key],str):
                raise ValueError(key + ' must be a string.')
        thumb=item.get('thumbnail','')
        if thumb:
            u=urlsplit(thumb)
            if u.scheme:
                if u.scheme!='https' or not u.hostname:raise ValueError('Thumbnail URL must use HTTPS.')
            elif not (thumb.startswith('assets/') and '..' not in Path(thumb).parts and (ROOT/'docs'/thumb).is_file()):
                raise ValueError('Local thumbnail not found: '+thumb)


def date_label(date):
    if not date:return 'Date not listed'
    parts=date.split('-')
    if len(parts)==1:return date
    return f'{calendar.month_name[int(parts[1])]} {parts[0]}'


def thumbnail(item):
    if item.get('thumbnail'):return item['thumbnail']
    url=urlsplit(item['url'])
    video_id=url.path.strip('/') if url.hostname=='youtu.be' else parse_qs(url.query).get('v',[''])[0]
    if re.fullmatch(r'[\w-]{11}',video_id):return f'https://i.ytimg.com/vi/{video_id}/hqdefault.jpg'
    return 'assets/video-placeholder.svg'


def card(item):
    provider='Vimeo' if 'vimeo.com' in urlsplit(item['url']).hostname else 'YouTube'
    date=item.get('date');label=e(date_label(date))
    stamp=f'<time datetime="{date}">{label}</time>' if date else label
    event=f'<p class="video-event">{e(item.get("event",""))}</p>' if item.get('event') else ''
    return f'''<article class="video-card"><a href="{e(item['url'],quote=True)}" aria-label="Watch {e(item['title'],quote=True)} on {provider}">
<div class="video-image"><img src="{e(thumbnail(item),quote=True)}" alt="" width="480" height="270" loading="lazy" decoding="async"><span class="video-play" aria-hidden="true">▶</span></div>
<p class="video-meta">{stamp} · {provider}</p><h3>{e(item['title'])}</h3>{event}<span class="video-watch">Watch on {provider} <span aria-hidden="true">↗</span></span></a></article>'''


def archive_row(item):
    provider='Vimeo' if 'vimeo.com' in urlsplit(item['url']).hostname else 'YouTube'
    date=item.get('date');label=e(date_label(date))
    stamp=f'<time datetime="{date}">{label}</time>' if date else label
    event=e(item.get('event',''))
    metadata=' · '.join(part for part in (event,provider) if part)
    return f'<li><a href="{e(item["url"],quote=True)}"><span class="talk-date">{stamp}</span><span><span class="talk-list-title">{e(item["title"])}</span><span class="talk-list-meta">{metadata}</span></span><span aria-hidden="true">↗</span></a></li>'


def render(items):
    validate(items)
    ordered=sorted(items,key=lambda v:v.get('date') or '',reverse=True)
    out=['<nav class="topic-nav" aria-label="Talk topics">',*[f'<a href="#{slug}">{e(title)}</a>' for slug,title in TOPICS],'</nav>']
    for slug,title in TOPICS:
        group=[v for v in ordered if v['topic']==slug]
        if not group:continue
        featured=[v for v in group if v.get('featured')]
        if len(featured)>3:raise ValueError('Choose at most three featured videos for '+title)
        selected=featured+[v for v in group if not v.get('featured')][:3-len(featured)]
        selected=sorted(selected,key=lambda v:v.get('date') or '',reverse=True)
        out += [f'<section class="video-topic" id="{slug}" aria-labelledby="{slug}-heading"><h2 id="{slug}-heading">{e(title)}</h2><div class="video-grid">',*[card(v) for v in selected],'</div>',
                f'<details class="more-talks"><summary aria-label="More videos in {e(title)}"><span>More videos <span class="talk-total">· Full list ({len(group)})</span></span><span class="more-indicator" aria-hidden="true">+</span></summary>',
                '<p class="talk-order">All recordings in this topic, newest first.</p><ol class="talk-archive">',*[archive_row(v) for v in group],'</ol></details></section>']
    return '\n'.join(out)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    items=json.loads((ROOT/'docs/talks.json').read_text());path=ROOT/'docs/talks.html';old=path.read_text()
    if old.count(START)!=1 or old.count(END)!=1:raise ValueError('Missing or repeated generated markers.')
    before,rest=old.split(START);_,after=rest.split(END)
    new=before+START+'\n'+render(items)+'\n'+END+after
    if args.check:
        if new!=old:raise SystemExit('Talks page is out of date. Run python3 scripts/build_talks.py')
        print(f'PASS: {len(items)} talks; page matches data.')
    else:path.write_text(new);print(f'Built {len(items)} talks.')
if __name__=='__main__':main()
