import wikipedia
from sphinx.util import url_re


class WikipediaContent:
    def __init__(self, topic: str):
        self.page = wikipedia.page(topic)

    def url(self) -> str:
        return self.page.url

    def html(self) -> str:
        return self.page.html()

class FakeWikipediaContent:
    def __init__(self, topic: str):
        if topic == "List of mountains by elevation":
            self.url_ = "https://en.wikipedia.org/wiki/List_of_mountains_by_elevation"
            with open("mountains.html", "rb") as ifp:
                self.html_ = ifp.read().decode("utf-8")

    def url(self) -> str:
        return self.url_

    def html(self) -> str:
        return self.html_

if __name__ == '__main__':
    real_or_fake = "Fake"

    if real_or_fake == "Real":
        wiki = WikipediaContent("List of mountains by elevation")
        should_write = True
    else:
        wiki = FakeWikipediaContent("List of mountains by elevation")
        should_write = False

    print(wiki.url())
    html = wiki.html()
    print(len(html), " bytes")

    if should_write:
        with open("mountains.html", "wb") as ofp:
            ofp.write(str.encode(html))
            print("mountains.html written out")
