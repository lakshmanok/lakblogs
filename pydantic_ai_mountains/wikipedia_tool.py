import wikipedia
from abc import ABC, abstractmethod

class WikipediaContent(ABC):
    @abstractmethod
    def url(self) -> str:
        pass

    @abstractmethod
    def html(self) -> str:
        pass

    def snippet(self, search_term, num_bytes=200):
        content = self.html()
        idx = content.find(search_term)
        if idx >= 0:
            return content[idx-num_bytes//2:idx+num_bytes//2]
        else:
            return None

class OnlineWikipediaContent(WikipediaContent):
    def __init__(self, topic: str):
        print(f"Will query online Wikipedia for information on {topic}")
        self.page = wikipedia.page(topic)

    def url(self) -> str:
        return self.page.url

    def html(self) -> str:
        return self.page.html()

class FakeWikipediaContent(WikipediaContent):
    def __init__(self, topic: str):
        if topic == "List of mountains by elevation":
            print(f"Will used cached Wikipedia information on {topic}")
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
        wiki = OnlineWikipediaContent("List of mountains by elevation")
        with open("mountains.html", "wb") as ofp:
            ofp.write(str.encode(wiki.html()))
            print("mountains.html written out")
    else:
        wiki = FakeWikipediaContent("List of mountains by elevation")

    # try out the wiki object
    print(wiki.url())
    print(len(wiki.html()), " bytes")
    print(wiki.snippet("Mount Robson", 200))
    print(wiki.snippet("Dragontail Peak", 200))
    print(wiki.snippet("Mount Hood", 200))



