import chromadb
import bs4
import requests
import shutil
import os
import re
from langchain.text_splitter import SentenceTransformersTokenTextSplitter

URL = "https://www.bridgeworld.com/pages/readingroom/bws/bwscompletesystem.html"
URL_LOCAL = URL.split('/')[-1]
CHROMA_COLLECTION_NAME = "bridge_world_system"
CHROMADB_DIR = "db/"


def download_file(url: str) -> str:
    local_filename = URL_LOCAL
    if not os.path.exists(local_filename):
        print(f"Downloading {URL} to {local_filename}.")
        with requests.get(url, stream=True) as r:
            with open(local_filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    else:
        print(f"Using already downloaded {local_filename}.")
    return local_filename


if __name__ == '__main__':
    # create chromadb collection
    chroma_client = chromadb.PersistentClient(path=CHROMADB_DIR)
    collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    text_splitter = SentenceTransformersTokenTextSplitter()

    # update paragraphs into chromadb collection
    download_file(URL)
    with open(URL_LOCAL, 'r') as f:
        soup = bs4.BeautifulSoup(f.read(), 'html.parser')
        last_header = ""
        paragraphs = soup.find_all("p")
        for n, paragraph in enumerate(paragraphs):
            paragraph_id = f"{URL_LOCAL}_{n}"
            text = paragraph.text.strip()
            # find the previous header
            header = paragraph.find_all(re.compile("^h[1-5]$"))
            if header:
                header = header[0].text.strip()
                last_header = header
            else:
                header = last_header
            # print(paragraph_id, "->", header, "->", len(text), "->", text[:30])

            # split the text into chunks and insert into chromadb
            ids = []
            documents = []
            metadatas = []
            chunks = text_splitter.create_documents([text]) # takes array of documents
            for chunk_no, chunk in enumerate(chunks):
                ids.append(f"{paragraph_id}#{chunk_no}")
                documents.append(chunk.page_content)
                metadatas.append({"title": header, "source": URL})
            if ids:
                collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"{int(0.5 + 100.0 * n / len(paragraphs))}% ({collection.count()})", end=" ", flush=True)
            if n % 10 == 0:
                print()
