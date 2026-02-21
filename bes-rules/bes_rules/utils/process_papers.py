import asyncio
import json

from poe_api_wrapper import AsyncPoeApi, PoeApi
from datetime import datetime

import feedparser
from pydantic import BaseModel


class Publication(BaseModel):
    title: str
    authors: str
    date: datetime

    def __str__(self):
        return f"{self.title}, {self.authors}, published: {self.date}"

    def get_info_for_gpt(self):
        return f"Title: {self.title}; Authors: {self.authors}"


def poe_tests():
    with open(r"J:\poe-tokens.json", "r") as file:
        tokens = json.load(file)

    def main_chat():
        client = PoeApi(tokens=tokens)
        print(client.get_chat_history())

    async def main():
        client = await AsyncPoeApi(tokens=tokens).create()
        print(client.get_chat_history())
        message = "Do you know Jonathan Kriwet"
        async for chunk in client.send_message(bot="claude_3_sonnet_200k", message=message):
            print(chunk["response"], end='', flush=True)
    main_chat()
    #asyncio.run(main())


def parse_rss_feed(url):
    feed = feedparser.parse(url)
    publications = []
    for entry in feed.entries:
        date_string = entry.summary.split("Publication date: ")[-1].split("<")[0]
        date_string = date_string.replace("Available online ", "")
        authors = entry.summary.split("Author(s): ")[-1].split("<")[0]
        date = datetime.strptime(date_string, "%d %B %Y")
        publications.append(Publication(date=date, title=entry.title, authors=authors))
    return publications


def get_latest_rss_feeds():
    journals = {
        "Applied Energy": "https://rss.sciencedirect.com/publication/science/03062619",
        "Energy and Buildings": "https://rss.sciencedirect.com/publication/science/03787788",
    }
    all_publications = []
    for journal, url in journals.items():
        publications = parse_rss_feed(url)
        print(f"Found {len(publications)} publications for journal {journal}")
        all_publications.extend(publications)
    return all_publications


def get_tokens():
    with open(r"J:\poe-tokens.json", "r") as file:
        return json.load(file)


def get_relevant_publications(publications: list):
    if len(publications) == 0:
        return []

    async def main(_publications):
        client = await AsyncPoeApi(tokens=get_tokens()).create()
        chunk_size = 10
        i = 0
        relevant_publications = []
        while True:
            if i + chunk_size >= len(_publications):
                end = len(_publications)
            else:
                end = i + chunk_size
            message = "\n".join(["- " + publication.get_info_for_gpt() for publication in _publications[i:end]])
            print(i, "\n", message)
            response = ""
            #async for chunk in client.send_message(bot="relevantliterature", message=message):
            #    response += chunk["response"]
            #    print(chunk["response"], end='', flush=True)
            relevant_publications.append(response)
            if end == len(_publications):
                break
            i += chunk_size
    asyncio.run(main(publications))


if __name__ == '__main__':
    publications = get_latest_rss_feeds()
    get_relevant_publications(publications)
