import json
import os
import re
from collections import defaultdict
from typing import List
from urllib.parse import unquote

import pkg_resources

DATA_DIR_PATH = pkg_resources.resource_filename("nle", "minihack/dat")


class NethackWiki:
    """A class representing Nethack Wiki Data - pages and links between them."""

    def __init__(
        self,
        raw_wiki_file_name: str = f"{DATA_DIR_PATH}/nethackwikidata.json",
        processed_wiki_file_name: str = f"{DATA_DIR_PATH}/processednethackwiki.json",
        save_processed_json: bool = True,
        ignore_inpage_anchors: bool = True,
    ) -> None:
        if os.path.isfile(processed_wiki_file_name):
            with open(processed_wiki_file_name, "r") as json_file:
                self.wiki = json.load(json_file)
        elif os.path.isfile(raw_wiki_file_name):
            raw_json = load_json(raw_wiki_file_name)
            self.wiki = process_json(
                raw_json, ignore_inpage_anchors=ignore_inpage_anchors
            )
            if save_processed_json:
                with open(processed_wiki_file_name, "w+") as json_file:
                    json.dump(self.wiki, json_file)
        else:
            raise ValueError(
                """One of `raw_wiki_file_name` or `processed_wiki_file_name`
                must be supplied as argument and be a file. Try using
                `nle/minihack/scripts/get_nhwiki_data.sh` to download the
                data."""
            )

    def get_page_text(self, page: str) -> str:
        return self.wiki.get(page, {}).get("text", "")

    def get_page_data(self, page: str) -> dict:
        return self.wiki.get(page, {})


def load_json(file_name: str) -> list:
    """Load a file containing a json object per line into a list of dicts."""
    with open(file_name, "r") as json_file:
        input_json = []
        for line in json_file:
            input_json.append(json.loads(line))
    return input_json


def process_json(wiki_json: List[dict], ignore_inpage_anchors) -> dict:
    """Process a list of json pages of the wiki into one dict of all pages."""
    result: dict = {}
    redirects = {}
    result["_global_counts"] = defaultdict(int)

    def href_normalise(x: str):
        result = unquote(x.lower())
        if ignore_inpage_anchors:
            result = result.split("#")[0]
        return result.replace("_", " ")

    for page in wiki_json:
        relevant_page_info = dict(
            title=page["wikipedia_title"].lower(),
            length=len("".join(page["text"])),
            categories=page["categories"].split(","),
            raw_text="".join(page["text"]),
            text=clean_page_text(page["page_data"]),
        )
        # breakpoint()
        # noqa: E731
        relevant_page_info["anchors"] = [
            dict(
                text=anchor["text"].lower(),
                page=href_normalise(anchor.get("title", anchor.get("href"))),
                start=anchor["start"],
            )
            for anchor in page["anchors"]
        ]
        redirect_anchors = [
            anchor
            for anchor in page["anchors"]
            if anchor.get("title")
            and href_normalise(anchor["href"]) != href_normalise(anchor["title"])
        ]
        redirects.update(
            {
                href_normalise(anchor["href"]): href_normalise(anchor["title"])
                for anchor in redirect_anchors
            }
        )
        unique_anchors: dict = defaultdict(int)
        for anchor in relevant_page_info["anchors"]:
            unique_anchors[anchor["page"]] += 1
            result["_global_counts"][anchor["page"]] += 1
        relevant_page_info["unique_anchors"] = dict(unique_anchors)
        result[relevant_page_info["title"]] = relevant_page_info
    for alias, page in redirects.items():
        result[alias] = result[page]
    return result


def clean_page_text(text: List[str]) -> str:
    """Clean Markdown text to make it more passable into an NLP model.

    This is currently very basic, and more advanced parsing could be employed
    if necessary."""

    return re.sub(r"[^a-zA-Z0-9_\s\.]", "", ",".join(text))
