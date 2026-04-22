#!/usr/bin/env python3

from __future__ import annotations

import argparse
import cgi
import sys
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from tqdm import tqdm


DEFAULT_URL = "https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0"
CHUNK_SIZE = 1024 * 1024


def force_dropbox_download(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["dl"] = "1"
    return urlunparse(parsed._replace(query=urlencode(query)))


def infer_filename(response: requests.Response, output: Optional[str]) -> Path:
    if output:
        return Path(output).expanduser()

    content_disposition = response.headers.get("content-disposition", "")
    _, params = cgi.parse_header(content_disposition)
    filename = params.get("filename") or params.get("filename*")

    if filename and "''" in filename:
        filename = filename.split("''", 1)[1]

    if not filename:
        final_path = Path(urlparse(response.url).path)
        filename = final_path.name or "dropbox_download.zip"

    return Path(filename).expanduser()


def download(url: str, output: Optional[str]) -> Path:
    final_url = force_dropbox_download(url)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
        }
    )

    with session.get(final_url, stream=True, allow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        target_path = infer_filename(response, output)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        total_size = int(response.headers.get("content-length", 0))
        desc = target_path.name

        with open(target_path, "wb") as file_obj, tqdm(
            total=total_size if total_size > 0 else None,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
        ) as progress:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                written = file_obj.write(chunk)
                progress.update(written)

    return target_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a Dropbox shared file/folder with a tqdm progress bar."
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=DEFAULT_URL,
        help="Dropbox shared link. Defaults to the link provided in the request.",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file path. If omitted, the script infers the filename from Dropbox.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        saved_path = download(args.url, args.output)
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved to: {saved_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
