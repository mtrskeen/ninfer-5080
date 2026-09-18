#!/usr/bin/env python3
"""Build NIAH chat fixtures at arbitrary approximate token lengths."""
import json
from pathlib import Path

REPO = Path("/repo")
SRC = REPO / "examples/cli/messages/long_niah_8k.json"
OUT = Path("/fixtures")
CHARS_PER_TOKEN = 32338 / 7680


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = json.loads(SRC.read_text())
    src = src["messages"] if isinstance(src, dict) else src
    system, user = src[0]["content"], src[1]["content"]
    prefix = user.split("<document>", 1)[0]
    body = user.split("<document>", 1)[1].rsplit("</document>", 1)[0]
    tail = user.rsplit("</document>", 1)[1]
    needle = ("\n\nOFFICIAL RECORD: The recovery code for the ORCHID relay is 493817, "
              "and its registered status color is COBALT.\n\n")
    body_clean = body.replace(needle, "")
    body_len = len(body_clean)
    for tokens in (32768, 98304, 123392):
        chars = int(tokens * CHARS_PER_TOKEN)
        repeats = max(1, chars // body_len + 1)
        filled = (body_clean * repeats)[:chars]
        mid = len(filled) // 2
        doc = filled[:mid] + needle + filled[mid:]
        fixture = [
            {"role": "system", "content": system},
            {"role": "user", "content": prefix + "<document>\n" + doc + "</document>\n\n" + tail.lstrip("\n")},
        ]
        path = OUT / f"niah_{tokens}.json"
        path.write_text(json.dumps(fixture, ensure_ascii=False))
        print(f"wrote {path} approx_tokens={tokens} chars={len(doc)}")


if __name__ == "__main__":
    main()
