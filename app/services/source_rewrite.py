"""Conservative wording edits shared by the two resume suggestion workflows."""

import re


def rewrite_source_passage(text: str) -> str:
    text = re.sub(r"^(?:我在|我曾在)", "在", text.strip())
    return re.sub(r"^我(?=使用|负责|实现|参与|开发|维护|编写|完成)", "", text)
