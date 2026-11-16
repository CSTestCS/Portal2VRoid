"""Minimal, order-preserving Valve KeyValues parser/writer (for .res files)."""

from __future__ import annotations

import copy


class KV:
    """A KeyValues node: ``key`` plus either a string ``value`` or a list of ``children``."""

    __slots__ = ("key", "value", "children", "cond")

    def __init__(self, key: str, value: str | None = None, children: list["KV"] | None = None, cond: str = ""):
        self.key = key
        self.value = value
        self.children = children if children is not None else ([] if value is None else None)
        self.cond = cond

    @property
    def is_block(self) -> bool:
        return self.value is None

    def get(self, key: str, default=None):
        """First child value with this key (case-insensitive)."""
        k = key.lower()
        for c in self.children or []:
            if c.key.lower() == k:
                return c.value if not c.is_block else c
        return default

    def find(self, key: str) -> "KV | None":
        k = key.lower()
        for c in self.children or []:
            if c.key.lower() == k:
                return c
        return None

    def set(self, key: str, value: str) -> None:
        node = self.find(key)
        if node is not None and not node.is_block:
            node.value = value
            # drop conditional duplicates of the same key so the new value wins everywhere
            self.children = [c for c in self.children if c is node or c.key.lower() != key.lower()]
        else:
            self.children.append(KV(key, value))

    def remove(self, key: str) -> None:
        k = key.lower()
        self.children = [c for c in self.children or [] if c.key.lower() != k]

    def blocks(self) -> list["KV"]:
        return [c for c in self.children or [] if c.is_block]

    def clone(self) -> "KV":
        return copy.deepcopy(self)


def _tokenize(text: str):
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in " \t\r\n﻿":
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j + 1
        elif ch == "{" or ch == "}":
            yield ch, False
            i += 1
        elif ch == '"':
            j = i + 1
            buf = []
            while j < n and text[j] != '"':  # no escape sequences (Valve default for .res)
                buf.append(text[j])
                j += 1
            yield "".join(buf), True
            i = j + 1
        elif ch == "[":
            j = text.find("]", i)
            j = n if j < 0 else j + 1
            yield text[i:j], "cond"
            i = j
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n{}"':
                j += 1
            yield text[i:j], True
            i = j


def parse(text: str) -> KV:
    """Parse text into a root KV block whose children are the top-level entries."""
    root = KV("", None, [])
    stack = [root]
    pending_key: str | None = None
    pending_cond = ""
    last: KV | None = None
    for tok, kind in _tokenize(text):
        if kind == "cond":
            if pending_key is not None:
                pending_cond = tok  # `"Key" [$X360] { ... }`: belongs to the block being opened
            elif last is not None:
                last.cond = tok  # `"key" "value" [$WIN32]`
            continue
        if tok == "{" and kind is False:
            node = KV(pending_key or "", None, [], pending_cond)
            pending_cond = ""
            stack[-1].children.append(node)
            stack.append(node)
            pending_key = None
            last = node
        elif tok == "}" and kind is False:
            if len(stack) > 1:
                stack.pop()
            pending_key = None
            last = None
        elif pending_key is None:
            pending_key = tok
        else:
            node = KV(pending_key, tok)
            stack[-1].children.append(node)
            pending_key = None
            last = node
    return root


def _q(s: str) -> str:
    return '"' + s.replace('"', "'") + '"'


def dump(root: KV) -> str:
    out: list[str] = []

    def emit(node: KV, depth: int):
        ind = "\t" * depth
        cond = f" {node.cond}" if node.cond else ""
        if node.is_block:
            out.append(f"{ind}{_q(node.key)}{cond}")
            out.append(f"{ind}{{")
            for c in node.children:
                emit(c, depth + 1)
            out.append(f"{ind}}}")
        elif node.key.startswith("#"):  # #base / #include directives stay unquoted
            out.append(f"{ind}{node.key} {_q(node.value)}")
        else:
            out.append(f"{ind}{_q(node.key)}\t\t{_q(node.value)}{cond}")

    for c in root.children:
        emit(c, 0)
    return "\n".join(out) + "\n"
