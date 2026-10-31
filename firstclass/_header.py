"""`from_header`: parse the struct/enum definitions of a C/C++ header into firstclass classes.

A small C preprocessor + declaration parser, stdlib only. Anything that cannot be represented
(unions, flexible arrays, unknown types, `#pragma pack(2)`, ...) is skipped and reported in
`.skipped` instead of failing the whole header.
"""
from __future__ import annotations

import ast
import collections
import ctypes
import enum
import os
import re
from pathlib import Path
from typing import NamedTuple

from . import _specs as S
from ._core import firstclass

Tok = collections.namedtuple("Tok", "kind text")  # kinds: id num chr str op attr pack


class _Skip(Exception):
    """This declaration cannot be represented; the message is the reason."""


class _ExprError(ValueError):
    pass


# --- tokenizer ---------------------------------------------------------------

_COMMENTS = re.compile(r'"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|/\*.*?\*/|//[^\n]*', re.S)
_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<id>[A-Za-z_$][\w$]*)
  | (?P<num>0[xX][0-9a-fA-F]+[uUlL]*|0[bB][01]+[uUlL]*|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlL]?|\d+[uUlL]*)
  | (?P<chr>'(?:\\.|[^\\'])+')
  | (?P<str>"(?:\\.|[^\\"])*")
  | (?P<op><<=|>>=|\.\.\.|::|->|<<|>>|<=|>=|==|!=|&&|\|\||[-+*/%&|^~!<>=?:;,.(){}\[\]\#])
  | (?P<other>.)
""", re.X)


def _strip_comments(text: str) -> str:
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    return _COMMENTS.sub(lambda m: " " if m.group()[0] == "/" else m.group(), text)


def _tokenize(text: str) -> list[Tok]:
    out = []
    for m in _TOKEN.finditer(text):
        kind = m.lastgroup
        if kind not in ("ws", "other"):
            out.append(Tok(kind, m.group()))
    return out


def _int_value(text: str) -> int:
    body = text.rstrip("uUlL")
    if re.fullmatch(r"0[xX][0-9a-fA-F]+", body):
        return int(body, 16)
    if re.fullmatch(r"0[bB][01]+", body):
        return int(body, 2)
    if re.fullmatch(r"0[0-7]+", body):
        return int(body, 8)
    if re.fullmatch(r"\d+", body):
        return int(body)
    raise _ExprError(f"not an integer constant: {text}")


def _match(toks: list[Tok], i: int) -> int:
    """Index of the bracket closing toks[i] ('(' '[' '{'), or the last index if unbalanced."""
    open_, close = {"(": ("(", ")"), "[": ("[", "]"), "{": ("{", "}")}[toks[i].text]
    depth = 0
    for j in range(i, len(toks)):
        if toks[j].kind == "op":
            if toks[j].text == open_:
                depth += 1
            elif toks[j].text == close:
                depth -= 1
                if depth == 0:
                    return j
    return len(toks) - 1


# --- constant expressions ----------------------------------------------------

_BINARY_LEVELS = [("||",), ("&&",), ("|",), ("^",), ("&",), ("==", "!="), ("<", ">", "<=", ">="),
                  ("<<", ">>"), ("+", "-"), ("*", "/", "%")]


class _Expr:
    def __init__(self, toks, ctx):
        self.t, self.i, self.ctx = toks, 0, ctx

    def value(self) -> int:
        if not self.t:
            raise _ExprError("empty expression")
        v = self.ternary()
        if self.i != len(self.t):
            raise _ExprError(f"unexpected token {self.t[self.i].text!r}")
        return v

    def peek(self):
        return self.t[self.i].text if self.i < len(self.t) and self.t[self.i].kind == "op" else None

    def ternary(self):
        c = self.binary(0)
        if self.peek() == "?":
            self.i += 1
            a = self.ternary()
            if self.peek() != ":":
                raise _ExprError("expected ':'")
            self.i += 1
            b = self.ternary()
            return a if c else b
        return c

    def binary(self, level):
        if level == len(_BINARY_LEVELS):
            return self.unary()
        v = self.binary(level + 1)
        while self.peek() in _BINARY_LEVELS[level]:
            op = self.peek()
            self.i += 1
            r = self.binary(level + 1)
            v = self.apply(op, v, r)
        return v

    @staticmethod
    def apply(op, a, b):
        if op in ("/", "%") and b == 0:
            raise _ExprError("division by zero")
        if op == "/":
            q = abs(a) // abs(b)
            return q if (a < 0) == (b < 0) else -q
        if op == "%":
            return a - b * _Expr.apply("/", a, b)
        if op in ("<<", ">>") and not 0 <= b < 128:
            raise _ExprError("bad shift count")
        return {
            "||": lambda: int(bool(a or b)), "&&": lambda: int(bool(a and b)), "|": lambda: a | b,
            "^": lambda: a ^ b, "&": lambda: a & b, "==": lambda: int(a == b), "!=": lambda: int(a != b),
            "<": lambda: int(a < b), ">": lambda: int(a > b), "<=": lambda: int(a <= b),
            ">=": lambda: int(a >= b), "<<": lambda: a << b, ">>": lambda: a >> b,
            "+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
        }[op]()

    def unary(self):
        op = self.peek()
        if op in ("-", "+", "~", "!"):
            self.i += 1
            v = self.unary()
            return {"-": -v, "+": v, "~": ~v, "!": int(not v)}[op]
        if op == "(":
            close = _match(self.t, self.i)
            inner = self.t[self.i + 1:close]
            if inner and self.ctx.is_type_words(inner) and close + 1 < len(self.t):
                self.i = close + 1  # cast: ignore the type
                return self.unary()
            self.i += 1
            v = self.ternary()
            if self.peek() != ")":
                raise _ExprError("expected ')'")
            self.i += 1
            return v
        return self.primary()

    def primary(self):
        if self.i >= len(self.t):
            raise _ExprError("unexpected end of expression")
        tok = self.t[self.i]
        if tok.kind == "num":
            self.i += 1
            return _int_value(tok.text)
        if tok.kind == "chr":
            self.i += 1
            return ord(ast.literal_eval("'" + tok.text[1:-1].replace("'", "\\'") + "'")[0])
        if tok.kind == "id":
            if tok.text == "sizeof":
                if self.i + 1 < len(self.t) and self.t[self.i + 1].text == "(":
                    close = _match(self.t, self.i + 1)
                    inner = self.t[self.i + 2:close]
                    self.i = close + 1
                    return self.ctx.sizeof_words([x.text for x in inner])
                raise _ExprError("sizeof needs a parenthesised type")
            name = tok.text
            self.i += 1
            while self.peek() == "::" and self.i + 1 < len(self.t) and self.t[self.i + 1].kind == "id":
                name += "::" + self.t[self.i + 1].text
                self.i += 2
            return self.ctx.constant(name)
        raise _ExprError(f"unexpected token {tok.text!r}")


# --- type model --------------------------------------------------------------

_BUILTIN = {"void", "char", "short", "int", "long", "float", "double", "signed", "unsigned", "_Bool", "bool"}
_QUALS = {"const", "volatile", "register", "static", "extern", "inline", "mutable", "constexpr",
          "typename", "thread_local", "_Thread_local"}
_DROP = {"__extension__", "__restrict", "__restrict__", "restrict", "__inline", "__inline__"}
_PTR_BITS = ctypes.sizeof(ctypes.c_void_p) * 8
_INTS = {
    (8, False): S.uint8_t, (16, False): S.uint16_t, (32, False): S.uint32_t, (64, False): S.uint64_t,
    (8, True): S.int8_t, (16, True): S.int16_t, (32, True): S.int32_t, (64, True): S.int64_t,
}


class FieldDecl(NamedTuple):
    name: str
    words: tuple
    ptr: bool
    dims: list
    bits: int | None
    error: str | None


class TypedefDecl(NamedTuple):
    words: tuple
    ptr: bool
    dims: list
    error: str | None


class StructDecl:
    def __init__(self, name, kind, pack, attrs):
        self.name, self.kind, self.pack, self.attrs = name, kind, pack, attrs
        self.fields: list[FieldDecl] = []
        self.reason: str | None = "union unsupported" if kind == "union" else None


class EnumDecl:
    def __init__(self, name, members, underlying, scoped):
        self.name, self.members, self.underlying, self.scoped = name, members, underlying, scoped


def _named_scalar(name):
    m = re.fullmatch(r"(u?)int(8|16|32|64)_t", name)
    if m:
        return ("int", int(m.group(2)), not m.group(1))
    if name in ("size_t", "uintptr_t"):
        return ("int", _PTR_BITS, False)
    if name in ("ssize_t", "ptrdiff_t", "intptr_t"):
        return ("int", _PTR_BITS, True)
    return None


def _builtin_type(words, long_size):
    c = collections.Counter(words)
    if "void" in c:
        return ("void",)
    if "_Bool" in c or "bool" in c:
        return ("bool",)
    if "float" in c:
        return ("float",)
    if "double" in c:
        if "long" in c:
            raise _Skip("long double unsupported")
        return ("double",)
    if "char" in c:
        if "unsigned" in c:
            return ("int", 8, False)
        if "signed" in c:
            return ("int", 8, True)
        return ("char",)
    if "short" in c:
        bits = 16
    elif c["long"] >= 2:
        bits = 64
    elif c["long"] == 1:
        bits = long_size * 8
    else:
        bits = 32
    return ("int", bits, "unsigned" not in c)


class _Ctx:
    """Everything the parser learns: shared by the top-level parser and sub-parsers."""

    def __init__(self, long_size):
        self.structs: dict[str, StructDecl] = {}
        self.enums: dict[str, EnumDecl] = {}
        self.typedefs: dict[str, TypedefDecl] = {}
        self.constants: dict[str, int] = {}
        self.pack = None
        self.long_size = long_size
        self._anon = 0

    def anon_name(self):
        self._anon += 1
        return f"_anon{self._anon}"

    def constant(self, name):
        if name in self.constants:
            return self.constants[name]
        last = name.rsplit("::", 1)[-1]
        if last in self.constants and "::" in name:
            return self.constants[last]
        raise _ExprError(f"unknown constant {name}")

    def eval(self, toks):
        return _Expr(toks, self).value()

    def is_type_words(self, toks):
        return all(t.kind == "id" and (t.text in _BUILTIN or t.text in self.typedefs
                                       or _named_scalar(t.text)) or t.text == "*" for t in toks)

    def sizeof_words(self, words):
        words = [w for w in words if w not in _QUALS]
        if "*" in words:
            return _PTR_BITS // 8
        if len(words) == 1 and words[0] in self.typedefs:
            td = self.typedefs[words[0]]
            return self.sizeof_words(list(td.words)) if not td.ptr and not td.dims else _PTR_BITS // 8
        sc = _named_scalar(words[0]) if len(words) == 1 else None
        base = sc or (_builtin_type(words, self.long_size) if all(w in _BUILTIN for w in words) else None)
        if base is None:
            raise _ExprError(f"sizeof({' '.join(words)}) unsupported")
        return {"int": (base[1] // 8 if base[0] == "int" else 0), "char": 1, "bool": 1,
                "float": 4, "double": 8}.get(base[0]) or (base[1] // 8)

    def rename(self, old, new):
        if old in self.structs:
            decl = self.structs.pop(old)
            decl.name = new
            self.structs[new] = decl
        elif old in self.enums:
            decl = self.enums.pop(old)
            decl.name = new
            self.enums[new] = decl


# --- preprocessor ------------------------------------------------------------

class _Preprocessor:
    def __init__(self, defines):
        self.macros: dict[str, list[Tok]] = {}
        self.funclike: set[str] = set()
        self.pack_stack: list = []
        self.pack = None
        for k, v in (defines or {}).items():
            self.macros[k] = _tokenize(str(v))

    def expand(self, toks, stack=()):
        out = []
        for t in toks:
            if t.kind == "id" and t.text in self.macros and t.text not in stack:
                out.extend(self.expand(self.macros[t.text], stack + (t.text,)))
            else:
                out.append(t)
        return out

    def cond(self, toks, ctx):
        out, i = [], 0
        while i < len(toks):
            t = toks[i]
            if t.kind == "id" and t.text == "defined":
                j = i + 1
                paren = j < len(toks) and toks[j].text == "("
                j += paren
                name = toks[j].text if j < len(toks) else ""
                j += 1 + paren
                out.append(Tok("num", "1" if name in self.macros or name in self.funclike else "0"))
                i = j
            else:
                out.append(t)
                i += 1
        out = [Tok("num", "0") if t.kind == "id" and t.text != "sizeof" else t for t in self.expand(out)]
        try:
            return bool(_Expr(out, ctx).value())
        except (_ExprError, ValueError, KeyError):
            return False

    def run(self, text, ctx):
        out: list[Tok] = []
        conds: list[list[bool]] = []  # [parent_active, taken, active]
        active = lambda: all(c[2] for c in conds)  # noqa: E731
        for line in _strip_comments(text).splitlines():
            s = line.strip()
            if not s.startswith("#"):
                if active():
                    out.extend(self.expand(_tokenize(line)))
                continue
            toks = _tokenize(s[1:])
            if not toks:
                continue
            d, rest = toks[0].text, toks[1:]
            if d in ("if", "ifdef", "ifndef"):
                parent = active()
                if d == "if":
                    val = parent and self.cond(rest, ctx)
                else:
                    name = rest[0].text if rest else ""
                    val = parent and ((name in self.macros or name in self.funclike) == (d == "ifdef"))
                conds.append([parent, val, val])
            elif d == "elif" and conds:
                c = conds[-1]
                val = c[0] and not c[1] and self.cond(rest, ctx)
                c[1] |= val
                c[2] = val
            elif d == "else" and conds:
                c = conds[-1]
                c[2] = c[0] and not c[1]
                c[1] = True
            elif d == "endif" and conds:
                conds.pop()
            elif not active():
                continue
            elif d == "define" and rest and rest[0].kind == "id":
                name = rest[0].text
                # function-like only when '(' touches the name
                after = s[1:].lstrip()[len("define"):].lstrip()[len(name):]
                if after.startswith("("):
                    self.funclike.add(name)
                    self.macros.pop(name, None)
                else:
                    self.macros[name] = rest[1:]
                    self.funclike.discard(name)
            elif d == "undef" and rest:
                self.macros.pop(rest[0].text, None)
                self.funclike.discard(rest[0].text)
            elif d == "pragma" and rest and rest[0].text == "pack":
                self.handle_pack([t.text for t in rest[1:] if t.text not in ("(", ")", ",")])
                out.append(Tok("pack", self.pack))
        return out

    def handle_pack(self, args):
        if not args:
            self.pack = None
        elif args[0] == "push":
            self.pack_stack.append(self.pack)
            nums = [a for a in args[1:] if a.isdigit()]
            if nums:
                self.pack = int(nums[-1]) or None
        elif args[0] == "pop":
            self.pack = self.pack_stack.pop() if self.pack_stack else None
        elif args[0].isdigit():
            self.pack = int(args[0]) or None


def _fold_attributes(toks):
    """Drop attribute noise; keep `packed` / `aligned` as 'attr' tokens."""
    out, i = [], 0
    while i < len(toks):
        t = toks[i]
        if t.kind == "id" and t.text in ("__attribute__", "__attribute", "__declspec", "alignas", "_Alignas"):
            if i + 1 < len(toks) and toks[i + 1].text == "(":
                close = _match(toks, i + 1)
                names = {x.text.strip("_") for x in toks[i + 2:close] if x.kind == "id"}
                if t.text in ("alignas", "_Alignas") or "aligned" in names or "align" in names:
                    out.append(Tok("attr", "aligned"))
                if "packed" in names:
                    out.append(Tok("attr", "packed"))
                i = close + 1
                continue
        if t.kind == "id" and t.text.strip("_") == "packed":
            out.append(Tok("attr", "packed"))
        elif t.kind == "id" and t.text in _DROP:
            pass
        else:
            out.append(t)
        i += 1
    return out


# --- declaration parser ------------------------------------------------------

class _Parser:
    def __init__(self, toks, ctx):
        self.t, self.i, self.ctx = toks, 0, ctx

    # small helpers
    def at(self, j):
        return self.t[j].text if 0 <= j < len(self.t) else None

    def is_op(self, j, s):
        return 0 <= j < len(self.t) and self.t[j].kind == "op" and self.t[j].text == s

    def statement_end(self, i):
        t, n, j, seen_paren = self.t, len(self.t), i, False
        while j < n:
            if t[j].kind == "op":
                x = t[j].text
                if x == ";":
                    return j + 1
                if x in "([":
                    seen_paren |= x == "("
                    j = _match(t, j) + 1
                    continue
                if x == "{":
                    j = _match(t, j) + 1
                    if seen_paren:  # function body ends the statement
                        return j + 1 if self.is_op(j, ";") else j
                    continue
            j += 1
        return n

    # top level
    def parse_all(self):
        t, ctx = self.t, self.ctx
        while self.i < len(t):
            tok = t[self.i]
            start = self.i
            if tok.kind == "pack":
                ctx.pack = tok.text
                self.i += 1
            elif tok.kind == "op" and tok.text in (";", "}"):
                self.i += 1
            elif tok.kind == "id" and tok.text == "namespace":
                j = self.i
                while j < len(t) and not (t[j].kind == "op" and t[j].text in ("{", ";")):
                    j += 1
                self.i = j + 1  # enter the body; its closing '}' is skipped above
            elif tok.kind == "id" and tok.text == "extern" and self.at(self.i + 1) and \
                    self.t[self.i + 1].kind == "str" and self.is_op(self.i + 2, "{"):
                self.i += 3
            elif tok.kind == "id" and tok.text == "typedef":
                self.i += 1
                self.parse_typedef()
            elif tok.kind == "id" and tok.text == "using":
                self.parse_using()
            elif tok.kind == "id" and tok.text in ("struct", "union", "class", "enum") and self.is_definition():
                self.parse_specifiers()
                self.i = self.statement_end(self.i)
            elif not self.try_constant():
                self.i = self.statement_end(self.i)
            if self.i == start:  # always make progress
                self.i += 1

    def is_definition(self):
        t, j = self.t, self.i + 1
        if t[self.i].text == "enum" and self.at(j) in ("class", "struct"):
            j += 1
        while j < len(t) and t[j].kind in ("attr", "id"):
            j += 1
        if self.is_op(j, ":"):
            while j < len(t) and not (t[j].kind == "op" and t[j].text in ("{", ";")):
                j += 1
        return self.is_op(j, "{")

    def try_constant(self):
        end = self.statement_end(self.i)
        stmt = [x for x in self.t[self.i:end] if not (x.kind == "op" and x.text == ";")]
        eq = next((k for k, x in enumerate(stmt) if x.kind == "op" and x.text == "="), None)
        if eq is None or eq < 2 or stmt[eq - 1].kind != "id":
            return False
        head = [x.text for x in stmt[:eq]]
        if "(" in head or not ({"const", "constexpr"} & set(head)):
            return False
        rhs = stmt[eq + 1:]
        comma = next((k for k, x in enumerate(rhs) if x.kind == "op" and x.text == ","), len(rhs))
        try:
            self.ctx.constants[stmt[eq - 1].text] = self.ctx.eval(rhs[:comma])
        except (_ExprError, ValueError):
            pass
        self.i = end
        return True

    # typedef / using
    def parse_typedef(self):
        words = self.parse_specifiers()
        decls = []
        while self.i < len(self.t):
            decls.append(self.parse_declarator())
            if self.is_op(self.i, ","):
                self.i += 1
                continue
            break
        self.i = self.statement_end(self.i)
        if words and words[0] in ("struct", "enum", "union") and len(words) == 2 and words[1].startswith("_anon"):
            plain = next((d for d in decls if d[0] and not d[1] and not d[2]), None)
            if plain:
                self.ctx.rename(words[1], plain[0])
                words = (words[0], plain[0])
        for name, ptr, dims, _bits, err in decls:
            if name:
                self.ctx.typedefs[name] = TypedefDecl(words, ptr, dims, err)

    def parse_using(self):
        j = self.i + 1
        if self.at(j) == "namespace" or not (j + 1 < len(self.t) and self.t[j].kind == "id" and self.is_op(j + 1, "=")):
            self.i = self.statement_end(self.i)
            return
        name = self.t[j].text
        self.i = j + 2
        words = self.parse_specifiers()
        ptr = False
        while self.is_op(self.i, "*"):
            ptr = True
            self.i += 1
        self.ctx.typedefs[name] = TypedefDecl(words, ptr, [], None)
        self.i = self.statement_end(self.i)

    # specifiers (incl. inline struct/enum definitions)
    def parse_specifiers(self):
        t, words = self.t, []
        while self.i < len(t):
            tok = t[self.i]
            if tok.kind in ("attr", "pack"):
                self.i += 1
            elif tok.kind != "id":
                break
            elif tok.text in _QUALS:
                self.i += 1
            elif tok.text in _BUILTIN:
                words.append(tok.text)
                self.i += 1
            elif tok.text in ("struct", "union", "class", "enum"):
                return tuple(self.parse_tag())
            elif not words:
                words.append(tok.text)
                self.i += 1
                while self.is_op(self.i, "::") and self.i + 1 < len(t) and t[self.i + 1].kind == "id":
                    words += ["::", t[self.i + 1].text]  # qualified name, resolved later
                    self.i += 2
                break
            else:
                break
        return tuple(words)

    def parse_tag(self):
        t, ctx = self.t, self.ctx
        kw = t[self.i].text
        j, scoped, attrs = self.i + 1, False, set()
        if kw == "enum" and self.at(j) in ("class", "struct"):
            scoped = True
            j += 1
        while j < len(t) and t[j].kind == "attr":
            attrs.add(t[j].text)
            j += 1
        name = None
        if j < len(t) and t[j].kind == "id":
            name = t[j].text
            j += 1
        while j < len(t) and t[j].kind == "attr":
            attrs.add(t[j].text)
            j += 1
        clause = None
        if self.is_op(j, ":"):
            k = j + 1
            while k < len(t) and not (t[k].kind == "op" and t[k].text in ("{", ";")):
                k += 1
            if self.is_op(k, "{"):
                clause, j = t[j + 1:k], k
        kind = "struct" if kw == "class" else kw
        if not self.is_op(j, "{"):
            self.i = j
            return (kind, name)
        close = _match(t, j)
        body = t[j + 1:close]
        self.i = close + 1
        while self.i < len(t) and t[self.i].kind == "attr":
            attrs.add(t[self.i].text)
            self.i += 1
        reg = name or ctx.anon_name()
        if kind == "enum":
            self.define_enum(reg, body, clause, scoped)
        else:
            decl = StructDecl(reg, kind, ctx.pack, attrs)
            ctx.structs[reg] = decl  # visible to its own members (self pointers)
            if clause:
                decl.reason = "inheritance unsupported"
            _Parser(body, ctx).parse_struct_body(decl)
        return (kind, reg)

    def define_enum(self, name, body, clause, scoped):
        ctx, members, value = self.ctx, [], -1
        declared = {}
        parts, cur = [], []
        for x in body:
            if x.kind == "op" and x.text == ",":
                parts.append(cur)
                cur = []
            else:
                cur.append(x)
        parts.append(cur)
        for part in parts:
            if not part:
                continue
            mname = part[0].text
            if len(part) > 1 and part[1].text == "=":
                saved = dict(ctx.constants)
                ctx.constants.update(declared)
                try:
                    value = ctx.eval(part[2:])
                except _ExprError:
                    ctx.constants = saved
                    ctx.enums[name] = EnumDecl(name, [], None, scoped)
                    ctx.enums[name].error = f"cannot evaluate {mname}"
                    return
                ctx.constants = saved
            else:
                value += 1
            members.append((mname, value))
            declared[mname] = value
            ctx.constants[f"{name}::{mname}" if scoped else mname] = value
        decl = EnumDecl(name, members, tuple(x.text for x in clause) if clause else None, scoped)
        ctx.enums[name] = decl

    # struct members
    def parse_struct_body(self, decl):
        ctx = self.ctx
        while self.i < len(self.t):
            tok = self.t[self.i]
            start = self.i
            if tok.kind in ("pack", "attr") or (tok.kind == "op" and tok.text == ";"):
                self.i += 1
                continue
            if tok.kind == "id" and tok.text in ("public", "private", "protected") and self.is_op(self.i + 1, ":"):
                self.i += 2
                continue
            end = self.statement_end(self.i)
            stmt = [x for x in self.t[self.i:end]]
            self.i = max(end, start + 1)
            if stmt and stmt[-1].kind == "op" and stmt[-1].text == ";":
                stmt = stmt[:-1]
            if not stmt:
                continue
            first = stmt[0].text
            if first in ("typedef", "using"):
                sub = _Parser(stmt, ctx)
                sub.i = 1 if first == "typedef" else 0
                sub.parse_typedef() if first == "typedef" else sub.parse_using()
                continue
            if first in ("static", "friend", "template", "static_assert", "_Static_assert"):
                continue
            if any(x.kind == "id" and x.text == "virtual" for x in stmt):
                decl.reason = "virtual members unsupported"
                continue
            if self.is_method(stmt):
                continue
            _Parser(stmt, ctx).parse_field_stmt(decl)
        if not decl.fields and not decl.reason:
            decl.reason = "empty struct"

    @staticmethod
    def is_method(stmt):
        depth, k = 0, 0
        while k < len(stmt):
            x = stmt[k]
            if x.kind == "op" and x.text in "{[":
                k = _match(stmt, k)
            elif x.kind == "op" and x.text == "(":
                nxt = stmt[k + 1] if k + 1 < len(stmt) else None
                return not (nxt is not None and nxt.kind == "op" and nxt.text == "*")
            k += 1
        return False

    def parse_field_stmt(self, decl):
        words = self.parse_specifiers()
        inline_anon = bool(words) and words[0] in ("struct", "union", "enum") and len(words) == 2 \
            and words[1].startswith("_anon")
        if not words:
            decl.reason = decl.reason or "unsupported member declaration"
            return
        found = []
        while self.i < len(self.t):
            name, ptr, dims, bits, err = self.parse_declarator()
            if name is None and bits is None:
                break
            if name is None:
                decl.reason = decl.reason or "unnamed bitfield (padding) unsupported"
            else:
                found.append(FieldDecl(name, words, ptr, dims, bits, err))
            if self.is_op(self.i, ","):
                self.i += 1
                continue
            break
        if not found and inline_anon:
            decl.reason = decl.reason or "anonymous member unsupported"
        if inline_anon and len(found) == 1 and not found[0].ptr:
            new = f"{decl.name}_{found[0].name}"
            self.ctx.rename(words[1], new)
            found[0] = found[0]._replace(words=(words[0], new))
        decl.fields.extend(found)

    def parse_declarator(self):
        t, ctx = self.t, self.ctx
        ptr, err, name, dims = False, None, None, []
        while self.i < len(t) and (self.is_op(self.i, "*") or self.is_op(self.i, "&") or t[self.i].kind == "attr"
                                   or (t[self.i].kind == "id" and t[self.i].text in _QUALS)):
            ptr |= self.is_op(self.i, "*")
            self.i += 1
        if self.is_op(self.i, "(") and self.is_op(self.i + 1, "*"):  # function pointer
            ptr = True
            self.i += 2
            while self.i < len(t) and t[self.i].kind == "id" and t[self.i].text in _QUALS:
                self.i += 1
            if self.i < len(t) and t[self.i].kind == "id":
                name = t[self.i].text
                self.i += 1
            err = self.parse_dims(dims) or err
            if self.is_op(self.i, ")"):
                self.i += 1
            if self.is_op(self.i, "("):
                self.i = _match(t, self.i) + 1
        else:
            if self.i < len(t) and t[self.i].kind == "id":
                name = t[self.i].text
                self.i += 1
            err = self.parse_dims(dims) or err
        bits = None
        if self.is_op(self.i, ":"):
            k = self.i + 1
            while k < len(t) and not (t[k].kind == "op" and t[k].text in (",", "=")):
                k += 1
            try:
                bits = ctx.eval(t[self.i + 1:k])
            except (_ExprError, ValueError) as e:
                bits, err = 0, err or f"bitfield width: {e}"
            self.i = k
        if self.is_op(self.i, "="):  # default member initialiser
            depth = 0
            while self.i < len(t):
                if t[self.i].kind == "op":
                    if t[self.i].text in "([{":
                        self.i = _match(t, self.i)
                    elif t[self.i].text == "," and depth == 0:
                        break
                self.i += 1
        return name, ptr, dims, bits, err

    def parse_dims(self, dims):
        t, err = self.t, None
        while self.is_op(self.i, "["):
            close = _match(t, self.i)
            expr = t[self.i + 1:close]
            self.i = close + 1
            if not expr:
                dims.append(None)
                continue
            try:
                dims.append(self.ctx.eval(expr))
            except (_ExprError, ValueError) as e:
                dims.append(None)
                err = err or f"array size: {e}"
        return err


# --- building classes ---------------------------------------------------------

class HeaderTypes(dict):
    """name -> generated class (structs and enums, typedef aliases included).

    Also `.structs`, `.enums`, `.constants` (#defines, consts, enum members as ints) and
    `.skipped` (name -> reason it could not be converted). Names are reachable as attributes too.
    """

    def __init__(self):
        super().__init__()
        self.structs, self.enums, self.constants, self.skipped = {}, {}, {}, {}

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class _Builder:
    def __init__(self, ctx, byteorder, align):
        self.ctx, self.byteorder, self.align = ctx, byteorder, align
        self.structs: dict[str, type] = {}
        self.enums: dict[str, tuple] = {}
        self.failed: dict[str, str] = {}
        self.building: set[str] = set()

    def lower(self, words, depth=0):
        """words -> (base, is_pointer, inner_dims)"""
        ctx = self.ctx
        # `std::uint64_t` / `ns::Type` -> last component
        words = tuple(w for i, w in enumerate(words)
                      if w != "::" and not (i + 1 < len(words) and words[i + 1] == "::"))
        if depth > 16 or not words:
            raise _Skip("cannot resolve type")
        if len(words) == 1:
            w = words[0]
            if w in ctx.typedefs:
                td = ctx.typedefs[w]
                if td.error:
                    raise _Skip(td.error)
                base, ptr, dims = self.lower(td.words, depth + 1)
                return base, ptr or td.ptr, list(td.dims) + dims
            if w in ctx.structs:
                return ("struct", w), False, []
            if w in ctx.enums:
                return ("enum", w), False, []
            sc = _named_scalar(w)
            if sc:
                return sc, False, []
        if words[0] in ("struct", "union", "enum") and len(words) == 2:
            if words[0] == "union":
                raise _Skip("union unsupported")
            return (words[0], words[1]), False, []
        if all(w in _BUILTIN for w in words):
            return _builtin_type(words, ctx.long_size), False, []
        raise _Skip(f"unknown type {' '.join(words)}")

    def struct(self, name):
        if name in self.structs:
            return self.structs[name]
        if name in self.failed:
            raise _Skip(f"depends on skipped struct {name} ({self.failed[name]})")
        if name in self.building:
            raise _Skip(f"struct {name} contains itself")
        decl = self.ctx.structs.get(name)
        if decl is None:
            raise _Skip(f"unknown struct {name}")
        self.building.add(name)
        try:
            cls = self._build_struct(decl)
        except (_Skip, TypeError, ValueError) as e:
            self.failed[name] = str(e)
            raise _Skip(str(e)) from None
        finally:
            self.building.discard(name)
        self.structs[name] = cls
        return cls

    def _build_struct(self, decl):
        if decl.reason:
            raise _Skip(decl.reason)
        pack = decl.pack
        if "aligned" in decl.attrs:
            raise _Skip("aligned attribute unsupported")
        if pack not in (None, 1) and pack < 8:
            raise _Skip(f"#pragma pack({pack}) unsupported (only 1 or natural)")
        aligned = not (pack == 1 or "packed" in decl.attrs)
        if self.align is not None:
            aligned = self.align
        ann = {}
        for f in decl.fields:
            if f.name in ann:
                raise _Skip(f"duplicate field {f.name}")
            try:
                ann[f.name] = self.annotation(f)
            except (_Skip, TypeError, ValueError) as e:
                raise _Skip(f"{decl.name}.{f.name}: {e}") from None
        cls = type(decl.name, (), {"__annotations__": ann, "__module__": "firstclass.header"})
        cls = firstclass(byteorder=self.byteorder, align=aligned)(cls)
        cls.sizeof()  # validates the layout now rather than at first use
        return cls

    def enum(self, name):
        if name in self.enums:
            return self.enums[name]
        decl = self.ctx.enums.get(name)
        if decl is None:
            raise _Skip(f"unknown enum {name}")
        if getattr(decl, "error", None):
            raise _Skip(decl.error)
        if decl.underlying:
            base, ptr, dims = self.lower(decl.underlying)
            if base[0] != "int" or ptr or dims:
                raise _Skip("enum underlying type must be an integer")
            storage = _INTS[(base[1], base[2])]
        else:
            vals = [v for _, v in decl.members] or [0]
            lo, hi = min(vals), max(vals)
            if lo >= 0:
                storage = S.uint32_t if hi < 2 ** 32 else S.uint64_t
            else:
                storage = S.int32_t if lo >= -2 ** 31 and hi < 2 ** 31 else S.int64_t
        cls = enum.IntEnum(decl.name, [(k, v) for k, v in decl.members], module="firstclass.header")
        annotation = S.enum_t[cls, storage]  # raises if a value does not fit the storage
        self.enums[name] = (cls, annotation)
        return self.enums[name]

    def base_annotation(self, base):
        kind = base[0]
        if kind == "int":
            return _INTS[(base[1], base[2])]
        if kind == "float":
            return S.float32_t
        if kind == "double":
            return S.float64_t
        if kind == "bool":
            return S.bool_t
        if kind == "char":
            return S.char_t
        if kind == "struct":
            return self.struct(base[1])
        if kind == "enum":
            return self.enum(base[1])[1]
        raise _Skip("void field")

    def annotation(self, f):
        if f.error:
            raise _Skip(f.error)
        base, ptr, inner = self.lower(f.words)
        ptr = ptr or f.ptr
        dims = list(f.dims) + inner
        if f.bits is not None:
            if base[0] != "int" or ptr or dims:
                raise _Skip("bitfield needs a plain integer type")
            return S.bitfield[_INTS[(base[1], base[2])], f.bits]
        ann = ctypes.c_void_p if ptr else self.base_annotation(base)
        if not dims:
            return ann
        if any(d is None or d <= 0 for d in dims):
            raise _Skip("flexible or zero-length array unsupported")
        dims = list(dims)
        last = dims.pop()
        if not ptr and base == ("char",):
            elem = S.char_array[last]
        elif not ptr and base == ("int", 8, False):
            elem = S.bytes_t[last]
        else:
            elem = S.array_t[ann, last]
        while dims:
            elem = S.array_t[elem, dims.pop()]
        return elem


def from_header(path=None, *, text=None, byteorder: str = "little", align: bool | None = None,
                defines: dict | None = None, long_size: int | None = None) -> HeaderTypes:
    """Parse a C/C++ header (`path`, or source `text=`) into firstclass classes.

    structs/classes become `@firstclass` dataclasses, enums/enum classes become `IntEnum`s.
    Layout follows the C rules: natural alignment padding, unless the struct is
    `#pragma pack(1)` / `__attribute__((packed))`. Override with `align=True/False`.
    `defines` presets macros; `long_size` is the size of C `long` (default: this machine's).
    Things that cannot be represented are listed in the result's `.skipped`.
    """
    if (path is None) == (text is None):
        raise TypeError("pass exactly one of path or text=")
    if path is not None:
        text = Path(os.fspath(path)).read_text(encoding="utf-8", errors="replace")
    if byteorder not in ("little", "big"):
        raise ValueError(f"byteorder must be 'little' or 'big', got {byteorder!r}")
    long_size = long_size or ctypes.sizeof(ctypes.c_long)
    ctx = _Ctx(long_size)
    pre = _Preprocessor(defines)
    toks = _fold_attributes(pre.run(text, ctx))
    _Parser(toks, ctx).parse_all()

    builder = _Builder(ctx, byteorder, align)
    result = HeaderTypes()
    result.constants = dict(ctx.constants)
    for name in list(ctx.structs):
        if name.startswith("_anon"):
            continue
        try:
            result.structs[name] = result[name] = builder.struct(name)
        except _Skip as e:
            result.skipped[name] = str(e)
    for name in list(ctx.enums):
        if name.startswith("_anon"):
            continue
        try:
            cls = builder.enum(name)[0]
            result.enums[name] = result[name] = cls
        except (_Skip, TypeError, ValueError) as e:
            result.skipped[name] = str(e)
    for name, td in ctx.typedefs.items():
        if td.ptr or td.dims or name in result or name in result.skipped:
            continue
        try:
            base, ptr, dims = builder.lower((name,))
        except _Skip:
            continue
        if ptr or dims or base[0] not in ("struct", "enum"):
            continue
        if base[1] in result:
            result[name] = result[base[1]]
        elif base[1] in result.skipped:
            result.skipped[name] = result.skipped[base[1]]
    return result
