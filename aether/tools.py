"""
tools.py — Agentic tool registry.

Tools are exposed to the cognitive loop as HD "trigger vectors" — when the
working memory matches a tool trigger above a threshold, the tool fires.

Built-in tools:
  - calc       : arithmetic expressions, parsed safely
  - time       : current date/time
  - recall     : search episodic memory by keyword
  - teach      : learn a triple (s, p, o) from user input
  - python     : run a sandboxed Python expression (whitelisted)
  - list_kb    : list known triples

Each tool returns a text result that gets encoded back into HD and added
to working memory — this is how the agent "perceives" its own tool output.
"""

from __future__ import annotations
import ast
import re
import operator as op
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple, Any

from .hd import HDVector, DIM
from .encoder import TextEncoder
from .memory import AssociativeMemory


# ---- safe arithmetic evaluator ----
_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
}


def safe_eval_arith(expr: str) -> float:
    """Evaluate an arithmetic expression safely. No names, no calls."""
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expression")
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        if type(node.op) not in _OPS:
            raise ValueError(f"unsupported operator: {type(node.op).__name__}")
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        if type(node.op) not in _OPS:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


# ---- tool definitions ----
ToolFn = Callable[[str, "ToolContext"], str]


class ToolContext:
    """Passed to every tool invocation. Gives access to the agent's memory."""
    def __init__(self, encoder: TextEncoder, assoc: AssociativeMemory):
        self.encoder = encoder
        self.assoc = assoc


def tool_calc(args: str, ctx: ToolContext) -> str:
    try:
        result = safe_eval_arith(args)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return f"{args} = {result}"
    except Exception as e:
        return f"calc error: {e}"


def tool_time(args: str, ctx: ToolContext) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def tool_recall(args: str, ctx: ToolContext) -> str:
    """Search episodic memory for episodes matching the keywords."""
    if not args.strip():
        return "recall: empty query"
    query_vec = ctx.encoder.encode_text(args)
    results = ctx.assoc.retrieve_similar(query_vec, top_k=3)
    if not results:
        return "recall: nothing found"
    return " ; ".join(f"{t}({s:.2f})" for t, s in results)


def tool_teach(args: str, ctx: ToolContext) -> str:
    """Teach a triple: 'subject | predicate | object'."""
    parts = [p.strip() for p in args.split("|")]
    if len(parts) != 3:
        parts = [p.strip() for p in args.split(",")]
    if len(parts) != 3:
        # Try "subject is predicate object" / "X is Y"
        lower = args.lower()
        if " is " in lower:
            idx = lower.find(" is ")
            subj = args[:idx].strip()
            obj = args[idx + 4:].strip()
            ctx.assoc.learn_triple(subj, "is_a", obj)
            # Also learn the episode
            ctx.assoc.add_episode(args, ctx.encoder.encode_text(args))
            ctx.encoder.learn_sequence(args)
            return f"learned: {subj} is_a {obj}"
        return "teach: expected 'subject | predicate | object' or 'X is Y'"
    s, p, o = parts
    ctx.assoc.learn_triple(s, p, o)
    ctx.assoc.add_episode(args, ctx.encoder.encode_text(args))
    ctx.encoder.learn_sequence(args)
    return f"learned: ({s}, {p}, {o})"


def tool_list_kb(args: str, ctx: ToolContext) -> str:
    triples = ctx.assoc.list_triples()
    if not triples:
        return "KB is empty"
    return " ; ".join(f"({s},{p},{o})" for s, p, o in triples)


def tool_python(args: str, ctx: ToolContext) -> str:
    """Sandboxed Python: only arithmetic + a few safe builtins."""
    # Reuse arithmetic eval first; allow print-free expressions
    try:
        # Only allow arithmetic and string literals
        tree = ast.parse(args, mode="eval")
        return str(_eval_safe(tree.body))
    except Exception as e:
        return f"python error: {e}"


def _eval_safe(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS.get(type(node.op))(_eval_safe(node.left), _eval_safe(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS.get(type(node.op))(_eval_safe(node.operand))
    if isinstance(node, ast.List):
        return [_eval_safe(e) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_safe(e) for e in node.elts)
    raise ValueError(f"disallowed node: {type(node).__name__}")


def tool_compare(args: str, ctx: ToolContext) -> str:
    """Compare two entities by retrieving all known facts about each."""
    parts = [p.strip() for p in re.split(r"\s+(?:and|vs\.?|versus)\s+", args, maxsplit=1)]
    if len(parts) != 2:
        return "compare: expected 'X and Y' or 'X vs Y'"
    x, y = parts[0].lower(), parts[1].lower()
    x_facts = [(s, p, o) for s, p, o in ctx.assoc.list_triples() if s.lower() == x]
    y_facts = [(s, p, o) for s, p, o in ctx.assoc.list_triples() if s.lower() == y]
    if not x_facts and not y_facts:
        return f"compare: no facts about either {x} or {y}"
    lines = [f"Comparison of {x} and {y}:"]
    # Find shared predicates
    x_preds = {p: o for _, p, o in x_facts}
    y_preds = {p: o for _, p, o in y_facts}
    shared = set(x_preds.keys()) & set(y_preds.keys())
    for p in shared:
        same = "SAME" if x_preds[p].lower() == y_preds[p].lower() else "DIFFERENT"
        lines.append(f"  {p}: {x}={x_preds[p]}, {y}={y_preds[p]}  [{same}]")
    only_x = set(x_preds.keys()) - set(y_preds.keys())
    only_y = set(y_preds.keys()) - set(x_preds.keys())
    for p in only_x:
        lines.append(f"  {p}: only {x} has this -> {x_preds[p]}")
    for p in only_y:
        lines.append(f"  {p}: only {y} has this -> {y_preds[p]}")
    return "\n".join(lines)


def tool_explain(args: str, ctx: ToolContext) -> str:
    """Explain a concept by retrieving all known facts about it."""
    subject = args.strip().lower()
    if not subject:
        return "explain: expected a subject"
    facts = [(s, p, o) for s, p, o in ctx.assoc.list_triples() if s.lower() == subject]
    if not facts:
        return f"explain: I don't know anything about {subject}"
    lines = [f"Here's what I know about {subject}:"]
    for s, p, o in facts:
        # Human-readable rendering
        if p == "capital_of":
            lines.append(f"  - It is the capital of {o}.")
        elif p == "located_in":
            lines.append(f"  - It is located in {o}.")
        elif p == "is_a":
            lines.append(f"  - It is a {o}.")
        else:
            lines.append(f"  - {p}: {o}")
    return "\n".join(lines)


def tool_summarize(args: str, ctx: ToolContext) -> str:
    """Summarize N most recent episodes."""
    try:
        n = int(args.strip()) if args.strip() else 5
    except ValueError:
        n = 5
    n = max(1, min(n, 20))
    episodes = ctx.assoc.episodes[-n:]
    if not episodes:
        return "summarize: nothing to summarize"
    lines = [f"Summary of last {len(episodes)} memories:"]
    for i, ep in enumerate(episodes, 1):
        # Truncate long episodes
        text = ep.payload if len(ep.payload) < 80 else ep.payload[:77] + "..."
        lines.append(f"  {i}. {text}")
    return "\n".join(lines)


def tool_translate(args: str, ctx: ToolContext) -> str:
    """Translate a sentence using the KB (word-by-word via is_a / synonym).

    Format: 'lang:EN>FR sentence' or just 'sentence' for EN>FR default.
    """
    args = args.strip()
    if not args:
        return "translate: nothing to translate"
    # Find tokens in KB with a 'translation' predicate
    tokens = re.findall(r"[A-Za-zÀ-ÿ0-9]+|[^\sA-Za-zÀ-ÿ0-9]", args)
    translated = []
    for tok in tokens:
        result = ctx.assoc.query_triple(tok.lower(), "translation")
        if result:
            translated.append(result[0])
        else:
            translated.append(tok)
    return " ".join(translated)


def tool_define(args: str, ctx: ToolContext) -> str:
    """Define a word by looking up its is_a predicate."""
    subject = args.strip().lower()
    if not subject:
        return "define: expected a subject"
    result = ctx.assoc.query_triple(subject, "is_a")
    if result:
        ans, sim = result
        return f"{subject} is {ans} (confidence: {sim:.2f})"
    return f"define: I don't have a definition for {subject}"


def tool_forget(args: str, ctx: ToolContext) -> str:
    """Forget a fact (for testing — does not actually delete from SDM but
    removes from the triples list and re-seeds the vocab).
    """
    subject = args.strip().lower()
    if not subject:
        return "forget: expected a subject"
    before = len(ctx.assoc.triples)
    ctx.assoc.triples = [(s, p, o) for s, p, o in ctx.assoc.triples if s.lower() != subject]
    removed = before - len(ctx.assoc.triples)
    return f"forgot {removed} fact(s) about {subject}"


def tool_count(args: str, ctx: ToolContext) -> str:
    """Count things in the KB: count triples, count vocab, count episodes."""
    what = args.strip().lower()
    if what in ("triples", "facts", "kb"):
        return f"{len(ctx.assoc.triples)} triples in KB"
    if what in ("vocab", "tokens", "words"):
        return f"{len(ctx.assoc.vocab)} tokens in vocabulary"
    if what in ("episodes", "memories", "history"):
        return f"{len(ctx.assoc.episodes)} episodes stored"
    return f"count: unknown target '{what}'. Try: triples, vocab, episodes"


# ---- registry ----
class ToolRegistry:
    """Maps tool names -> (function, trigger_vector, trigger_threshold).

    The trigger vector is the HD encoding of the tool's natural-language
    description. The cognitive loop matches working memory against these
    to decide when to fire a tool.
    """

    def __init__(self, encoder: TextEncoder):
        self.encoder = encoder
        self.tools: Dict[str, ToolFn] = {}
        self.triggers_: Dict[str, Tuple[HDVector, float]] = {}
        self.descriptions: Dict[str, str] = {}

    def register(self, name: str, description: str, fn: ToolFn, threshold: float = 0.18) -> None:
        self.tools[name] = fn
        self.descriptions[name] = description
        trig_vec = self.encoder.encode_text(description)
        self.triggers_[name] = (trig_vec, threshold)

    def triggers(self):
        """Yield (name, trigger_vector, threshold) for the cognitive loop."""
        for name, (vec, thr) in self.triggers_.items():
            yield name, vec, thr

    def call(self, name: str, args: str, ctx: ToolContext) -> str:
        if name not in self.tools:
            return f"unknown tool: {name}"
        return self.tools[name](args, ctx)

    def match(self, wm: HDVector) -> Optional[Tuple[str, float, str]]:
        """Find the best matching tool for a working-memory vector.

        Returns (name, similarity, args_hint) or None.
        """
        best_name, best_sim = None, -1.0
        for name, (vec, thr) in self.triggers_.items():
            sim = wm.similarity(vec)
            if sim > best_sim:
                best_sim, best_name = sim, name
        if best_name is None:
            return None
        return best_name, best_sim, self.descriptions[best_name]


def default_tools(encoder: TextEncoder) -> ToolRegistry:
    """Register the default AETHER toolset (v2 — extended)."""
    reg = ToolRegistry(encoder)
    reg.register("calc",     "calculate arithmetic compute plus minus times divide",       tool_calc,     threshold=0.14)
    reg.register("time",     "what time date now today current",                           tool_time,     threshold=0.18)
    reg.register("recall",   "remember recall memory search find episode",                 tool_recall,   threshold=0.16)
    reg.register("teach",    "learn remember teach fact know store knowledge",             tool_teach,    threshold=0.14)
    reg.register("list_kb",  "list show knowledge facts triples database",                 tool_list_kb,  threshold=0.16)
    reg.register("python",   "evaluate python expression code",                            tool_python,   threshold=0.18)
    reg.register("compare",  "compare versus difference between two entities",             tool_compare,  threshold=0.16)
    reg.register("explain",  "explain describe concept tell me about what is",             tool_explain,  threshold=0.14)
    reg.register("summarize","summarize summary recent memories episodes",                 tool_summarize,threshold=0.16)
    reg.register("translate","translate translation language",                             tool_translate,threshold=0.18)
    reg.register("define",   "define definition what is meaning",                          tool_define,   threshold=0.14)
    reg.register("forget",   "forget delete remove fact",                                  tool_forget,   threshold=0.18)
    reg.register("count",    "count how many number of",                                   tool_count,    threshold=0.16)
    return reg
