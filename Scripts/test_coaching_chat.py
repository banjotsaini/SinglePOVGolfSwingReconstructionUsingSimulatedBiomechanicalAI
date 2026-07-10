"""Offline tests for coaching_chat.py — no ANTHROPIC_API_KEY, no network.

Drives the FULL tool-use loop with ScriptedBackend (canned model turns) to prove:
tool dispatch, multi-turn memory, refusal-by-tool-result, the compare tool, the
max-iteration cap, and the transcript grounding verifier. Run:

    .venv/Scripts/python.exe Scripts/test_coaching_chat.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import coaching_chat as C

ROOT = Path(__file__).parent.parent
SC_A = ROOT / "Data" / "demo" / "1292" / "1292_scorecard.json"

_PASS = _FAIL = 0


def check(name: str, cond: bool, extra: str = ""):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
        print(f"  PASS  {name}")
    else:
        _FAIL += 1
        print(f"  FAIL  {name}  {extra}")


# -- scripted-turn builders ------------------------------------------------- #
def use(name, inp, tid="t1"):
    return ([{"type": "tool_use", "id": tid, "name": name, "input": inp}], "tool_use")


def say(text):
    return ([{"type": "text", "text": text}], "end_turn")


def ctx_single():
    return C.SwingContext.from_files(SC_A)


def ctx_pair():
    # use the same clip as "earlier" so compare is deterministic (delta 0)
    return C.SwingContext(a=json.loads(SC_A.read_text()), kb=C.v2.load_kb(),
                          b=json.loads(SC_A.read_text()))


# =========================================================================== #
print("\n[1] Tools return grounded data over a real scorecard")
ctx = ctx_single()
li = C._t_list_indicators(ctx, {})
check("list_indicators returns all 15", li["count"] == 15)
key = li["indicators"][0]["key"]
gi = C._t_get_indicator(ctx, {"key": "tempo_ratio"})
check("get_indicator(tempo_ratio) measured", gi.get("measured") and "value" in gi)
check("get_indicator unknown -> measured False",
      C._t_get_indicator(ctx, {"key": "nope"}).get("measured") is False)
gt = C._t_explain_term(ctx, {"term": "tour range"})
check("explain_term known term", gt["in_glossary"] is True)
check("explain_term unknown term", C._t_explain_term(ctx, {"term": "banana"})["in_glossary"] is False)
fl = C._t_get_flagged_observations(ctx, {})
check("get_flagged_observations returns the weight-shift flag", fl["count"] >= 1)
sm = C._t_get_swing_summary(ctx, {})
check("summary counts add up", sm["n_indicators"] == 15 and sm["n_flagged"] >= 1)

# find a low-confidence metric to drive the refusal tests
lowconf = [i["key"] for i in li["indicators"] if not i["reliable"]]
check("at least one low-confidence indicator exists", len(lowconf) >= 1, str(lowconf))
LC = lowconf[0] if lowconf else "left_arm_bend_top_deg"
check("get_indicator low-conf flags reliable False",
      C._t_get_indicator(ctx, {"key": LC}).get("reliable") is False)

# =========================================================================== #
print("\n[2] Full loop: answerable question fetches the metric, then answers")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}),
    say("Your tempo looks tour-like — the backswing-to-downswing rhythm is right in the typical range."),
])
convo = C.Conversation(ctx, backend)
res = convo.ask("How was my tempo?")
check("one tool call logged", len(res.tool_log) == 1 and res.tool_log[0]["name"] == "get_indicator")
check("loop stopped on end_turn", res.stopped_reason == "end_turn")
check("answer non-empty", bool(res.answer))
g = C.verify_chat_grounding(ctx, res)
check("answerable turn is grounded", g["grounded"], str(g["violations"]))
check("tempo_ratio recorded as fetched-reliable", "tempo_ratio" in g["fetched_reliable"])

# =========================================================================== #
print("\n[3] Refusal: unmeasured topic -> list, then no number")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("list_indicators", {}),
    say("I can't tell how far the ball went — ball flight isn't something this swing analysis measures."),
])
res = C.Conversation(ctx, backend).ask("How far did the ball go?")
g = C.verify_chat_grounding(ctx, res)
check("unmeasured refusal is grounded (no range claim)", g["grounded"], str(g["violations"]))
check("no indicator fetched", g["fetched_reliable"] == [])

# =========================================================================== #
print("\n[4] Refusal: low-confidence metric is flagged but not leaked")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": LC}),
    say("That measurement isn't reliable enough from a single camera, so I can't assess it."),
])
res = C.Conversation(ctx, backend).ask("Was my lead arm straight at the top?")
g = C.verify_chat_grounding(ctx, res)
check("clean low-conf refusal is grounded", g["grounded"], str(g["violations"]))
check("low-conf key recorded", LC in g["fetched_low_conf"])

# leak case: the model wrongly states the low-conf metric's value/label
label = C.v2.load_kb()["indicators"].get(LC, {}).get("label", LC)
backend = C.ScriptedBackend([
    use("get_indicator", {"key": LC}),
    say(f"Your {label} was well within the tour range at the top."),
])
res = C.Conversation(ctx_single(), backend).ask("Was my lead arm straight at the top?")
g = C.verify_chat_grounding(ctx_single(), res)
check("leaking the low-conf metric is caught",
      any(v["type"] == "low_confidence_leak" for v in g["violations"]), str(g["violations"]))

# NAMING the low-conf metric to DECLINE it (curly apostrophe, no value/range) is fine
backend = C.ScriptedBackend([
    use("get_indicator", {"key": LC}),
    say(f"{label} here is low confidence from a single camera, so I can’t assess it."),
])
g = C.verify_chat_grounding(ctx_single(), C.Conversation(ctx_single(), backend).ask("lead arm?"))
check("naming a low-conf metric to decline is NOT a leak", g["grounded"], str(g["violations"]))
# and the value must never be stated even while declining
val = C._t_get_indicator(ctx_single(), {"key": LC})["value"]
backend = C.ScriptedBackend([use("get_indicator", {"key": LC}), say(f"It measured {val} at the top.")])
g = C.verify_chat_grounding(ctx_single(), C.Conversation(ctx_single(), backend).ask("lead arm?"))
check("stating the low-conf value IS caught",
      any(v["type"] == "low_confidence_leak" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[5] Prescriptive answer is caught by the verifier")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "hip_lateral_shift_pct"}),
    say("Your weight shift is limited; you should work on driving toward the target."),
])
res = C.Conversation(ctx, backend).ask("What should I do about my weight shift?")
g = C.verify_chat_grounding(ctx, res)
check("prescriptive language is flagged",
      any(v["type"] == "prescriptive" for v in g["violations"]), str(g["violations"]))
# but a REFUSAL that merely names fix-words must NOT be flagged prescriptive
backend = C.ScriptedBackend([say("I can describe your swing, but I can't tell you what to work on or drill.")])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx_single(), backend).ask("what should I fix?"))
check("refusal echoing fix-words is NOT prescriptive",
      not any(v["type"] == "prescriptive" for v in g["violations"]), str(g["violations"]))
# "I can't prescribe... but here's what stood out" + grounded numbers is fine (real Codex case)
v = C._t_get_indicator(ctx_single(), {"key": "hip_lateral_shift_pct"})["value"]
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "hip_lateral_shift_pct"}),
    say(f"I can’t prescribe what to change or practice, but your weight shift was {v}% — below the tour range."),
])
g = C.verify_chat_grounding(ctx_single(), C.Conversation(ctx_single(), backend).ask("what should I work on?"))
check("describe-after-declining-to-prescribe is grounded", g["grounded"], str(g["violations"]))

# =========================================================================== #
print("[5b] Broad 'biggest takeaways' answer over many tools stays grounded")
ctx = ctx_single()
# a good summary calls several tools then states multiple fetched numbers
tempo = C._t_get_indicator(ctx, {"key": "tempo_ratio"})["value"]
sh = C._t_get_indicator(ctx, {"key": "shoulder_turn_top_deg"})["value"]
backend = C.ScriptedBackend([
    ([{"type": "tool_use", "id": "1", "name": "get_swing_summary", "input": {}},
      {"type": "tool_use", "id": "2", "name": "get_flagged_observations", "input": {}}], "tool_use"),
    ([{"type": "tool_use", "id": "3", "name": "get_indicator", "input": {"key": "tempo_ratio"}},
      {"type": "tool_use", "id": "4", "name": "get_indicator", "input": {"key": "shoulder_turn_top_deg"}}], "tool_use"),
    say(f"Biggest takeaways: your tempo is {tempo} and shoulder turn {sh}, both tour-like; weight shift was the one outside the range."),
])
res = C.Conversation(ctx, backend).ask("What are the biggest takeaways from my swing?")
g = C.verify_chat_grounding(ctx, res)
check("multi-tool summary is grounded", g["grounded"], str(g["violations"]))
check("summary used 4 tool calls across 2 rounds", len(res.tool_log) == 4)

# =========================================================================== #
print("\n[6] Multi-turn memory: follow-up keeps history")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say("Your shoulder turn is a touch under the tour range at the top."),
    use("get_indicator", {"key": "hip_turn_top_deg"}),
    say("Your hip turn, by contrast, is within the typical range."),
])
convo = C.Conversation(ctx, backend)
convo.ask("How was my shoulder turn?")
res2 = convo.ask("And my hips?")
# after 2 user turns + tool rounds, history should contain both user questions
user_turns = [m for m in convo.messages if m["role"] == "user" and isinstance(m["content"], str)]
check("both user questions in history", len(user_turns) == 2)
check("second turn fetched hip_turn", any(e["input"].get("key") == "hip_turn_top_deg" for e in res2.tool_log))

# =========================================================================== #
print("\n[7] compare_indicator only exists with a 2nd clip; works when present")
ctx1 = ctx_single()
check("no compare tool for single clip",
      "compare_indicator" not in [t["name"] for t in C.tool_specs(with_compare=False)])
ctxp = ctx_pair()
specs = [t["name"] for t in C.Conversation(ctxp, C.ScriptedBackend([])).tools]
check("compare tool offered with 2 clips", "compare_indicator" in specs)
cmp = C._t_compare_indicator(ctxp, {"key": "tempo_ratio"})
check("compare returns delta + direction", cmp["comparable"] and "direction" in cmp)
cmp_nokey = C._t_compare_indicator(ctx1, {"key": "tempo_ratio"})
check("compare without 2nd clip refuses", cmp_nokey["comparable"] is False)

# =========================================================================== #
print("\n[8] Max-iteration cap forces a final answer (and keeps message shape legal)")
ctx = ctx_single()
# script: keep calling tools forever -> loop must cut it off and force an answer
spam = [use("list_indicators", {}) for _ in range(C.MAX_TOOL_ITERS)]
spam.append(say("final answer after the cap"))
convo = C.Conversation(ctx, C.ScriptedBackend(spam))
res = convo.ask("loop please")
check("loop respected the iteration cap", res.iterations <= C.MAX_TOOL_ITERS)
check("a final answer was still produced", bool(res.answer))
# no two consecutive user messages (Anthropic requires alternation / merged tool_results)
roles = [m["role"] for m in convo.messages]
no_double_user = all(not (roles[i] == roles[i+1] == "user") for i in range(len(roles)-1))
check("no two consecutive user messages after cap", no_double_user, str(roles))

# =========================================================================== #
print("\n[9] Outgoing message shape: tool_result follows tool_use with matching id")
ctx = ctx_single()
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}, tid="abc123"),
    say("Your tempo is fine."),
])
C.Conversation(ctx, backend).ask("tempo?")
# the 2nd create() call must have received: ...assistant(tool_use) then user(tool_result id=abc123)
sent = backend.calls[1]["messages"]
asst = [m for m in sent if m["role"] == "assistant"][-1]
tu = [b for b in asst["content"] if b["type"] == "tool_use"][0]
tr_msg = sent[-1]
tr = tr_msg["content"][0]
check("last sent message is a user tool_result", tr_msg["role"] == "user" and tr["type"] == "tool_result")
check("tool_result id matches the tool_use id", tr["tool_use_id"] == tu["id"] == "abc123")

# =========================================================================== #
print("\n[10] Multiple tool calls in one assistant turn are all dispatched")
ctx = ctx_single()
multi = ([{"type": "tool_use", "id": "a", "name": "get_indicator", "input": {"key": "tempo_ratio"}},
          {"type": "tool_use", "id": "b", "name": "get_indicator", "input": {"key": "shoulder_turn_top_deg"}}],
         "tool_use")
backend = C.ScriptedBackend([multi, say("Both look good.")])
res = C.Conversation(ctx, backend).ask("tempo and shoulder turn?")
check("both tool calls logged", len(res.tool_log) == 2)
# the single user turn after must carry BOTH tool_results
tr_msg = backend.calls[1]["messages"][-1]
check("both tool_results in one user message", len(tr_msg["content"]) == 2
      and {b["tool_use_id"] for b in tr_msg["content"]} == {"a", "b"})

# =========================================================================== #
print("\n[11] Numeric grounding: an invented number is caught, a fetched one passes")
ctx = ctx_single()
real = C._t_get_indicator(ctx, {"key": "tempo_ratio"})["value"]
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}),
    say(f"Your tempo is {real}, right in the tour range."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("tempo?"))
check("stating the fetched value is grounded", g["grounded"], str(g["violations"]))
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "tempo_ratio"}),
    say("Your tempo is 9.87, right in the tour range."),  # invented value
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("tempo?"))
check("an invented number is flagged ungrounded",
      any(v["type"] == "ungrounded_number" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[12] Display rounding + dash ranges are grounded (live-endpoint repro)")
# live bug: "21 degrees" for a fetched 21.25 and "38–68" (en-dash) for pro_band
# [37.5, 68.1] were flagged as violations [21.0, 38.0, -68.0]
ctx = ctx_single()
ind = C._t_get_indicator(ctx, {"key": "shoulder_turn_top_deg"})
val, (lo, hi) = ind["value"], ind["pro_band"]
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say(f"Your shoulder turn was about {round(val)} degrees at the top; "
        f"the tour range is {round(lo)}–{round(hi)} degrees."),  # en-dash
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("shoulder turn?"))
check("rounded value + en-dash tour range is grounded", g["grounded"], str(g["violations"]))
check("en-dash never parsed as a minus sign",
      not any(v.get("value", 0) < 0 for v in g["violations"]), str(g["violations"]))
# em-dash and plain hyphen ranges behave the same
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say(f"Tour range: {round(lo)}—{round(hi)} degrees; you turned {round(val)}."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("shoulder turn?"))
check("em-dash range is grounded too", g["grounded"], str(g["violations"]))
# a genuinely negative number is still parsed as negative (sign not after a digit)
neg_toks = C._NUM_TOKEN.findall("swayed -3.1 inches, range 38-68")
check("standalone minus kept, range dash split",
      neg_toks == ["-3.1", "38", "68"], str(neg_toks))
# display rounding must NOT excuse an invented number
backend = C.ScriptedBackend([
    use("get_indicator", {"key": "shoulder_turn_top_deg"}),
    say(f"Your shoulder turn was {round(val) + 7} degrees at the top."),
])
g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("shoulder turn?"))
check("a number ~7 off the fetched value is still flagged",
      any(v["type"] == "ungrounded_number" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print("\n[13] Low-conf: tour band citable in a refusal; rounded value leak caught")
ctx = ctx_single()
lc_ind = C._t_get_indicator(ctx, {"key": LC})
lc_val, lc_band = lc_ind["value"], lc_ind.get("pro_band")
if lc_band:
    backend = C.ScriptedBackend([
        use("get_indicator", {"key": LC}),
        say(f"Tour pros are typically {round(lc_band[0])}–{round(lc_band[1])} here, but this "
            f"measurement isn't reliable enough from a single camera, so I can't assess yours."),
    ])
    g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("lead arm?"))
    check("citing the tour band of a low-conf metric in a refusal is grounded",
          g["grounded"], str(g["violations"]))
if abs(round(lc_val)) >= 13 and all(abs(round(lc_val) - round(b)) > 1 for b in (lc_band or [])):
    backend = C.ScriptedBackend([
        use("get_indicator", {"key": LC}),
        say(f"It measured about {round(lc_val)} at the top."),
    ])
    g = C.verify_chat_grounding(ctx, C.Conversation(ctx, backend).ask("lead arm?"))
    check("stating the ROUNDED low-conf value is still a leak",
          any(v["type"] == "low_confidence_leak" for v in g["violations"]), str(g["violations"]))

# =========================================================================== #
print(f"\n{'='*50}\n  {_PASS} passed, {_FAIL} failed\n{'='*50}")
sys.exit(1 if _FAIL else 0)
