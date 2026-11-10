#!/usr/bin/env python3
"""Image cases for strix-serve-deepseek-chat-tests, from the model release's reference encoder (encoding/encoding.py).

Each case: the OpenAI-format request, the prompt text the reference renders (image blocks become the
<｜deepseek_image｜> placeholder) and the image URLs in prompt order (encode_messages(..., return_multi_modal_data)).

    python3 tools/ds41/make_chat_image_cases.py --reference DeepSeek-V4.1-Flash-1017/encoding \
        tests/data/serve/deepseek_v41_chat_image_cases.json
"""
import argparse
import importlib.util
import json

IMG = lambda n: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,IMG{n}"}}
TXT = lambda t: {"type": "text", "text": t}
TOOLS = [{"type": "function", "function": {"name": "screenshot", "description": "Take a screenshot", "parameters": {"type": "object", "properties": {"url": {"type": "string"}}}}}]

CASES = [
    {"messages": [{"role": "user", "content": [TXT("What is in this picture?"), IMG(1)]}], "thinking": False},
    {"messages": [{"role": "user", "content": [IMG(1), TXT("Compare with"), IMG(2), TXT("Which is larger?")]}], "thinking": True},
    {"messages": [{"role": "user", "content": [IMG(1)]}], "thinking": True},
    {"messages": [{"role": "system", "content": "You are a UI tester."},
                  {"role": "user", "content": "Open the page and check the layout."},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "screenshot", "arguments": "{\"url\": \"http://x\"}"}}]},
                  {"role": "tool", "tool_call_id": "c1", "content": [TXT("Screenshot of http://x:"), IMG(1)]}],
     "tools": TOOLS, "thinking": True},
    {"messages": [{"role": "user", "content": "Take two screenshots."},
                  {"role": "assistant", "content": "", "tool_calls": [
                      {"id": "a", "type": "function", "function": {"name": "screenshot", "arguments": {"url": "http://a"}}},
                      {"id": "b", "type": "function", "function": {"name": "screenshot", "arguments": {"url": "http://b"}}}]},
                  {"role": "tool", "tool_call_id": "b", "content": [IMG(2), TXT("page b")]},
                  {"role": "tool", "tool_call_id": "a", "content": [TXT("page a"), IMG(1)]},
                  {"role": "user", "content": [TXT("And this one:"), IMG(3)]}],
     "tools": TOOLS, "thinking": False},
    {"messages": [{"role": "user", "content": [TXT("first part"), TXT("second part")]},
                  {"role": "assistant", "content": [TXT("answer one"), TXT("answer two")]},
                  {"role": "user", "content": "thanks"}], "thinking": False},
    {"messages": [{"role": "system", "content": [TXT("rule one"), TXT("rule two")]},
                  {"role": "user", "content": [IMG(1), TXT("describe")]},
                  {"role": "assistant", "content": "A red circle."},
                  {"role": "user", "content": [TXT("now this"), IMG(2)]}], "thinking": True},
    {"messages": [{"role": "user", "content": "run it"},
                  {"role": "assistant", "content": "", "tool_calls": [{"id": "t", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]},
                  {"role": "tool", "tool_call_id": "t", "content": [TXT("line one"), TXT("line two")]}],
     "tools": TOOLS, "thinking": True},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True, help="the release's encoding/ directory")
    ap.add_argument("out")
    a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("encoding", f"{a.reference}/encoding.py")
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    out = []
    for c in CASES:
        msgs = json.loads(json.dumps(c["messages"]))
        if c.get("tools"):  # request-level tools live in the first system message (inserted when there is none), as the API does
            if msgs[0]["role"] == "system":
                msgs[0] = dict(msgs[0], tools=c["tools"])
            else:
                msgs.insert(0, {"role": "system", "content": "", "tools": c["tools"]})
        prompt, media = enc.encode_messages(msgs, thinking_mode="thinking" if c["thinking"] else "chat", return_multi_modal_data=True)
        urls = [im.get("url") for im in media["images"]]
        out.append({"messages": c["messages"], "tools": c.get("tools"), "thinking": c["thinking"], "reasoning_effort": None,
                    "preserve_thinking": False, "expected": prompt, "images": urls})
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(len(out), "cases")


if __name__ == "__main__":
    main()
