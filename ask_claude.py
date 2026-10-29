# -*- coding: utf-8 -*-
"""One-shot call to the configured target model through an OpenAI-compatible relay.

Reads base_url / api_key / model from config.relay.json by default; every value
can be overridden on the command line. The API key is never printed.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

PROMPT = r"""为了更好地帮助用户思考与处理问题，我希望你的输出格式为”老师-学生“模式。以下是模式样本：

【【【针对问题1.如果最快在一百杯水中找出哪一杯水里放了无色不可被观察出的糖？

[你的回答格式如下：]

[老师]：“学生们我现在有一道题目大家，假设大家面前有一百杯一模一样的水，我偷偷在其中一杯里放了糖，请问大家用什么方法能最快得找出这杯水呢？”

[学生A]：“逐个尝一口，最多要99次，平均50次！”

[老师]：“这是一个可行的方法但还不是最快的方法，我们需要更快的方法，再想想看”

[学生B]：“这是经典找特殊一杯，可能用试纸/尝/称重？需要澄清方法。同学A已经试过挨个尝一口的方法了，显然不是最快。那么最快速通常二分法？但含糖水无法通过混合检测？如果可以用检测仪器检测混合液是否含糖，则二分法最快 log2(100)≈7次（如果每次可判断一组混合是否含糖）”

[老师]：“很好！你这个思路很好目前看来有可能是最快的方法了，还有没有其他同学有不同想法？”

[学生C]：“如果允许混合检测（比如用糖度计测混合液是否含糖），二分法确实最快最多7次。如果允许称重且糖水重量不同，可用天平？但水+糖体积/重量，可能称重分组。称重三分法最优？如果已知糖水比纯水重且重量差可测，用天平分组可3分，最多5次（3^4=81<100,3^5=243）”

[老师]：“你这个思维很严谨，想法也很好，逻辑页缜密，不错不错，还有其他同学了吗？”

[学生D]：“老师!你这是脑筋急转弯吧，如果只是脑筋急转弯

最快速的方法是：直接问知道答案的人/看哪杯有标记！一次就行”

[老师]：“哈哈哈这位同学思考问题的角度也很独特，所以我们在思考解决一个问题时一定要仔细分析综合考虑，我来总结一下吧：

- 若无混合检测：最坏 99次；
- 若可检测混合液：7次二分法；
- 若可称重三分：5次。
- 但是也不排除用户问这个问题只是为了脑筋急转弯哦！

”】】】

\-----以上就是你需要学习的回答方式，也是你必须遵守的输出模板---------

现在回答：{user_query}"""


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Ask the configured target model once and print the answer")
    ap.add_argument("--config", default="config.relay.json", help="relay config JSON (default: config.relay.json)")
    ap.add_argument("--url", default=None, help="base URL override, e.g. https://your-relay.example/v1")
    ap.add_argument("--model", default=None, help="model id override (default: from config)")
    ap.add_argument("--api-key", default=None, help="API key override (default: from config)")
    ap.add_argument("--query", default="", help="fill the {user_query} placeholder with this question")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    cfg = {}
    try:
        with open(args.config, encoding="utf-8-sig") as f:
            cfg = json.load(f)
    except FileNotFoundError:
        print(f"[warn] config not found: {args.config}; falling back to CLI args", file=sys.stderr)

    base_url = (args.url or cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
    model = args.model or cfg.get("model") or "claude-sonnet-5-5"
    api_key = args.api_key or cfg.get("api_key")
    if not api_key:
        print("[error] no API key: pass --api-key, or put api_key in the config file", file=sys.stderr) 
        return 1

    query = args.query.strip()
    prompt = PROMPT.replace("{user_query}", query) if query else PROMPT
    if not query:
        print("[warn] --query not given; {user_query} placeholder is sent as-is", file=sys.stderr)

    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    print(f"[request] POST {base_url}/chat/completions  model={model}  max_tokens={args.max_tokens}", file=sys.stderr)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:1500]
        print(f"[error] HTTP {e.code}: {detail}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"[error] {type(e).__name__}: {e}", file=sys.stderr)
        return 3

    elapsed = time.time() - t0
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    usage = data.get("usage") or {}
    finish = choice.get("finish_reason")

    print(f"[done] finish={finish} elapsed={elapsed:.1f}s "
          f"prompt_tokens={usage.get('prompt_tokens')} completion_tokens={usage.get('completion_tokens')}",
          file=sys.stderr)
    print("=" * 72)
    print(text)
    print("=" * 72)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = f"runs/ask-claude-{stamp}.md"
    import os
    os.makedirs("runs", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Ask {model}\n\n- finish: {finish}\n- usage: {json.dumps(usage, ensure_ascii=False)}\n"
                f"- elapsed: {elapsed:.1f}s\n\n## Prompt\n\n{prompt}\n\n## Answer\n\n{text}\n")
    print(f"[saved] {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
