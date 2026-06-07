#!/usr/bin/env python3
import json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

API_URL = "https://opencode.ai/zen/go/v1"
API_KEY = "sk-u6EN2W6TWpbEql89F3f0B6UO1tKnDASphEwScUV8oODvOJmmHtQdA68ydIvdsYG4"
MODEL = "deepseek-v4-flash"
DATASET_PATH = "./acg_simpleqa.jsonl"
OUTPUT_PATH = "./results_flash_full.json"

SYSTEM_PROMPT = (
    "你是ACG（动画漫画游戏）领域专家。"
    "请只输出答案本身，不要解释。"
    "如果不知道，只输出：我不知道"
)

def load(path):
    xs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                xs.append(json.loads(line))
    return xs

def ok(pred, gold):
    pred = pred.strip().lower()
    gold = gold.strip().lower()
    if not pred or pred == chr(25105)+chr(19981)+chr(30693)+chr(36947):
        return False
    if gold in pred:
        return True
    if len(pred) >= 2 and pred in gold:
        return True
    return False

def ask(client, q):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":q}],
                temperature=0.0,
                max_tokens=256,
                extra_body={"thinking":{"type":"disabled"}},
            )
            c = resp.choices[0].message.content
            return c.strip() if c else ""
        except Exception:
            if attempt < 2:
                time.sleep(1)
    return ""

examples = load(DATASET_PATH)
total = len(examples)
print(f"加载 {total} 条样本，开始 V4 Flash 全量评测（不开思考）...")

client = OpenAI(base_url=API_URL, api_key=API_KEY, timeout=60)
results = [None] * total
correct = wrong = idk = errors = 0

def process(i, ex):
    q, gold = ex["question"], ex["answer"]
    pred = ask(client, q)
    c = ok(pred, gold)
    is_idk = pred == chr(25105)+chr(19981)+chr(30693)+chr(36947)
    return i, {
        "category": ex["category"],
        "question": q,
        "answer": gold,
        "urls": ex.get("urls",[]),
        "flash_pred": pred,
        "flash_correct": c,
        "flash_idk": is_idk,
    }

with ThreadPoolExecutor(max_workers=20) as exe:
    futs = {exe.submit(process, i, ex): i for i, ex in enumerate(examples)}
    with tqdm(total=total, desc="V4 Flash 全量", unit="题") as pbar:
        for fut in as_completed(futs):
            try:
                idx, res = fut.result()
            except:
                idx = futs[fut]
                ex = examples[idx]
                res = {"category":ex["category"],"question":ex["question"],"answer":ex["answer"],"urls":ex.get("urls",[]),"flash_pred":"[ERROR]","flash_correct":False,"flash_idk":False}
                errors += 1
            results[idx] = res
            if res["flash_correct"]:
                correct += 1
            elif res["flash_idk"]:
                idk += 1
            else:
                wrong += 1
            pbar.set_postfix({"对":correct,"错":wrong,"不知":idk,"err":errors})
            pbar.update(1)

out = {
    "config": {"model":MODEL,"api_url":API_URL,"total":total,"thinking_disabled":True},
    "metrics": {"total":total,"correct":correct,"wrong":wrong,"idk":idk,"errors":errors,"correct_rate":round(correct/total,4),"wrong_rate":round(wrong/total,4),"idk_rate":round(idk/total,4)},
    "details": results,
}
with open(OUTPUT_PATH,"w",encoding="utf-8") as f:
    json.dump(out,f,ensure_ascii=False,indent=2)

print(f"
=== V4 Flash 全量评测完成 ===")
print(f"总题数: {total}")
print(f"正确: {correct} ({correct/total:.2%})")
print(f"错误: {wrong} ({wrong/total:.2%})")
print(f"不知道: {idk} ({idk/total:.2%})")
print(f"异常: {errors}")
print(f"结果: {OUTPUT_PATH}")
