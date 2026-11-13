import json
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import ask

async def main():
    gold_path = Path(__file__).parent / "gold.jsonl"
    result_path = Path(__file__).parent / "results" / "week1_eval.jsonl"
    
    # 关键修改：使用 utf-8-sig 读取，自动忽略 BOM 标记
    rows = [json.loads(line) for line in open(gold_path, encoding="utf-8-sig") if line.strip()]
    
    hit_count = 0
    total_latency = 0
    
    with open(result_path, "w", encoding="utf-8") as out:
        for r in rows:
            print(f"Testing {r['id']}...")
            pred = await ask(r["question"])
            ans = pred["answer"].lower()
            
            must = [k.lower() in ans for k in r.get("must_include", [])]
            forbid = [k.lower() in ans for k in r.get("forbid", [])]
            
            hit = all(must) and not any(forbid)
            
            if hit:
                hit_count += 1
            total_latency += pred["latency_ms"]
                
            out.write(json.dumps({
                "id": r["id"],
                "question": r["question"],
                "hit": hit,
                "latency_ms": pred["latency_ms"],
                "answer": pred["answer"]
            }, ensure_ascii=False) + "\n")
            
            print(f"  -> Hit: {hit}, Latency: {pred['latency_ms']}ms")

    hit_rate = hit_count / len(rows) if rows else 0
    avg_latency = total_latency / len(rows) if rows else 0
    
    print("\n" + "="*30)
    print(f"评测完成！总题数: {len(rows)}")
    print(f"Hit Rate: {hit_rate * 100:.1f}%")
    print(f"平均延迟: {avg_latency:.0f} ms")
    print("="*30)

if __name__ == "__main__":
    asyncio.run(main())
