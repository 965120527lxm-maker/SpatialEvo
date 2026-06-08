import subprocess, sys, shutil
from pathlib import Path

lessons = [
    "01-hypergraph-construction",
    "02-graph-normalization", 
    "03-hgnn-layer",
    "04-mlp-encoder",
    "05-single-slice-model",
    "06-pseudo-spot-aggregation",
    "07-regression-translator",
    "08-six-loss-trainer",
    "09-graph-aware-evaluation",
    "10-graph-transformer-layer",
]

root = Path(__file__).parent.resolve()
results = []
for lesson in lessons:
    d = root / lesson
    starter = d / "starter.py"
    solution = d / "solution.py"
    test = d / "test.py"
    cache = d / "__pycache__"
    bak = d / "starter.py.bak"
    
    if not starter.exists():
        results.append((lesson, "SKIP", "no starter.py"))
        continue
    
    shutil.copy(starter, bak)
    shutil.copy(solution, starter)
    if cache.exists():
        shutil.rmtree(cache)
    
    try:
        res = subprocess.run(
            [sys.executable, str(test)],
            capture_output=True, text=True, timeout=30
        )
        ok = res.returncode == 0
        msg = (res.stdout + res.stderr).strip()
        results.append((lesson, "PASS" if ok else "FAIL", msg))
    except Exception as e:
        results.append((lesson, "ERROR", str(e)))
    finally:
        shutil.copy(bak, starter)
        bak.unlink()
        if cache.exists():
            shutil.rmtree(cache)

for lesson, status, msg in results:
    print(f"{lesson}: {status}")
    if status != "PASS":
        print(f"  {msg[:500]}")
