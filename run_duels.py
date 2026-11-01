"""Judge generated Three.js modules against each other with the official multi-stage GLM judge.

    python run_duels.py --images ../3D-API-Base-Worker/img/r0 \
        --contender base=../3D-API-Base-Worker/outputs/eval-r0/3D-API-Base-Worker \
        --contender gen=../3D-API-Base-Worker/outputs/eval-r0/3d-gen-api --out results/r0

A contender folder holds one module per prompt id, as `<id>/final.js` (local.py output) or `<id>.js`.
Without --leader every pair of contenders duels on every prompt; with --leader each other contender duels the
leader (left = challenger, right = leader) and passes when (wins - losses) / prompts >= --win-margin, as in the
competition's qualification. Bundles, judge traces and duel results are cached, so a rerun only does new work.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import sys
from collections import Counter
from pathlib import Path

import httpx
from loguru import logger

from arena.assets import AssetStore
from arena.bundle import Renderer, RenderSettings, build_bundle, bundle_key
from arena.dino import Dino
from arena.env import load_env
from arena.glm import FakeGlm, GlmClient
from judge_service.common.competition.generations import GenerationResult
from judge_service.judges.multi_stage import evaluate_duel

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"


def load_prompts(source: Path) -> dict[str, bytes]:
    if source.is_dir():
        return {p.stem: p.read_bytes() for p in sorted(source.glob("*.png"))}
    prompts = {}
    with httpx.Client(timeout=60, follow_redirects=True) as http:
        for line in source.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            stem = Path(line.split("?")[0]).stem
            cached = CACHE / "downloads" / f"{stem}.png"
            if line.startswith(("http://", "https://")):
                if not cached.exists():
                    cached.parent.mkdir(parents=True, exist_ok=True)
                    cached.write_bytes(http.get(line).raise_for_status().content)
                prompts[stem] = cached.read_bytes()
            else:
                prompts[stem] = (source.parent / line if not Path(line).is_absolute() else Path(line)).read_bytes()
    return prompts


def find_module(folder: Path, prompt_id: str) -> Path | None:
    for candidate in (folder / prompt_id / "final.js", folder / f"{prompt_id}.js"):
        if candidate.is_file():
            return candidate
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="folder of <id>.png, or a prompts.txt of paths/URLs")
    p.add_argument("--contender", action="append", required=True, metavar="NAME=FOLDER", help="repeat, at least 2")
    p.add_argument("--leader", help="contender every other one duels against (qualification mode)")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "run")
    p.add_argument("--seed", type=int, default=42, help="judge seed")
    p.add_argument("--win-margin", type=float, default=0.07)
    p.add_argument("--parallel", type=int, default=1, help="duels at once (the official judge runs them one by one)")
    p.add_argument("--max-vlm-calls", type=int, default=32, help="concurrent VLM calls (official MAX_CONCURRENT_VLM_CALLS)")
    p.add_argument("--limit", type=int, help="only the first N prompts")
    p.add_argument("--only", action="append", help="prompt id to judge (repeatable)")
    p.add_argument("--renderer-url", default=None, help="render-service-js URL (env RENDERER_URL)")
    p.add_argument("--img-size", type=int, default=518)
    p.add_argument("--lighting", default="neutral", choices=["neutral", "studio", "follow"])
    p.add_argument("--glm-url", default=None, help="OpenAI-compatible endpoint serving GLM (env GLM_BASE_URL)")
    p.add_argument("--glm-key", default=None, help="env GLM_API_KEY")
    p.add_argument("--glm-model", default=None, help="served model name (env GLM_MODEL, default glm-4.6v-flash)")
    p.add_argument("--dry-run", action="store_true", help="fake VLM answers: checks renders, embeddings and plumbing")
    return p.parse_args()


async def amain() -> int:
    load_env(ROOT / ".env")
    args = parse_args()
    contenders = dict(c.split("=", 1) for c in args.contender)
    contenders = {name: Path(folder) for name, folder in contenders.items()}
    if len(contenders) < 2 or (args.leader and args.leader not in contenders):
        sys.exit("need at least 2 contenders, and --leader must be one of them")
    missing = [name for name, folder in contenders.items() if not folder.is_dir()]
    if missing:
        sys.exit(f"contender folder not found: {', '.join(missing)}")

    args.out.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} {message}")
    logger.add(args.out / "judge.log", level="DEBUG")

    prompts = load_prompts(args.images)
    if args.only:
        prompts = {k: v for k, v in prompts.items() if k in set(args.only)}
    if args.limit:
        prompts = dict(list(prompts.items())[: args.limit])
    if not prompts:
        sys.exit("no prompt images found")

    assets = AssetStore(CACHE)
    if args.dry_run:
        vlm: GlmClient | FakeGlm = FakeGlm(assets)
        model = "fake"
    else:
        url = args.glm_url or os.environ.get("GLM_BASE_URL")
        if not url:
            sys.exit("set GLM_BASE_URL (or --glm-url) to the GLM endpoint, or use --dry-run")
        model = args.glm_model or os.environ.get("GLM_MODEL", "glm-4.6v-flash")
        vlm = GlmClient(assets, url, args.glm_key or os.environ.get("GLM_API_KEY", "EMPTY"), model, args.max_vlm_calls)
    try:
        logger.info(f"VLM preflight ({model}): {await vlm.preflight()!r}")
    except Exception as error:  # noqa: BLE001 - stop before judging with a dead endpoint
        sys.exit(f"VLM endpoint check failed: {type(error).__name__}: {error}")

    # Bundles: renders + embeddings for every contender module.
    settings = RenderSettings(url=args.renderer_url or os.environ.get("RENDERER_URL", "http://127.0.0.1:8000"),
                              img_size=args.img_size, lighting=args.lighting)
    renderer, dino = Renderer(settings), Dino()
    sides: dict[tuple[str, str], tuple[GenerationResult, str]] = {}
    prompt_urls = {}

    async def prepare(name: str, folder: Path, pid: str) -> None:
        module = find_module(folder, pid)
        if module is None:
            sides[name, pid] = GenerationResult(failure_reason="no module"), "missing"
            return
        source = module.read_text(encoding="utf-8")
        key = bundle_key(source, prompts[pid], settings)
        bundle = CACHE / "bundles" / key
        ok = await build_bundle(renderer, dino, source, prompts[pid], bundle)
        views = assets.url(bundle) if ok else None
        sides[name, pid] = GenerationResult(js=str(module), views=views, size=len(source.encode())), key

    for pid, png in prompts.items():
        path = CACHE / "prompts" / f"{hashlib.sha256(png).hexdigest()[:16]}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
        prompt_urls[pid] = assets.url(path)
    logger.info(f"Building view bundles for {len(contenders)} contenders x {len(prompts)} prompts")
    try:
        await asyncio.gather(*(prepare(n, f, pid) for n, f in contenders.items() for pid in prompts))
    finally:
        await renderer.aclose()
    for (name, pid), (gen, _) in sorted(sides.items()):
        if gen.views is None:
            logger.warning(f"{name} / {pid[:12]}: no views ({gen.failure_reason or 'renderer rejected the module'})")

    # Duels.
    names = list(contenders)
    pairs = [(c, args.leader) for c in names if c != args.leader] if args.leader else list(itertools.combinations(names, 2))
    results_path = args.out / "duels.jsonl"
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done[row["key"]] = row
    gate = asyncio.Semaphore(args.parallel)
    sem = asyncio.Semaphore(args.max_vlm_calls)
    rows: list[dict] = []

    async with httpx.AsyncClient(transport=assets.transport()) as http:

        async def duel(pid: str, left: str, right: str) -> None:
            (lgen, lkey), (rgen, rkey) = sides[left, pid], sides[right, pid]
            key = f"{pid}|{left}|{right}|{lkey}|{rkey}|{args.seed}|{model}"
            if key in done:
                rows.append(done[key])
                return
            async with gate:
                winner, detail = await evaluate_duel(vlm, http, sem, prompt_urls[pid], lgen, rgen, args.seed,
                                                     log_id=f"{pid[:12]} / {left} vs {right}")
            row = {"key": key, "prompt": pid, "left": left, "right": right, "winner": winner.value,
                   "decided_by": detail.get("decided_by", detail.get("reason", "")), "detail": detail}
            rows.append(row)
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        await asyncio.gather(*(duel(pid, left, right) for pid in prompts for left, right in pairs))

    summarize(rows, names, prompts, pairs, args, vlm)
    return 0


def summarize(rows: list[dict], names: list[str], prompts: dict, pairs: list, args: argparse.Namespace, vlm) -> None:
    score = {n: Counter() for n in names}
    per_pair = {pair: Counter() for pair in pairs}
    decided = Counter()
    for row in rows:
        left, right, w = row["left"], row["right"], row["winner"]
        decided[row["decided_by"].split(" [")[0]] += 1
        per_pair[left, right][w] += 1
        if w == "left":
            score[left]["win"] += 1
            score[right]["loss"] += 1
        elif w == "right":
            score[right]["win"] += 1
            score[left]["loss"] += 1
        else:
            score[left]["draw"] += 1
            score[right]["draw"] += 1
    lines = [f"# Duels: {len(prompts)} prompts, {len(rows)} duels, judge seed {args.seed}", ""]
    if args.leader:
        lines += [f"Leader: **{args.leader}**, win margin {args.win_margin:.0%}", "",
                  "| challenger | win | draw | loss | margin | result |", "|---|---|---|---|---|---|"]
        for (left, right), c in per_pair.items():
            margin = (c["left"] - c["right"]) / len(prompts)
            verdict = "PASS" if margin >= args.win_margin else "fail"
            lines.append(f"| {left} | {c['left']} | {c['draw']} | {c['right']} | {margin:+.1%} | {verdict} |")
    else:
        lines += ["| contender | win | draw | loss | net |", "|---|---|---|---|---|"]
        for name in sorted(names, key=lambda n: score[n]["win"] - score[n]["loss"], reverse=True):
            s = score[name]
            lines.append(f"| {name} | {s['win']} | {s['draw']} | {s['loss']} | {s['win'] - s['loss']:+d} |")
        lines += ["", "| pair (left vs right) | left wins | draws | right wins |", "|---|---|---|---|"]
        for (left, right), c in per_pair.items():
            lines.append(f"| {left} vs {right} | {c['left']} | {c['draw']} | {c['right']} |")
    lines += ["", "Decided by: " + ", ".join(f"{k} {v}" for k, v in decided.most_common())]
    lines += [f"VLM calls this run: {vlm.calls}, failed: {vlm.errors}"]
    if vlm.errors:
        lines += ["", f"WARNING: {vlm.errors} VLM calls failed; the judge scores those as neutral 5-5. Check judge.log."]
    text = "\n".join(lines)
    (args.out / "summary.md").write_text(text + "\n", encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps({"score": score, "rows": len(rows)}, indent=1), encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    sys.exit(asyncio.run(amain()))
