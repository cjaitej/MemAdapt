"""Train several variants at once, one per GPU.

    python scripts/train_pair.py --variants b6_amt_joint b7_uncoupled \
        --data-dir /kaggle/input/fineweb-edu-docs --out-dir /kaggle/working/runs \
        --max-steps 12000 --batch-size 16 -- --compile

Any argument after a bare `--` is forwarded verbatim to every `amt.train` process, so
this wrapper never has to grow a copy of the trainer's flags.

Why independent processes and not DDP
-------------------------------------
`amt/train.py` is single-GPU by design: it takes `cuda:0` and never initialises a
process group. On Kaggle's `T4 x2` that leaves half the hardware idle. Pinning
CUDA_VISIBLE_DEVICES per child makes each process see exactly one card as `cuda:0`,
so two variants train side by side with no change to the trainer at all.

That is also the better trade for this project, not just the cheaper one. DDP makes
ONE run finish twice as fast; two processes finish TWO runs in the same wall clock.
The experiment matrix (RESEARCH_PLAN.md §7.1) is a set of runs that are only
meaningful *compared against each other*, so a variant finishing early is worth
nothing until its control finishes too -- throughput is the axis that matters, not
latency. DDP would also have to be taught what the memory bank means across ranks:
the bank is indexed per batch element and per document stream, so sharding a batch
across two cards silently changes what each stream can retrieve.

What this does NOT do
---------------------
Nothing here makes two runs comparable on its own. Comparability comes from the
token budget: `--total-batch-tokens x --max-steps` must match across variants, and
`--seed` should too. `--batch-size` is free to differ -- it is a memory/speed knob
that grad accumulation absorbs -- which is what lets a memory variant drop to a
smaller micro-batch without leaving the comparison.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, ".")


def gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:                                     # noqa: BLE001
        return 0


def build(variant, run_name, gpu, args, passthrough):
    # -u matters. A child's stdout is a pipe here, not a terminal, so Python
    # block-buffers it (~8KB) and nothing appears until the buffer fills or the
    # process exits -- which for a training run means hours of apparent silence
    # followed by everything at once. Popen's bufsize=1 does not help: it line-buffers
    # the parent's reading side, not the child's writing side.
    cmd = [sys.executable, "-u", "-m", "amt.train",
           "--variant", variant,
           "--run-name", run_name,
           "--data-dir", args.data_dir,
           "--out-dir", args.out_dir,
           "--batch-size", str(args.batch_size),
           "--total-batch-tokens", str(args.total_batch_tokens),
           "--max-steps", str(args.max_steps),
           "--ckpt-every", str(args.ckpt_every),
           "--seed", str(args.seed)]
    if args.resume:
        cmd += ["--resume", args.resume]
    return cmd + passthrough


def child_env(gpu, run_dir):
    env = os.environ.copy()
    # The child sees one card, numbered 0. train.py asks for "cuda" and gets the one
    # we chose, with no device-selection code in the trainer.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Belt and braces with `-u`: this also reaches anything the child spawns.
    env["PYTHONUNBUFFERED"] = "1"
    # Separate inductor caches. Two processes compiling the same graph at the same
    # moment otherwise race on one cache directory, and the failure mode is a
    # corrupted entry that only shows up as a compile error on a later run.
    env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(run_dir, ".inductor")
    return env


def pump(proc, name, log_path, width, seen):
    """Stream a child's output to the console (prefixed) and to its own log file."""
    with open(log_path, "w", encoding="utf-8", buffering=1) as log:
        for line in proc.stdout:
            log.write(line)
            seen[name] = time.perf_counter()
            print(f"[{name:<{width}}] {line.rstrip()}", flush=True)


def heartbeat(seen, stop, width, every=120):
    """Say a quiet child is still alive.

    `torch.compile` spends minutes producing nothing before the first step prints, so
    silence is normal and indistinguishable from a hang -- and after a buffering bug
    that made silence the default, it should never be ambiguous again.
    """
    while not stop.wait(every):
        now = time.perf_counter()
        for name, last in sorted(seen.items()):
            quiet = now - last
            if quiet >= every:
                print(f"[{name:<{width}}] ... alive, {quiet/60:.0f} min since its last "
                      "line (compiling?)", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", required=True,
                    help="one variant per GPU, e.g. b6_amt_joint b7_uncoupled")
    ap.add_argument("--run-names", nargs="+",
                    help="defaults to <prefix>_<variant>")
    ap.add_argument("--prefix", default="r1")
    ap.add_argument("--gpus", nargs="+", type=int,
                    help="GPU ordinals to use; defaults to 0..n-1")
    ap.add_argument("--data-dir", default="data/fineweb_edu_docs")
    ap.add_argument("--out-dir", default="runs")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--total-batch-tokens", type=int, default=65536,
                    help="the token budget per step; keep identical across variants "
                         "or the comparison is not at matched tokens")
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--ckpt-every", type=int, default=500,
                    help="small enough that a killed session loses minutes, not hours")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--resume", default=None,
                    help="passed to every run; use 'latest' with the same --run-names")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and exit")
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    gpus = args.gpus if args.gpus is not None else list(range(max(gpu_count(), 1)))
    names = args.run_names or [f"{args.prefix}_{v}" for v in args.variants]
    if len(names) != len(args.variants):
        sys.exit("--run-names must have one entry per --variant")
    if len(args.variants) > len(gpus):
        sys.exit(f"{len(args.variants)} variants but only {len(gpus)} GPU(s): "
                 f"{gpus}. Run them in two batches, or pass --gpus explicitly.")

    jobs = []
    for variant, name, gpu in zip(args.variants, names, gpus):
        run_dir = os.path.join(args.out_dir, name)
        jobs.append((variant, name, gpu, run_dir, build(variant, name, gpu, args,
                                                        passthrough)))

    planned = args.max_steps * args.total_batch_tokens
    print(f"{len(jobs)} run(s), {planned/1e6:,.0f}M tokens each "
          f"({args.max_steps:,} steps x {args.total_batch_tokens:,})\n")
    for variant, name, gpu, run_dir, cmd in jobs:
        print(f"  GPU {gpu}  {name:<20} {' '.join(cmd)}")
    print()
    if args.dry_run:
        return 0

    width = max(len(n) for _, n, _, _, _ in jobs)
    procs, threads = [], []
    seen, stop = {}, threading.Event()
    started = time.perf_counter()
    try:
        beat = threading.Thread(target=heartbeat, args=(seen, stop, width), daemon=True)
        beat.start()
        for variant, name, gpu, run_dir, cmd in jobs:
            # Created here, not while planning, so --dry-run touches nothing.
            os.makedirs(run_dir, exist_ok=True)
            log_path = os.path.join(run_dir, "train.log")
            proc = subprocess.Popen(cmd, env=child_env(gpu, run_dir),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            procs.append((name, proc))
            seen[name] = time.perf_counter()
            t = threading.Thread(target=pump, args=(proc, name, log_path, width, seen),
                                 daemon=True)
            t.start()
            threads.append(t)
            print(f"[{name:<{width}}] started on GPU {gpu}, pid {proc.pid}, "
                  f"log {log_path}", flush=True)

        codes = [(name, proc.wait()) for name, proc in procs]
        for t in threads:
            t.join(timeout=10)
    except KeyboardInterrupt:
        print("\ninterrupted -- terminating children", flush=True)
        for _, proc in procs:
            proc.terminate()
        for _, proc in procs:
            proc.wait()
        return 130
    finally:
        stop.set()

    mins = (time.perf_counter() - started) / 60
    print(f"\nfinished in {mins:.1f} min")
    for name, code in codes:
        print(f"  {name:<20} exit {code}" + ("" if code == 0 else "   <-- FAILED"))
    failed = [n for n, c in codes if c != 0]
    if failed:
        print(f"\n{len(failed)} run(s) failed; see <out-dir>/<run>/train.log. A run "
              "that died partway can be continued with --resume latest and the same "
              "--run-names.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
