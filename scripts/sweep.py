"""Sweep the depth penalty to draw the Pareto curve, one run per GPU.

    python scripts/sweep.py --lambdas 0.01 0.03 0.05 0.1 0.2 \
        --init-from runs/s2_routers/best.pt --out-dir runs/sweep \
        --max-steps 3000 -- --compile

Everything after a bare `--` is forwarded verbatim to every `agpt.train` process, so
this wrapper never has to grow a copy of the trainer's flags.

`--lambda-depth` is the only knob that trades perplexity for depth, so this sweep *is*
figure 3. One point is not a Pareto curve, and a single (ppl, FLOPs) pair cannot be
compared against a dense baseline in any way that settles anything: the interesting
question is the shape of the trade, not one sample from it.

Why independent processes and not DDP
-------------------------------------
`agpt/train.py` is single-GPU by design: it takes `cuda:0` and never initialises a
process group. Pinning `CUDA_VISIBLE_DEVICES` per child makes each process see exactly
one card as `cuda:0`, so N runs proceed side by side with no change to the trainer.

That is also the better trade here, not merely the cheaper one. DDP makes ONE run
finish twice as fast; two processes finish TWO runs in the same wall clock. A sweep is
a set of runs that are only meaningful *as a set* -- a lambda that finishes early is
worth nothing until the rest of the curve exists -- so throughput is the axis that
matters, not latency.

What this does NOT do
---------------------
Nothing here makes the points comparable on its own. Comparability comes from the token
budget: `--total-batch-tokens x --max-steps` and `--seed` must match across points.
Every run writes them to `<run>/config.json`, so check there rather than trusting the
command you think you ran.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import subprocess
import threading
import time

import torch


def stream(proc, name, log_path):
    """Tee a child's output to the console (prefixed) and to its own log file."""
    with open(log_path, "w", encoding="utf-8") as log:
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(f"[{name}] {line}", end="", flush=True)


def launch(lam, gpu, args, passthrough):
    run_name = f"lam{lam:g}".replace(".", "p")
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cmd = [sys.executable, "-u", "-m", "agpt.train",
           "--stage", args.stage, "--variant", "adaptive",
           "--lambda-depth", str(lam),
           "--out-dir", args.out_dir, "--run-name", run_name,
           "--data-dir", args.data_dir, "--max-steps", str(args.max_steps),
           "--batch-size", str(args.batch_size), "--seed", str(args.seed)]
    if args.init_from:
        cmd += ["--init-from", args.init_from]
    cmd += passthrough

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    if args.dry_run:
        print(f"GPU {gpu}: {' '.join(cmd)}")
        return None, run_name, run_dir

    # Merged stderr and unbuffered output, so the interleaved log stays in order.
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    t = threading.Thread(target=stream,
                         args=(proc, run_name, os.path.join(run_dir, "train.log")),
                         daemon=True)
    t.start()
    return (proc, t), run_name, run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.01, 0.03, 0.05, 0.1, 0.2])
    ap.add_argument("--stage", default="joint", choices=["routers", "joint"])
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--data-dir", default="data/wikitext103")
    ap.add_argument("--out-dir", default="runs/sweep")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gpus", type=int, default=None,
                    help="how many to use; defaults to every visible CUDA device")
    ap.add_argument("--dry-run", action="store_true")
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    n_gpu = args.gpus or max(torch.cuda.device_count(), 1)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"{len(args.lambdas)} runs over {n_gpu} GPU(s): {args.lambdas}\n")

    # Round-robin onto the cards, waiting whenever every card is busy. A simple slot
    # pool rather than a queue library: the runs are long and few, so the scheduling
    # overhead is irrelevant and being able to read this matters more.
    pending = list(args.lambdas)
    running = {}                       # gpu -> (handle, run_name)
    done = []
    while pending or running:
        while pending and len(running) < n_gpu:
            gpu = next(g for g in range(n_gpu) if g not in running)
            handle, name, run_dir = launch(pending.pop(0), gpu, args, passthrough)
            if handle is None:
                done.append((name, run_dir, 0))
                continue
            running[gpu] = (handle, name, run_dir)
        if not running:
            break
        time.sleep(2)
        for gpu, ((proc, thread), name, run_dir) in list(running.items()):
            if proc.poll() is not None:
                thread.join(timeout=5)
                print(f"[{name}] exited with {proc.returncode}")
                done.append((name, run_dir, proc.returncode))
                del running[gpu]

    manifest = os.path.join(args.out_dir, "sweep.json")
    with open(manifest, "w") as f:
        json.dump({"lambdas": args.lambdas, "args": vars(args),
                   "runs": [{"name": n, "dir": d, "returncode": rc}
                            for n, d, rc in done]}, f, indent=2)

    failed = [n for n, _, rc in done if rc]
    print(f"\n{len(done) - len(failed)}/{len(done)} runs finished -> {manifest}")
    if failed:
        print(f"FAILED: {failed}  (see <run>/train.log)")
    else:
        print("\nNext:\n"
              "  for each run:  python scripts/evaluate.py --ckpt <run>/best.pt\n"
              "  then collect the (ppl, flops_frac_layers) pairs into a JSON list and\n"
              "  pass it to scripts/figures.py --sweep to draw the Pareto curve.")


if __name__ == "__main__":
    main()
