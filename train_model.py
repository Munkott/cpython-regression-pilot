"""Fit and evaluate a compact full-vocabulary neural language model from scratch."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from zipfile import ZipFile

import run

PLAN_PATH = Path(__file__).with_name("training_plan.json")
PLAN = json.loads(PLAN_PATH.read_bytes())
CTX, EMB, HIDDEN = (PLAN[key] for key in ("context", "embedding", "hidden"))
VOCAB, STRIDE = PLAN["vocab"], PLAN["group_stride"]
GROUPS = (VOCAB + STRIDE - 1) // STRIDE
FINAL_GROUP_WIDTH = VOCAB - (GROUPS - 1) * STRIDE
FLOAT = "float64"
SHAPES = {
    "group_embedding": (GROUPS, EMB),
    "member_embedding": (STRIDE, EMB),
    "context_weights": (CTX * EMB, HIDDEN),
    "context_bias": (HIDDEN,),
    "group_weights": (HIDDEN, GROUPS),
    "group_bias": (GROUPS,),
    "member_weights": (HIDDEN, STRIDE),
    "member_bias": (STRIDE,),
    "conditional_bias": (GROUPS, STRIDE),
}


def clock():
    return time.monotonic(), run.cpu()


def elapsed(start):
    return {"wall_s": time.monotonic() - start[0], "cpu_s": run.cpu() - start[1]}


def load_numpy(wheel, site):
    global np
    if run.digest(wheel) != PLAN["numpy"]["sha256"]:
        raise ValueError("unexpected NumPy wheel")
    if sys.version_info[:2] != (3, 12):
        raise ValueError("the pinned wheel requires CPython 3.12")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_CORETYPE"] = "NEHALEM"
    site.mkdir()
    with ZipFile(wheel) as archive:
        archive.extractall(site)
    sys.path.insert(0, str(site.resolve()))
    import numpy as np
    if np.__version__ != PLAN["numpy"]["version"]:
        raise ValueError("unexpected NumPy version")


def initialize(seed):
    rng = np.random.default_rng(seed)
    scale = {"group_embedding": 0.10, "member_embedding": 0.10,
             "context_weights": 1 / math.sqrt(CTX * EMB),
             "group_weights": 1 / math.sqrt(HIDDEN), "member_weights": 1 / math.sqrt(HIDDEN)}
    return {name: rng.normal(0, scale[name], size=shape) if name in scale
            else np.zeros(shape, dtype=FLOAT) for name, shape in SHAPES.items()}


def distribution(logits, target):
    adjusted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(adjusted)
    norm = exp.sum(axis=1, keepdims=True)
    summed_nll = (np.log(norm[:, 0]) - adjusted[np.arange(len(target)), target]).sum()
    return float(summed_nll), exp / norm


def forward(x, y, params, gradient=False):
    x_group, x_member = divmod(x, STRIDE)
    y_group, y_member = divmod(y, STRIDE)
    state = params["group_embedding"][x_group] + params["member_embedding"][x_member]
    flat = state.reshape(len(x), CTX * EMB)
    hidden = np.tanh(flat @ params["context_weights"] + params["context_bias"])
    group_scores = hidden @ params["group_weights"] + params["group_bias"]
    member_scores = hidden @ params["member_weights"] + params["member_bias"] + params["conditional_bias"][y_group]
    member_scores[y_group == GROUPS - 1, FINAL_GROUP_WIDTH:] = -np.inf
    loss_group, d_group = distribution(group_scores, y_group)
    loss_member, d_member = distribution(member_scores, y_member)
    nll = (loss_group + loss_member) / len(y)
    if not gradient:
        return nll
    idx = np.arange(len(y))
    d_group[idx, y_group] -= 1
    d_member[idx, y_member] -= 1
    d_group /= len(y)
    d_member /= len(y)
    hidden_grad = d_group @ params["group_weights"].T + d_member @ params["member_weights"].T
    activation_grad = hidden_grad * (1 - hidden**2)
    embed_grad = (activation_grad @ params["context_weights"].T).reshape(state.shape)
    grad = {
        "group_weights": hidden.T @ d_group, "group_bias": d_group.sum(axis=0),
        "member_weights": hidden.T @ d_member, "member_bias": d_member.sum(axis=0),
        "context_weights": flat.T @ activation_grad, "context_bias": activation_grad.sum(axis=0),
        "group_embedding": np.zeros_like(params["group_embedding"]),
        "member_embedding": np.zeros_like(params["member_embedding"]),
        "conditional_bias": np.zeros_like(params["conditional_bias"]),
    }
    np.add.at(grad["group_embedding"], x_group, embed_grad)
    np.add.at(grad["member_embedding"], x_member, embed_grad)
    np.add.at(grad["conditional_bias"], y_group, d_member)
    return nll, grad


def check_gradient():
    params = initialize(19)
    observed = np.arange(4 * CTX, dtype=np.int64).reshape(4, CTX)
    targets = np.array([0, 541, 1992, VOCAB - 1], dtype=np.int64)
    _, actual = forward(observed, targets, params, gradient=True)
    entries = {
        "group_embedding": [(0, 1), (2, 0)], "member_embedding": [(7, 0), (4, 2)],
        "context_weights": [(0, 0), (11, 6)], "context_bias": [(5,)],
        "group_weights": [(4, 1), (7, 2)], "group_bias": [(0,), (196,)],
        "member_weights": [(3, 1)], "member_bias": [(29,), (80,)],
        "conditional_bias": [(0, 0), (2, 29), (196, 80), (196, 81)],
    }
    epsilon = 1e-5
    errors = []
    for name, points in entries.items():
        for point in points:
            base = params[name][point]
            params[name][point] = base + epsilon
            above = forward(observed, targets, params)
            params[name][point] = base - epsilon
            below = forward(observed, targets, params)
            params[name][point] = base
            errors.append(abs((above - below) / (2 * epsilon) - actual[name][point]))
    if max(errors) > 3e-8:
        raise ValueError("finite differences failed")
    return {"coordinates":len(errors), "epsilon":epsilon, "max_abs_error":max(errors)}


class Corpus:
    def __init__(self, dataset, split, manifest):
        facts = PLAN["corpus"]["splits"][split]
        path = dataset / facts["file"]
        if run.digest(path) != facts["sha256"] or path.stat().st_size != facts["bytes"]:
            raise ValueError("dataset split hash or length differs")
        docs = [doc for doc in manifest["documents"] if doc["split"] == split]
        offset = 0
        for doc in docs:
            if doc["offset_tokens"] != offset or doc["tokens"] <= CTX:
                raise ValueError("invalid document window")
            offset += doc["tokens"]
        if offset != facts["tokens"]:
            raise ValueError("document inventory does not cover the split")
        self.tokens = np.memmap(path, dtype="<u2", mode="r", shape=(offset,))
        self.starts = np.array([doc["offset_tokens"] for doc in docs], dtype=np.int64)
        self.ends = np.cumsum([doc["tokens"] - CTX for doc in docs], dtype=np.int64)
        self.previous = np.concatenate(([0], self.ends[:-1]))

    def sample(self, seed, count):
        draw = np.random.default_rng(seed).integers(0, self.ends[-1], size=count, dtype=np.int64)
        document = np.searchsorted(self.ends, draw, side="right")
        return self.starts[document] + CTX + draw - self.previous[document]

    def slices(self, positions):
        context = self.tokens[positions[:, None] - np.arange(CTX, 0, -1)].astype(np.int64)
        target = self.tokens[positions].astype(np.int64)
        if context.max() >= VOCAB or target.max() >= VOCAB:
            raise ValueError("unexpected token ID")
        return context, target


def load_corpus(dataset, split):
    path = dataset / "manifest.json"
    if run.digest(path) != PLAN["corpus"]["manifest_sha256"]:
        raise ValueError("dataset manifest differs from the audited input")
    manifest = json.loads(path.read_bytes())
    return Corpus(dataset, split, manifest)


def array_digest(values):
    return hashlib.sha256(values.astype("<i8", copy=False).tobytes()).hexdigest()


def evaluate(params, x, y):
    batch = PLAN["batch"]
    total = 0.0
    for start in range(0, len(y), batch):
        targets = y[start:start + batch]
        total += forward(x[start:start + batch], targets, params) * len(targets)
    return total / len(y)


def heldout(dataset):
    validation = load_corpus(dataset, "validation")
    positions = validation.sample(PLAN["seed_eval"], PLAN["eval_samples"])
    x, y = validation.slices(positions)
    return x, y, positions


def choose_parameters(checkpoint):
    with np.load(checkpoint, allow_pickle=False) as saved:
        if set(saved.files) != set(SHAPES) or len(saved.files) != len(SHAPES):
            raise ValueError("wrong checkpoint inventory")
        params = {name: saved[name] for name in SHAPES}
    for name, values in params.items():
        if values.shape != SHAPES[name] or values.dtype != np.dtype(FLOAT) or not np.isfinite(values).all():
            raise ValueError("invalid checkpoint array")
    return params


def fit(dataset, output, *, exclude_documents=()):
    started = clock()
    train = load_corpus(dataset, "train")
    if exclude_documents:
        manifest = json.loads((dataset / "manifest.json").read_bytes())
        docs = [row for row in manifest["documents"] if row["split"] == "train"]
        keep = np.array([row["document"] not in exclude_documents for row in docs])
        lengths = (train.ends - train.previous)[keep]
        train.starts = train.starts[keep]
        train.ends = np.cumsum(lengths)
        train.previous = np.concatenate(([0],train.ends[:-1]))
    x_eval, y_eval, validation_positions = heldout(dataset)
    gradient_check = check_gradient()
    train_positions = train.sample(PLAN["seed_train"], PLAN["steps"] * PLAN["batch"])
    params = initialize(PLAN["seed_initial"])
    initial_nll = evaluate(params,x_eval,y_eval)
    moments = {key:np.zeros_like(values) for key,values in params.items()}
    scales = {key:np.zeros_like(values) for key,values in params.items()}
    b1, b2 = PLAN["adam_beta1"], PLAN["adam_beta2"]
    lr, epsilon = PLAN["learning_rate"], PLAN["adam_epsilon"]
    trained = []
    training = clock()
    for step in range(1, PLAN["steps"] + 1):
        limit = step * PLAN["batch"]
        x, y = train.slices(train_positions[limit - PLAN["batch"]:limit])
        nll, gradients = forward(x,y,params,gradient=True)
        for key, grad in gradients.items():
            first, second = moments[key], scales[key]
            first *= b1
            first += (1 - b1) * grad
            second *= b2
            second += (1 - b2) * grad**2
            params[key] -= lr * (first / (1 - b1**step)) / (np.sqrt(second / (1 - b2**step)) + epsilon)
        if step == 1 or step % 128 == 0:
            trained.append({"step":step, "train_nll":nll})
            print(f"step {step} train_nll={nll:.5f}",flush=True)
    fit_time = elapsed(training)
    final_nll = evaluate(params,x_eval,y_eval)
    perm = np.random.default_rng(PLAN["seed_shuffle"]).permutation(len(y_eval))
    shuffled_nll = evaluate(params, x_eval[perm], y_eval)
    counts = np.bincount(train.tokens[train_positions], minlength=VOCAB) + 1
    unigram_nll = float(-np.log(counts[y_eval] / counts.sum()).mean())
    if not math.isfinite(final_nll):
        raise ValueError("nonfinite validation loss")
    checkpoint = output / "checkpoint.npz"
    with checkpoint.open("xb") as stream:
        np.savez(stream, **params)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        "plan_sha256":run.digest(PLAN_PATH),
        "dataset_sha256":PLAN["corpus"]["manifest_sha256"],
        "checkpoint_sha256":run.digest(checkpoint),"checkpoint_bytes":checkpoint.stat().st_size,
        "parameters":sum(values.size for values in params.values()),
        "steps":PLAN["steps"], "train_targets":len(train_positions),
        "train_unique_positions":int(np.unique(train_positions).size),
        "validation_targets":len(y_eval),"validation_unique_positions":int(np.unique(validation_positions).size),
        "train_positions_sha256":array_digest(train_positions),
        "validation_positions_sha256":array_digest(validation_positions),
        "initial_nll":initial_nll,"final_nll":final_nll,"shuffled_context_nll":shuffled_nll,
        "unigram_nll":unigram_nll,"gradient_check":gradient_check,
        "trace":trained,"optimizer":fit_time,"evaluation_and_fit":elapsed(started),
    }


def score(dataset, checkpoint):
    params = choose_parameters(checkpoint)
    x,y,positions = heldout(dataset)
    return {"validation_targets":len(y),"validation_positions_sha256":array_digest(positions),
            "validation_nll":evaluate(params,x,y),"checkpoint_sha256":run.digest(checkpoint)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fit","score","compare"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    started = clock()
    args.out.mkdir()
    load_numpy(args.wheel,args.out/"site")
    if args.mode == "fit":
        results = fit(args.dataset,args.out)
    else:
        if args.checkpoint is None:
            parser.error("checkpoint required")
        results = score(args.dataset,args.checkpoint)
        if args.mode == "compare":
            if args.reference is None:
                parser.error("reference checkpoint required")
            current, reference = map(choose_parameters, (args.checkpoint,args.reference))
            results["max_weight_abs_difference"] = max(float(np.max(np.abs(current[key]-reference[key])))
                                                       for key in SHAPES)
    results["total"] = elapsed(started)
    run.put_json(args.out/"report.json",results)


if __name__ == "__main__":
    main()
