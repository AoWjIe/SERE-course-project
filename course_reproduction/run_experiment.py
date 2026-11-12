#!/usr/bin/env python3
"""Resource-constrained SERE reproduction on the 1,000-example ESC sample.

The script is intentionally dependency-free.  It uses the archived GPT-4 zero-shot
responses released with Gao et al. as E0, and a deterministic, auditable local
reasoner for E1/E2.  The retrieval path mirrors SERE (conceptual-path proxy,
syntactic-tree proxy, causal-pattern filtering, class-balanced top-k), but does
not claim to replace ConceptNet/Neo4j, spaCy, or the paper's GPT-4o-mini API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter
from pathlib import Path

SEED = 42
CAUSAL_CUES = {
    "cause", "caused", "causes", "causing", "because", "result", "resulted",
    "results", "lead", "led", "leads", "trigger", "triggered", "due",
    "therefore", "thus", "after", "following", "consequence", "so",
}
TEMPORAL_CUES = {"after", "before", "following", "later", "subsequently", "when"}
NEGATION = {"no", "not", "never", "without", "denied", "deny", "unlikely"}
COREF = {"it", "this", "that", "which", "who", "they", "he", "she"}
COORD = {"and", "while", "both", "as"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_token(token: str) -> str:
    token = re.sub(r"[^a-z0-9'-]+", "", token.lower())
    for suffix in ("ing", "edly", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def event_text(row, which: int) -> str:
    return " ".join(row["words"][i] for i in row["events"][which])


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def pos_proxy(token: str) -> str:
    t = token.lower()
    if re.fullmatch(r"[.,;:!?—-]", t): return "PUNCT"
    if t in CAUSAL_CUES: return "CAUSE"
    if t in NEGATION: return "NEG"
    if t in COREF: return "PRON"
    if t in COORD: return "CONJ"
    if t.endswith(("ed", "ing", "ize", "ise")): return "VERB"
    if t.endswith(("ly",)): return "ADV"
    if t.endswith(("tion", "ment", "ness", "ity")): return "NOUN"
    if t in {"a", "an", "the"}: return "DET"
    return "WORD"


def pattern(row) -> str:
    toks = [norm_token(x) for x in row["words"]]
    p0 = min(row["events"][0]); p1 = min(row["events"][1])
    lo, hi = sorted((p0, p1))
    between = set(toks[lo:hi + 1])
    whole = set(toks)
    if between & CAUSAL_CUES:
        return "Direct"
    if between & COREF or whole & COREF and any(x in whole for x in CAUSAL_CUES):
        return "Coreference"
    if len(whole & CAUSAL_CUES) >= 2 or ({"after", "following", "result"} & whole):
        return "Chain"
    if between & COORD and whole & CAUSAL_CUES:
        return "Fork"
    if between & COORD:
        return "Collider"
    return "No"


def conceptual_path(row):
    e0 = norm_token(event_text(row, 0)); e1 = norm_token(event_text(row, 1))
    toks = [norm_token(x) for x in row["words"]]
    p0 = min(row["events"][0]); p1 = min(row["events"][1])
    lo, hi = sorted((p0, p1))
    bridge = []
    for t in toks[lo:hi + 1]:
        if t in CAUSAL_CUES: bridge.append("CAUSE")
        elif t in TEMPORAL_CUES: bridge.append("TIME")
        elif t in NEGATION: bridge.append("NEG")
        elif t in COREF: bridge.append("COREF")
        elif t in COORD: bridge.append("COORD")
    if not bridge: bridge = ["RELATED"]
    direction = "FORWARD" if p0 <= p1 else "REVERSE"
    return [e0[:5], *bridge[:3], direction, e1[:5]]


def syntax_tree_proxy(row):
    p0 = min(row["events"][0]); p1 = min(row["events"][1])
    lo, hi = sorted((p0, p1))
    toks = row["words"]
    seq = ["ROOT", pos_proxy(toks[p0]), "BEFORE" if p0 <= p1 else "AFTER"]
    seq.extend(pos_proxy(t) for t in toks[lo + 1:hi] if pos_proxy(t) != "WORD")
    seq.append(pos_proxy(toks[p1]))
    distance = hi - lo
    seq.append("NEAR" if distance <= 3 else "MID" if distance <= 10 else "FAR")
    return seq[:14]


def conceptual_similarity(a, b):
    pa, pb = a["concept_path"], b["concept_path"]
    return 1.0 - edit_distance(pa, pb) / max(len(pa), len(pb), 1)


def syntax_similarity(a, b):
    return math.exp(-0.05 * edit_distance(a["syntax_proxy"], b["syntax_proxy"]))


def enrich(row):
    row = dict(row)
    row["text"] = " ".join(row["words"])
    row["event_x"] = event_text(row, 0)
    row["event_y"] = event_text(row, 1)
    row["pattern"] = pattern(row)
    row["concept_path"] = conceptual_path(row)
    row["syntax_proxy"] = syntax_tree_proxy(row)
    return row


def split_by_document(rows):
    docs = sorted({r["doc_name"] for r in rows})
    rng = random.Random(SEED); rng.shuffle(docs)
    n = len(docs); train_docs = set(docs[: round(.6*n)])
    dev_docs = set(docs[round(.6*n): round(.8*n)])
    parts = {"train": [], "dev": [], "test": []}
    for r in rows:
        key = "train" if r["doc_name"] in train_docs else "dev" if r["doc_name"] in dev_docs else "test"
        parts[key].append(r)
    return parts


def retrieve(query, corpus, k):
    scored = []
    for c in corpus:
        sp = conceptual_similarity(query, c)
        ss = syntax_similarity(query, c)
        score = .5 * sp + .5 * ss
        scored.append((c["pattern"] == query["pattern"], score, sp, ss, c))
    selected = []
    for label in (1, 0):
        need = k // 2 if label == 1 else k - k // 2
        pool = [x for x in scored if x[4]["bi_causal_label"] == label and x[0]]
        if len(pool) < need:
            used = {x[4]["item_id"] for x in pool}
            pool.extend(x for x in scored if x[4]["bi_causal_label"] == label and x[4]["item_id"] not in used)
        selected.extend(sorted(pool, key=lambda x: x[1], reverse=True)[:need])
    return sorted(selected, key=lambda x: x[1], reverse=True)


def retrieval_margin(retrieved):
    pos = [x[1] for x in retrieved if x[4]["bi_causal_label"] == 1]
    neg = [x[1] for x in retrieved if x[4]["bi_causal_label"] == 0]
    return (statistics.mean(pos) if pos else 0.0) - (statistics.mean(neg) if neg else 0.0)


def cue_score(row):
    toks = [norm_token(x) for x in row["words"]]
    p0, p1 = min(row["events"][0]), min(row["events"][1]); lo, hi = sorted((p0, p1))
    near = set(toks[max(0,lo-3):min(len(toks),hi+4)])
    return 1.0*(bool(near & CAUSAL_CUES)) - .7*(bool(near & NEGATION)) - .15*(hi-lo > 14)


def lexical_features(row):
    toks={norm_token(x) for x in row["words"] if len(norm_token(x))>2}
    feats={"w="+x for x in toks}
    feats.add("pattern="+row["pattern"])
    feats.add("x="+norm_token(row["event_x"]))
    feats.add("y="+norm_token(row["event_y"]))
    return feats


def build_nb(corpus):
    class_docs=Counter(r["bi_causal_label"] for r in corpus)
    counts={0:Counter(),1:Counter()}; vocab=set()
    for r in corpus:
        feats=lexical_features(r); vocab.update(feats)
        for f in feats: counts[r["bi_causal_label"]][f]+=1
    return {"class_docs":class_docs,"counts":counts,"vocab":vocab}


def nb_score(row, model):
    n=sum(model["class_docs"].values()); score=math.log((model["class_docs"][1]+1)/(model["class_docs"][0]+1))
    for f in lexical_features(row):
        p1=(model["counts"][1][f]+1)/(model["class_docs"][1]+2)
        p0=(model["counts"][0][f]+1)/(model["class_docs"][0]+2)
        score += math.log(p1/p0)
    return max(-2.5,min(2.5,score/8.0))


def score_row(row, retrieved, base_pred, alpha, beta, gamma, nb_model):
    base_logit = 0.55 if base_pred else -1.25
    return base_logit + alpha * retrieval_margin(retrieved) + beta * cue_score(row) + gamma*nb_score(row,nb_model)


def metrics(rows, pred_key):
    tp = sum(r["bi_causal_label"] == 1 and r[pred_key] == 1 for r in rows)
    fp = sum(r["bi_causal_label"] == 0 and r[pred_key] == 1 for r in rows)
    fn = sum(r["bi_causal_label"] == 1 and r[pred_key] == 0 for r in rows)
    tn = len(rows)-tp-fp-fn
    p = tp/(tp+fp) if tp+fp else 0; rc=tp/(tp+fn) if tp+fn else 0
    f1=2*p*rc/(p+rc) if p+rc else 0
    return {"n":len(rows),"positive":sum(r["bi_causal_label"] for r in rows),"tp":tp,"fp":fp,"fn":fn,"tn":tn,"precision":p,"recall":rc,"f1":f1}


def build_prompt(row, retrieved):
    examples=[]
    for rank, (_,score,sp,ss,e) in enumerate(retrieved,1):
        examples.append(f"Example {rank}: Text: {e['text']}\nEvent X: {e['event_x']}\nEvent Y: {e['event_y']}\nAnswer: {'Yes' if e['bi_causal_label'] else 'No'}\n[pattern={e['pattern']}; joint={score:.3f}; concept={sp:.3f}; syntax={ss:.3f}]")
    return "Given a text and two events, decide whether a causal relation exists in either direction. Refer to the structurally retrieved examples. Return JSON only.\n\n"+"\n\n".join(examples)+f"\n\nQuery: {row['text']}\nEvent X: {row['event_x']}\nEvent Y: {row['event_y']}\nAnswer format: {{\"Answer\": \"Yes or No\"}}"


def run_variant(rows, corpus, base, k, alpha, beta, gamma, threshold, method, nb_model):
    out=[]; start=time.perf_counter()
    for row in rows:
        retrieved=retrieve(row,corpus,k)
        base_pred=base[row["item_id"]]
        score=score_row(row,retrieved,base_pred,alpha,beta,gamma,nb_model)
        pred=int(score>=threshold)
        prompt=build_prompt(row,retrieved)
        trace=[]
        for rank,(matched,s,sp,ss,e) in enumerate(retrieved,1):
            trace.append({"rank":rank,"item_id":e["item_id"],"label":e["bi_causal_label"],"pattern":e["pattern"],"pattern_match":matched,"joint_score":round(s,6),"concept_score":round(sp,6),"syntax_score":round(ss,6)})
        out.append({**row,"method":method,"k":k,"base_prediction":base_pred,"retrieval_margin":round(retrieval_margin(retrieved),6),"cue_score":cue_score(row),"lexical_prior":round(nb_score(row,nb_model),6),"decision_score":round(score,6),"threshold":threshold,"prediction":pred,"prompt":prompt,"response":json.dumps({"Answer":"Yes" if pred else "No"}),"retrieved":trace})
    return out,time.perf_counter()-start


def tune(dev, corpus, base, nb_model):
    cache={k:{r["item_id"]:retrieve(r,corpus,k) for r in dev} for k in (2,4)}
    best=None
    for alpha in (0,2,4,6,8):
        for beta in (0,.5,1.0):
          for gamma in (0,.5,1,1.5,2):
            scores=[score_row(r,cache[2][r["item_id"]],base[r["item_id"]],alpha,beta,gamma,nb_model) for r in dev]
            for threshold in [x/20 for x in range(-20,41)]:
                trial=[{**r,"pred":int(s>=threshold)} for r,s in zip(dev,scores)]
                m=metrics(trial,"pred")
                candidate=(m["f1"],m["precision"],-abs(m["recall"]-m["precision"]),alpha,beta,gamma,threshold)
                if best is None or candidate>best: best=candidate
    return {"alpha":best[3],"beta":best[4],"gamma":best[5],"threshold":best[6],"dev_f1":best[0],"dev_precision":best[1]}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); args=ap.parse_args()
    root=args.root.resolve(); outdir=root/"results"; (outdir/"traces").mkdir(parents=True,exist_ok=True)
    data_path=root/"data/ESC_ECI.sample1000.jsonl"
    base_path=root/"data/ESC_GPT4_zero_shot_response.json"
    rows=[enrich(r) for r in read_jsonl(data_path)]
    base_rows=read_jsonl(base_path); base={r["item_id"]:int(r["gpt_pred_bi_causal_label"]) for r in base_rows}
    assert len(rows)==1000 and all(r["item_id"] in base for r in rows)
    parts=split_by_document(rows)
    docs={k:{r["doc_name"] for r in v} for k,v in parts.items()}
    assert not(docs["train"]&docs["dev"] or docs["train"]&docs["test"] or docs["dev"]&docs["test"])
    nb_model=build_nb(parts["train"])
    params=tune(parts["dev"],parts["train"],base,nb_model)
    e0=[]
    for r in parts["test"]:
        pred=base[r["item_id"]]
        e0.append({**r,"method":"E0 GPT-4 archived zero-shot","prediction":pred,"prompt":"See Gao et al. archived zero-shot ECI release; no few-shot examples.","response":json.dumps({"Answer":"Yes" if pred else "No"})})
    e1,t1=run_variant(parts["test"],parts["train"],base,2,params["alpha"],params["beta"],params["gamma"],params["threshold"],"E1 SERE-local k=2",nb_model)
    e2,t2=run_variant(parts["test"],parts["train"],base,4,params["alpha"],params["beta"],params["gamma"],params["threshold"],"E2 SERE-local k=4",nb_model)
    for name,data in (("E0",e0),("E1",e1),("E2",e2)):
        write_jsonl(outdir/"traces"/f"{name}_predictions.jsonl",data)
    summary=[]
    for name,data,secs in (("E0",e0,0.0),("E1",e1,t1),("E2",e2,t2)):
        m=metrics(data,"prediction"); m.update({"experiment":name,"runtime_seconds":round(secs,3),"avg_ms_per_sample":round(1000*secs/max(len(data),1),3)})
        summary.append(m)
    with (outdir/"comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["experiment","n","positive","tp","fp","fn","tn","precision","recall","f1","runtime_seconds","avg_ms_per_sample"]); w.writeheader(); w.writerows(summary)
    manifest={"seed":SEED,"data_sha256":hashlib.sha256(data_path.read_bytes()).hexdigest(),"source_rows":len(rows),"document_counts":{k:len(v) for k,v in docs.items()},"sample_counts":{k:len(v) for k,v in parts.items()},"positive_counts":{k:sum(r['bi_causal_label'] for r in v) for k,v in parts.items()},"tuned_on_dev":params,"paper_defaults":{"w_concept":.5,"w_syntax":.5,"e1_k":2,"temperature":0},"local_reasoner":"archived GPT-4 decision + retrieval margin + causal cue + train-only Bernoulli lexical prior","local_change_for_e2":"k only: 2 -> 4","implementation_boundary":"Archived GPT-4 outputs are used only as E0/base signal. Retrieval uses deterministic local proxies; no new external LLM call, Neo4j, ConceptNet, or spaCy parser."}
    (outdir/"run_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    errors=[r for r in e1 if r["prediction"]!=r["bi_causal_label"]]
    correct=[r for r in e1 if r["prediction"]==r["bi_causal_label"]]
    write_jsonl(outdir/"error_cases.jsonl",errors[:20]); write_jsonl(outdir/"true_cases.jsonl",correct[:10])
    print(json.dumps({"manifest":manifest,"results":summary},ensure_ascii=False,indent=2))


if __name__=="__main__": main()
