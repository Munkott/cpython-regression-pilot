"""Extract fields and verify a bounded invoice/PO sample without a paid API."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date
from decimal import Decimal
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import resource
import subprocess
from tempfile import TemporaryDirectory
import time

from fixtures import LABELS

DATA_FIELDS = ("invoice","po","date","net","tax","total")
ALIASES = {alias.casefold():field for style in LABELS for alias,field in zip(style,DATA_FIELDS)}
PATTERN = re.compile(r"^("+"|".join(re.escape(label) for label in ALIASES)+r")\s+(.+?)\s*$", re.I)
PRICE = re.compile(r"USD\s+(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)\.[0-9]{2}\Z")
ROOT = Path(__file__).resolve().parent

def clock():
    return time.monotonic(),sum(f.ru_utime+f.ru_stime for f in (resource.getrusage(resource.RUSAGE_SELF),
                                                               resource.getrusage(resource.RUSAGE_CHILDREN)))

def elapsed(start):
    wall,cpu=clock()
    return {"wall_s":wall-start[0],"cpu_s":cpu-start[1]}

def canonical(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()

def convert(field, value):
    if field=="date":
        return date.fromisoformat(value).isoformat()
    if field in ("invoice","po"):
        prefix="INV" if field=="invoice" else "PO"
        if re.fullmatch(prefix+r"-\d{2}-\d{6}",value):
            return value
        raise ValueError(value)
    if PRICE.fullmatch(value):
        return str(Decimal(value.removeprefix("USD ").replace(",","")))
    raise ValueError(value)

def parse(lines):
    fields = {}
    confidence = {}
    problems = []
    for line,score in lines:
        match=PATTERN.match(line.strip())
        if not match:
            continue
        key=ALIASES[match[1].casefold()]
        try:
            val=convert(key,match[2])
        except ValueError:
            problems.append("format:"+key)
            continue
        if key in fields and fields[key]!=val:
            problems.append("ambiguous:"+key)
            continue
        fields[key]=val
        confidence[key]=score
    for key in DATA_FIELDS:
        if key not in fields:
            problems.append("missing:"+key)
        elif confidence[key]<60:
            problems.append("low_confidence:"+key)
    return fields,problems

def ocr(source,lang="eng",psm="6", merge_rows=False):
    prefix = ROOT/"tess"
    env = os.environ | {
        "LD_LIBRARY_PATH":str(prefix/"usr/lib/x86_64-linux-gnu"),
        "TESSDATA_PREFIX":str(prefix/"usr/share/tesseract-ocr/5/tessdata"),
        "OMP_THREAD_LIMIT":"1",
    }
    args=[str(prefix/"usr/bin/tesseract"),str(source),"stdout","-l",lang,"--psm",psm,"tsv"]
    data=subprocess.check_output(args,env=env,timeout=20,stderr=subprocess.DEVNULL).decode("utf-8")
    groups={}
    words=[]
    for row in csv.DictReader(data.splitlines(),delimiter="\t"):
        if row["level"]=="5" and row["text"]:
            key=tuple(row[k] for k in ("page_num","block_num","par_num","line_num"))
            token = (int(row["left"]),int(row["top"]),int(row["height"]),row["text"],float(row["conf"]))
            groups.setdefault(key,[]).append(token)
            words.append(token)
    if merge_rows:
        lines=[]
        for word in sorted(words,key=lambda t:t[1]+t[2]/2):
            center=word[1]+word[2]/2
            if lines and abs(lines[-1][0]-center)<=max(5,word[2]*.48):
                lines[-1][1].append(word)
            else:
                lines.append((center,[word]))
        groups = {i:tokens for i,(_,tokens) in enumerate(lines)}
    output=[]
    for tokens in groups.values():
        ordered=sorted(tokens)
        line=" ".join(word[3] for word in ordered)
        match=PATTERN.match(line)
        if match:
            label_words=len(match[1].split())
            value_tokens=ordered[label_words:]
            confidence=min(token[4] for token in value_tokens)
        else:
            confidence=min(token[4] for token in ordered)
        output.append((line,confidence))
    return output


def scan_fields(source):
    first, issues = parse(ocr(source))
    if not issues:
        return first,issues
    second, issues2 = parse(ocr(source,psm="11",merge_rows=True))
    if not issues2 and all(first.get(key,value)==value for key,value in second.items()):
        return second,[]
    third, issues3 = parse(ocr(source,psm="4"))
    results={}
    flags=[]
    for key in DATA_FIELDS:
        candidates=Counter(result[key] for result in (first,second,third) if key in result)
        if not candidates:
            flags.append("missing:"+key)
            continue
        value,count=candidates.most_common(1)[0]
        results[key]=value
        certain=any(result.get(key)==value and not any(issue.endswith(":"+key) for issue in problems)
                    for result,problems in ((first,issues),(second,issues2),(third,issues3)))
        if count<2 or not certain or len(candidates)>1:
            flags.append("needs_confirmation:"+key)
    return results,flags


def extract(root,row,force_ocr=False):
    start=time.monotonic()
    source=root/row["file"]
    assert sha256(source.read_bytes()).hexdigest()==row["sha256"]
    if source.suffix==".png":
        fields,problems=scan_fields(source)
        route="ocr"
    else:
        if not force_ocr:
            output=subprocess.check_output(["pdftotext","-layout",str(source),"-"],timeout=10).decode()
            lines=[(line,100) for line in output.splitlines()]
        else:
            lines=[]
        if not lines or not any(PATTERN.match(line.strip()) for line,_ in lines):
            with TemporaryDirectory(prefix="osi-pdf-render-") as tmp:
                image=Path(tmp)/"scan"
                subprocess.run(["pdftoppm","-r","150","-png","-gray","-singlefile",str(source),str(image)],
                               check=True,timeout=15,stdout=subprocess.DEVNULL)
                fields,problems=scan_fields(image.with_suffix(".png"))
                route="ocr_from_pdf"
        else:
            route="native_pdf"
            fields,problems=parse(lines)
    return {"id":row["id"],"file":row["file"],"source_sha256":row["sha256"],"route":route,"fields":fields,
            "review":problems,"seconds":time.monotonic()-start}

def input_set(root):
    truth=json.loads((root/"truth.json").read_bytes())
    source=[{k:r[k] for k in ("id","file","sha256")} for r in truth]
    orders={r["id"]:r["reference_po"] for r in truth}
    return truth,source,orders

def consolidate(rows,orders):
    invoices=Counter(r["fields"]["invoice"] for r in rows if "invoice" in r["fields"])
    for r in rows:
        f=r["fields"]
        if f.get("po") is not None and f["po"]!=orders[r["id"]]:
            r["review"].append("purchase_order_mismatch")
        if "invoice" in f and invoices[f["invoice"]]>1:
            r["review"].append("duplicate_invoice")
        if all(k in f for k in ("net","tax","total")) and Decimal(f["net"])+Decimal(f["tax"])!=Decimal(f["total"]):
            r["review"].append("amounts_do_not_reconcile")
    return rows

def quality(rows,truth):
    expected={x["id"]:x for x in truth}
    assert {x["id"] for x in rows}==expected.keys()
    clean_wrong = flagged_wrong = 0
    flags = Counter(reason for row in rows for reason in row["review"])
    good_metrics={}
    for mode in ("native","scan"):
        selected=[r for r in rows if expected[r["id"]]["mode"]==mode]
        complete=sum(all(r["fields"].get(k)==expected[r["id"]][k] for k in DATA_FIELDS) for r in selected)
        good_metrics[mode]={"documents":len(selected),"exact_all_fields":complete,
                            "review":sum(bool(r["review"]) for r in selected)}
    issues=0
    for r in rows:
        exp=expected[r["id"]]
        if exp["exception"]:
            issues+=1
            if not r["review"]:
                raise AssertionError(f"Unflagged programmed exception {r['id']}")
        good=all(r["fields"].get(k)==exp[k] for k in DATA_FIELDS)
        if not good:
            if r["review"]:
                flagged_wrong+=1
            else:
                clean_wrong+=1
    semantic=[{k:r[k] for k in ("id","source_sha256","fields","review")} for r in rows]
    return {"documents":len(rows),"issues_programmed_and_flagged":issues,"wrong_sent_to_review":flagged_wrong,
            "wrong_without_review":clean_wrong,"total_review":sum(bool(x["review"]) for x in rows),
            "by_source":good_metrics,"flag_counts":flags,"product_sha256":sha256(canonical(semantic)).hexdigest()}

def scheduled(root,source,workers,force_ocr=False):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda row:extract(root,row,force_ocr),source))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--force-ocr",action="store_true")
    a=p.parse_args()
    truth,source,orders=input_set(a.root)
    start=clock()
    rows=consolidate(scheduled(a.root,source,a.workers,a.force_ocr),orders)
    receipt={"quality":quality(rows,truth),"run_time":elapsed(start),"workers":a.workers,
             "force_ocr":a.force_ocr,"routes":Counter(r["route"] for r in rows)}
    a.out.mkdir(exist_ok=True)
    (a.out/"result.json").write_bytes(canonical(receipt)+b"\n")
    with (a.out/"invoices.csv").open("w",encoding="utf-8",newline="") as dest:
        writer=csv.DictWriter(dest,fieldnames=("id","file","sha256","route",*DATA_FIELDS,"review"))
        writer.writeheader()
        for r in rows:
            writer.writerow({k:r[k] for k in ("id","file","route")}|r["fields"]|
                            {"review":";".join(r["review"]),"sha256":r["source_sha256"]})
    print(json.dumps(receipt,indent=2))

if __name__=="__main__":
    main()
