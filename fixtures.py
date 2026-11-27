"""Create fixed synthetic invoices, including native PDFs, scans and exceptions."""
from decimal import Decimal
from hashlib import sha256
import argparse
import json
from pathlib import Path
import subprocess

LABELS = (
    ("Invoice number","Purchase order","Invoice date","Subtotal","Tax amount","Amount due"),
    ("Document ID","Order reference","Issue date","Net amount","Sales tax","Invoice total"),
    ("Invoice ID","PO number","Date issued","Net total","Tax total","Total payable"),
)


def pdf(lines, style, faded=False):
    def literal(value):
        return "(" + value.replace("\\","\\\\").replace("(","\\(").replace(")","\\)") + ")"
    heading = 13+style
    actions = ["BT /F1 18 Tf 42 752 Td (DEMONSTRATION INVOICE - NO PAYMENT) Tj ET",
               "0.2 G 42 718 m 555 718 l S","BT /F1 10 Tf 52 701 Td (Synthetic testing document) Tj ET"]
    for index,(name,value) in enumerate(lines):
        y = 654-index*42
        x = 52 if style!=1 else 86
        position = 260 if style!=2 else 320
        colour = "0.46 g" if faded and index==5 else "0 g"
        actions.append(f"{colour} BT /F1 {heading} Tf {x} {y} Td {literal(name)} Tj ET "
                       f"BT /F1 {heading} Tf {position} {y} Td {literal(value)} Tj ET")
    actions.extend(["0 g BT /F1 11 Tf 52 330 Td (Test Services Ltd - Training use only) Tj ET",
                    "BT /F1 10 Tf 52 292 Td (This sample has no financial or contractual effect.) Tj ET"])
    stream = ("\n".join(actions)+"\n").encode("ascii")
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
               b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
               b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
               b"<< /Length "+str(len(stream)).encode()+b" >>\nstream\n"+stream+b"endstream"]
    result = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index,obj in enumerate(objects,1):
        offsets.append(len(result))
        result.extend(f"{index} 0 obj\n".encode()+obj+b"\nendobj\n")
    end = len(result)
    result.extend(b"xref\n0 6\n0000000000 65535 f \n")
    result.extend("".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode())
    result.extend(f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{end}\n%%EOF\n".encode())
    return result


def generate(root,count,offset=0):
    root.mkdir()
    good = []
    for idx in range(offset,offset+count):
        style=idx%3
        scan=idx%5 in (0,1,2)
        invoice=f"INV-26-{idx:06d}"
        order=f"PO-26-{idx:06d}"
        date=f"2026-{idx%11+1:02d}-{idx%27+1:02d}"
        net=Decimal(8142+137*idx)/100
        tax=(net/10).quantize(Decimal("0.01"))
        total=net+tax
        issue=("bad_total" if idx%67==1 else "wrong_po" if idx%73==2
               else "duplicate" if idx%89==3 else None)
        if issue=="bad_total":
            total+=Decimal("0.11")
        original_order=order
        if issue=="wrong_po":
            original_order=f"PO-99-{idx:06d}"
        if issue=="duplicate" and idx==offset:
            issue=None
        if issue=="duplicate":
            invoice=f"INV-26-{idx-1:06d}"
        formatted=[invoice,order,date, *(f"USD {value:,.2f}" for value in (net,tax,total))]
        name=f"doc-{idx:04d}"
        source=root/f"{name}.pdf"
        source.write_bytes(pdf(zip(LABELS[style],formatted),style,idx%17==1))
        if scan:
            subprocess.run(["pdftoppm","-r",str(115 if idx%17==1 else 150),"-png","-gray","-singlefile",
                            str(source),str(root/name)],check=True,stdout=subprocess.DEVNULL)
            source.unlink()
            source=root/f"{name}.png"
        good.append({"id":idx,"file":source.name,"sha256":sha256(source.read_bytes()).hexdigest(),
                     "mode":"scan" if scan else "native","invoice":invoice,"po":order,"date":date,
                     "net":str(net.quantize(Decimal(".01"))),"tax":str(tax),"total":str(total),
                     "reference_po":original_order,"exception":issue})
    (root/"truth.json").write_text(json.dumps(good,sort_keys=True,indent=2)+"\n")
    print(len(good),sum(row["mode"]=="scan" for row in good))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("count",type=int)
    p.add_argument("--offset",type=int,default=0)
    a=p.parse_args()
    generate(a.out,a.count,a.offset)
