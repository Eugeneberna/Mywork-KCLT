"""
Interim open knowledge source (CC0): Human Disease Ontology (doid.obo).
Writes triples (head, relation, tail) for s03_build_kg.py --triples_csv:
  is_a          disease -> parent disease          (hierarchy)
  has_symptom   disease -> symptom phrase           (parsed from the curated DO definitions)
  synonym       disease -> exact synonym
Use for the pilot only; the final experiments should use UMLS (s03_build_kg.py --umls_dir).

python s03b_doid_triples.py --obo doid.obo --out doid_triples.csv
"""
import argparse, re
import pandas as pd


def parse(path):
    terms, cur = [], None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line == "[Term]":
            cur = {"syn": [], "isa": [], "def": ""}; terms.append(cur); continue
        if line.startswith("["):
            cur = None; continue
        if cur is None or ": " not in line:
            continue
        k, v = line.split(": ", 1)
        if k == "id": cur["id"] = v
        elif k == "name": cur["name"] = v
        elif k == "synonym":
            m = re.match(r'"(.*)" EXACT', v)
            if m: cur["syn"].append(m.group(1))
        elif k == "is_a": cur["isa"].append(v.split(" ")[0])
        elif k == "def": cur["def"] = v
        elif k == "is_obsolete": cur["obs"] = True
    return [t for t in terms if "id" in t and "name" in t and not t.get("obs")]


def main(a):
    T = parse(a.obo); name = {t["id"]: t["name"] for t in T}
    rows = []
    for t in T:
        for p in t["isa"]:
            if p in name: rows.append((t["name"], "is_a", name[p]))
        for s in re.findall(r"has_symptom ([^,\.;]+?)(?=,| and |\.|;|$)", t["def"]):
            s = re.sub(r"\(.*?\)", "", s).strip()
            if 2 < len(s) < 60: rows.append((t["name"], "has_symptom", s))
        for s in t["syn"][:5]:
            rows.append((t["name"], "synonym", s))
    df = pd.DataFrame(rows, columns=["head", "relation", "tail"]).drop_duplicates()
    df.to_csv(a.out, index=False); print(df.relation.value_counts())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--obo", default="doid.obo"); ap.add_argument("--out", default="doid_triples.csv")
    main(ap.parse_args())
