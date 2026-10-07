"""
Step 3 - Build the medical knowledge graph.

MODE A (recommended, matches the paper): UMLS Metathesaurus (includes SNOMED CT US).
  Requires a free UMLS licence (https://uts.nlm.nih.gov). Download the UMLS "Full Release"
  and point --umls_dir at the folder containing MRCONSO.RRF, MRREL.RRF and MRSTY.RRF.

  python s03_build_kg.py --umls_dir /path/to/2025AA/META --out kg

MODE B (no UMLS licence yet / open alternative): a CSV of triples
  head,relation,tail     e.g.  Dengue fever,has_symptom,Fever
  python s03_build_kg.py --triples_csv my_triples.csv --out kg
  Triples must come from an external source (ontology, textbook, curated list),
  NEVER from the Symptom2Disease narratives themselves, otherwise labels leak into the KG.

Output
  kg/nodes.csv   nid, cui, name, synonyms ('|'-joined; ENG/SPA/FRE where available), is_disease, label
  kg/edges.csv   src, dst, rel      (undirected; both directions are added at load time)
  kg/seed_map.csv  the UMLS concept chosen for each of the 24 diseases - CHECK THIS BY HAND
  kg/kg_stats.json
"""
import argparse, os, collections
import pandas as pd
from common import DISEASE_CONCEPTS, DISEASE_ALIASES, save_json

# Semantic types kept as nodes (UMLS TUIs)
TUI_KEEP = {"T047", "T184", "T033", "T046", "T037", "T019", "T020", "T191", "T048", "T190",
            "T023", "T029", "T031", "T121", "T061"}
TUI_SYMPTOM = {"T184"}
REL_KEEP = {"PAR", "CHD", "RB", "RN", "RO"}
RELA_CLINICAL = {"has_manifestation", "manifestation_of", "has_associated_finding",
                 "associated_finding_of", "has_finding_site", "finding_site_of", "cause_of",
                 "due_to", "associated_with", "has_associated_morphology",
                 "associated_morphology_of", "may_treat", "may_be_treated_by", "clinically_associated_with",
                 "has_sign_or_symptom", "sign_or_symptom_of", "disease_has_finding", "is_finding_of_disease"}
LANG_KEEP = {"ENG", "SPA", "FRE"}
CONSO_COLS = ["CUI", "LAT", "TS", "LUI", "STT", "SUI", "ISPREF", "AUI", "SAUI", "SCUI", "SDUI",
              "SAB", "TTY", "CODE", "STR", "SRL", "SUPPRESS", "CVF"]
REL_COLS = ["CUI1", "AUI1", "STYPE1", "REL", "CUI2", "AUI2", "STYPE2", "RELA", "RUI", "SRUI",
            "SAB", "SL", "RG", "DIR", "SUPPRESS", "CVF"]


def rrf(path, cols, usecols, chunksize=2_000_000):
    return pd.read_csv(
        path, sep="|", header=None, names=cols + ["_"], usecols=usecols, dtype=str,
        quoting=3, chunksize=chunksize, na_filter=False, encoding="utf-8")


def build_umls(a):
    D = a.umls_dir
    print("[1/4] MRSTY: semantic types")
    sty = pd.read_csv(f"{D}/MRSTY.RRF", sep="|", header=None, usecols=[0, 1], names=["CUI", "TUI"],
                      dtype=str, quoting=3)
    keep_cui = set(sty[sty.TUI.isin(TUI_KEEP)].CUI)
    symptom_cui = set(sty[sty.TUI.isin(TUI_SYMPTOM)].CUI)

    print("[2/4] MRCONSO pass 1: locate the 24 disease concepts + symptom names")
    want = {v.lower(): k for k, v in DISEASE_CONCEPTS.items()}
    seed_hits = collections.defaultdict(collections.Counter)
    sym_pref = {}
    for ch in rrf(f"{D}/MRCONSO.RRF", CONSO_COLS, ["CUI", "LAT", "TS", "STT", "ISPREF", "SAB", "STR", "SUPPRESS"]):
        ch = ch[(ch.LAT == "ENG") & (ch.SUPPRESS == "N") & ch.CUI.isin(keep_cui)]
        low = ch.STR.str.lower()
        hit = ch[low.isin(want)]
        for cui, s, ts in zip(hit.CUI, hit.STR.str.lower(), hit.TS):
            seed_hits[want[s]][cui] += 2 if ts == "P" else 1
        sp = ch[ch.CUI.isin(symptom_cui) & (ch.TS == "P") & (ch.STT == "PF") & (ch.ISPREF == "Y")
                & ch.SAB.isin(["SNOMEDCT_US", "MSH", "MDR", "HPO"])]
        for cui, s in zip(sp.CUI, sp.STR):
            if len(s.split()) <= 4:
                sym_pref.setdefault(cui, s)
    seed = {}
    for label in DISEASE_CONCEPTS:
        if a.seed_map and os.path.exists(a.seed_map):
            break
        if seed_hits[label]:
            seed[label] = seed_hits[label].most_common(1)[0][0]
        else:
            print(f"  WARNING: no UMLS match for '{label}' - add it to a --seed_map CSV")
    if a.seed_map and os.path.exists(a.seed_map):
        sm = pd.read_csv(a.seed_map); seed = dict(zip(sm.label, sm.cui))
    seed_cuis = set(seed.values())

    print("[3/4] MRREL: relations")
    edges = []
    for ch in rrf(f"{D}/MRREL.RRF", REL_COLS, ["CUI1", "REL", "CUI2", "RELA", "SUPPRESS"]):
        ch = ch[(ch.SUPPRESS == "N") & ch.REL.isin(REL_KEEP) & (ch.CUI1 != ch.CUI2)
                & ch.CUI1.isin(keep_cui) & ch.CUI2.isin(keep_cui)]
        ch = ch[(ch.REL != "RO") | ch.RELA.isin(RELA_CLINICAL)]
        edges.append(ch[["CUI1", "CUI2", "REL", "RELA"]])
    E = pd.concat(edges, ignore_index=True)
    # node set: diseases + 1-hop neighbours of diseases + symptom concepts
    hop1 = set(E[E.CUI1.isin(seed_cuis)].CUI2) | set(E[E.CUI2.isin(seed_cuis)].CUI1)
    if a.max_hop1 and len(hop1) > a.max_hop1:   # keep clinically-typed neighbours first
        rel_n = E[(E.CUI1.isin(seed_cuis) | E.CUI2.isin(seed_cuis)) & (E.RELA != "")]
        pri = (set(rel_n.CUI1) | set(rel_n.CUI2)) & hop1
        hop1 = pri | set(list(hop1 - pri)[: max(0, a.max_hop1 - len(pri))])
    nodes = seed_cuis | hop1 | set(sym_pref)
    E = E[E.CUI1.isin(nodes) & E.CUI2.isin(nodes)]
    E["key"] = [tuple(sorted(x)) for x in zip(E.CUI1, E.CUI2)]
    E = E.drop_duplicates("key")
    deg = collections.Counter(list(E.CUI1) + list(E.CUI2))
    nodes = {c for c in nodes if deg[c] > 0 or c in seed_cuis}

    print("[4/4] MRCONSO pass 2: names and ENG/SPA/FRE synonyms")
    pref, syn = {}, collections.defaultdict(list)
    for ch in rrf(f"{D}/MRCONSO.RRF", CONSO_COLS, ["CUI", "LAT", "TS", "STT", "ISPREF", "STR", "SUPPRESS"]):
        ch = ch[ch.CUI.isin(nodes) & ch.LAT.isin(LANG_KEEP) & (ch.SUPPRESS == "N")]
        for cui, lat, ts, stt, isp, s in zip(ch.CUI, ch.LAT, ch.TS, ch.STT, ch.ISPREF, ch.STR):
            if lat == "ENG" and ts == "P" and stt == "PF" and isp == "Y":
                pref.setdefault(cui, s)
            if len(syn[cui]) < a.max_syn and s not in syn[cui] and len(s) < 80:
                syn[cui].append(s)
    write_kg(a.out, nodes, pref, syn, E[["CUI1", "CUI2", "REL"]].values.tolist(), seed)


def build_triples(a):
    T = pd.read_csv(a.triples_csv).dropna()
    names = sorted(set(T["head"]) | set(T["tail"]))
    cui = {n: f"N{i:06d}" for i, n in enumerate(names)}
    low = {n.lower(): n for n in names}
    seed = {}
    for label, concept in DISEASE_CONCEPTS.items():   # first matching name wins: concept, label, aliases
        for cand in [concept, label] + DISEASE_ALIASES.get(label, []):
            if cand.lower() in low:
                seed[label] = cui[low[cand.lower()]]; break
    missing = [l for l in DISEASE_CONCEPTS if l not in seed]
    for l in missing:  # disease without triples still gets an (isolated) node
        n = DISEASE_CONCEPTS[l]; cui[n] = f"D{len(cui):06d}"; seed[l] = cui[n]; names.append(n)
    pref = {cui[n]: n for n in names}
    syn = {cui[n]: [n] for n in names}
    E = [[cui[h], cui[t], r] for h, r, t in zip(T["head"], T["relation"], T["tail"])]
    write_kg(a.out, set(pref), pref, syn, E, seed)


def write_kg(out, nodes, pref, syn, edges, seed):
    os.makedirs(out, exist_ok=True)
    nodes = sorted(c for c in nodes if c in pref or syn.get(c))
    nid = {c: i for i, c in enumerate(nodes)}
    lab_of = {c: l for l, c in seed.items()}
    N = pd.DataFrame({"nid": range(len(nodes)), "cui": nodes,
                      "name": [pref.get(c, (syn.get(c) or [c])[0]) for c in nodes],
                      "synonyms": ["|".join(syn.get(c, [])) for c in nodes],
                      "is_disease": [int(c in lab_of) for c in nodes],
                      "label": [lab_of.get(c, "") for c in nodes]})
    Ed = pd.DataFrame([(nid[s], nid[d], r) for s, d, r in edges if s in nid and d in nid],
                      columns=["src", "dst", "rel"]).drop_duplicates()
    N.to_csv(f"{out}/nodes.csv", index=False); Ed.to_csv(f"{out}/edges.csv", index=False)
    pd.DataFrame([{"label": l, "cui": c, "name": pref.get(c, "")} for l, c in seed.items()]) \
        .to_csv(f"{out}/seed_map.csv", index=False)
    stats = {"nodes": len(N), "edges": len(Ed), "disease_nodes": int(N.is_disease.sum()),
             "diseases_mapped": len(seed), "edge_types": Ed.rel.value_counts().to_dict()}
    save_json(stats, f"{out}/kg_stats.json"); print(stats)
    print("CHECK kg/seed_map.csv by hand: each of the 24 labels must map to the right concept.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--umls_dir"); ap.add_argument("--triples_csv")
    ap.add_argument("--seed_map", help="optional CSV label,cui to override the automatic disease mapping")
    ap.add_argument("--out", default="kg")
    ap.add_argument("--max_hop1", type=int, default=3000)
    ap.add_argument("--max_syn", type=int, default=12)
    a = ap.parse_args()
    if a.umls_dir: build_umls(a)
    elif a.triples_csv: build_triples(a)
    else: ap.error("give --umls_dir or --triples_csv")
