"""Build the LaTeX table bodies in tables/ from the Python results CSVs.
Usage:  python3 make_tables.py   (reads ../results; writes tables/ and supplementary.tex here)"""
import csv, collections, statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent; R = HERE.parent / "results"; OUT = HERE / "tables"; OUT.mkdir(exist_ok=True)
S = {(r["condition"], r["method"]): r for r in csv.DictReader(open(R / "summary.csv"))}
T = list(csv.DictReader(open(R / "paired_tests.csv")))
SZ, MO, S2M, M2S = "In-domain Shenzhen", "In-domain Montgomery", "Shenzhen -> Montgomery", "Montgomery -> Shenzhen"
MAIN = ["Vanilla", "Focal", "Focal+Tversky", "Sym-AG", "AG-only", "Full"]

def pm(cond, m, k):
    r = S[(cond, m)]
    return f"${float(r[k + '_mean']):.3f} \\pm {float(r[k + '_sd']):.3f}$"

def mean(cond, m, k): return float(S[(cond, m)][k + "_mean"])

def best(cond, methods, k):
    return max(methods, key=lambda m: mean(cond, m, k))

def write(name, lines): (OUT / name).write_text("\n".join(lines) + "\n")

# --- in-domain tables (six main configurations) -----------------------------
for cond, name in ((SZ, "indomain_shenzhen.tex"), (MO, "indomain_montgomery.tex")):
    b = {k: best(cond, MAIN, k) for k in ("ER", "PG", "IoU")}
    rows = []
    for m in MAIN:
        cells = [pm(cond, m, "AUC")]
        for k in ("ER", "PG", "IoU"):
            c = pm(cond, m, k)
            cells.append(c.replace("$", "$\\mathbf{", 1)[:-1] + "}$" if b[k] == m else c)
        rows.append(f"{m} & " + " & ".join(cells) + " \\\\")
    write(name, rows)

# --- transfer tables ----------------------------------------------------------
D = {}
for line in open(R / "figures" / "tables.tex"):
    pass
def delta_rows(src_cond, tr_cond, name):
    # Delta AUC mean +/- SD comes from tables.tex (computed per seed there)
    txt = (R / "figures" / "tables.tex").read_text()
    key = "shenzhen -> montgomery" if tr_cond == S2M else "montgomery -> shenzhen"
    block = txt.split(f"% ---- Transfer: {key} ----")[1].split("\\bottomrule")[0]
    d = {}
    for l in block.splitlines():
        if "&" in l and "textbf" not in l:
            p = [c.strip() for c in l.rstrip("\\ ").split("&")]
            d[p[0]] = p[3]
    rows = [f"{m} & {pm(src_cond, m, 'AUC')} & {pm(tr_cond, m, 'AUC')} & {d[m]} & {pm(tr_cond, m, 'ER')} & {pm(tr_cond, m, 'IoU')} \\\\"
            for m in MAIN]
    write(name, rows)
delta_rows(SZ, S2M, "transfer_s2m.tex")
delta_rows(MO, M2S, "transfer_m2s.tex")

# --- clamp and rho ablations ---------------------------------------------------
def abl(methods, name, labels=None):
    rows = []
    for ds_label, ind, tr in (("Shenzhen", SZ, S2M), ("Montgomery", MO, M2S)):
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{Trained on {ds_label}}}}} \\\\")
        for i, m in enumerate(methods):
            lab = labels[i] if labels else m
            rows.append(f"\\quad {lab} & {pm(ind, m, 'AUC')} & {pm(tr, m, 'AUC')} & {pm(ind, m, 'ER')} & {pm(ind, m, 'PG')} & {pm(ind, m, 'IoU')} \\\\")
    write(name, rows)
abl(["Focal+Tversky", "Full", "Full (no clamp)", "Sym-AG"], "ablation_clamp.tex",
    ["Focal+Tversky (no AG term)", "Full ($\\rho = 0.75$, clamp)", "Full, no clamp ($\\rho = 0.75$)", "Sym-AG ($\\rho = 1$, no clamp)"])
abl(["AG-only rho=0.25", "AG-only rho=0.50", "AG-only", "AG-only rho=1.00"], "ablation_rho.tex",
    ["$\\rho = 0.25$", "$\\rho = 0.50$", "$\\rho = 0.75$ (default)", "$\\rho = 1.00$"])
abl(["Focal+Tversky (eroded masks)", "Focal+Tversky", "Focal+Tversky (dilated masks)",
     "AG-only (eroded masks)", "AG-only", "AG-only (dilated masks)"], "mask_perturb.tex",
    ["Focal+Tversky, eroded", "Focal+Tversky, original", "Focal+Tversky, dilated",
     "AG-only, eroded", "AG-only, original", "AG-only, dilated"])
abl(["Focal", "AG-only", "Masked input"], "masked_input.tex")

# --- occlusion ------------------------------------------------------------------
occ = collections.defaultdict(list)
for r in csv.DictReader(open(R / "occlusion.csv")):
    occ[(r["dataset"], r["method"])].append((float(r["occER"]), float(r["occPG"]), float(r["occIoU"])))
names = [("vanilla", "Vanilla"), ("focal", "Focal"), ("focal_tversky", "Focal+Tversky"), ("sym_noclamp", "Sym-AG"),
         ("anat_only", "AG-only"), ("full", "Full"), ("masked_input", "Masked input")]
rows = []
for key, lab in names:
    cells = []
    for ds, cond in (("shenzhen", SZ), ("montgomery", MO)):
        v = list(zip(*occ[(ds, key)]))
        cells.append(pm(cond, lab, "ER"))
        cells.append(f"${st.mean(v[0]):.3f} \\pm {st.stdev(v[0]):.3f}$")
    rows.append(f"{lab} & " + " & ".join(cells) + " \\\\")
write("occlusion.tex", rows)

# --- paired AUC tests ------------------------------------------------------------
rows = []
for cond in (SZ, MO, S2M, M2S):
    sub = [t for t in T if t["condition"] == cond and t["metric"] == "AUC"]
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{cond.replace('->', '$\\rightarrow$')}}}}} \\\\")
    for t in sub:
        comp = t["comparison"].replace(" - ", " $-$ ")
        rows.append(f"\\quad {comp} & ${float(t['diff']):+.3f}$ & $[{float(t['ci_low']):+.3f},\\ {float(t['ci_high']):+.3f}]$ & "
                    f"${float(t['wilcoxon_p']):.3f}$ & ${float(t['holm_p']):.3f}$ \\\\")
write("paired_auc.tex", rows)
print("wrote", sorted(p.name for p in OUT.glob("*.tex")))


# ============================ round 2 =========================================
import math, itertools
E = R / "eval"

def read_keyed(path):
    return {(r["condition"], r["method_name"]): r for r in csv.DictReader(open(path))}

REL = {a: read_keyed(E / f"reliance_{a}.csv") for a in ("resnet50", "convnext_tiny")}
CLIN = {a: read_keyed(E / f"clinical_{a}.csv") for a in ("resnet50", "convnext_tiny")}

def rpm(tab, cond, m, k):
    r = tab[(cond, m)]
    return f"${float(r[k + '_mean']):.3f} \\pm {float(r[k + '_sd']):.3f}$"

# loss variants (ResNet-50)
abl(["Focal", "AG-only", "AG-only (top-k outside)", "AG-only (saliency mass)"], "variants.tex",
    ["Focal (no spatial term)", "AG-only (outside mean; default)", "AG-only, top-$k$ outside", "AG-only, saliency mass"])

# occlusion at three patch sizes (replaces the single-size table)
OCC_M = ["Vanilla", "Focal", "AG-only", "AG-only (top-k outside)", "AG-only (saliency mass)",
         "Focal+Tversky", "Sym-AG", "Full", "Masked input"]
rows = []
for lab, cond in (("Shenzhen", SZ), ("Montgomery", MO)):
    rows.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{lab}}}}} \\\\")
    for m in OCC_M:
        rows.append(f"\\quad {m.replace('top-k', 'top-$k$')} & {pm(cond, m, 'ER')} & " +
                    " & ".join(rpm(REL['resnet50'], cond, m, f'occ_er_{p}') for p in (16, 32, 64)) + " \\\\")
write("occlusion.tex", rows)

# region ablation (ResNet-50, in-domain)
REG_M = ["Vanilla", "Focal", "AG-only", "AG-only (top-k outside)", "Focal+Tversky", "Full", "Masked input"]
rows = []
for lab, cond in (("Shenzhen", SZ), ("Montgomery", MO)):
    rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}}}} \\\\")
    for m in REG_M:
        rows.append(f"\\quad {m.replace('top-k', 'top-$k$')} & " + " & ".join(
            rpm(REL['resnet50'], cond, m, k) for k in ("auc_full", "auc_lungs_zero", "auc_outside_zero", "auc_lungs_blur", "auc_outside_blur")) + " \\\\")
write("region.tex", rows)

# ConvNeXt-Tiny: alignment per seed from the per-image file
agg = collections.defaultdict(lambda: collections.defaultdict(list))
for r in csv.DictReader(open(E / "per_image_convnext_tiny.csv")):
    key = (r["method"], r["dataset"], r["split"], int(r["seed"]))
    for k in ("er", "pg", "iou"):
        v = float(r[k])
        if not math.isnan(v):
            agg[key][k].append(v)
def cx(method, ds, split, k):
    v = [st.mean(a[k]) for (m, d, s, _), a in agg.items() if (m, d, s) == (method, ds, split)]
    return f"${st.mean(v):.3f} \\pm {st.stdev(v):.3f}$"
rows = []
for lab, ds, ind, tr in (("Shenzhen", "shenzhen", SZ, S2M), ("Montgomery", "montgomery", MO, M2S)):
    rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{Trained on {lab}}}}} \\\\")
    for key, name in (("focal", "Focal"), ("anat_only", "AG-only"), ("focal_tversky", "Focal+Tversky"), ("full", "Full")):
        rows.append(f"\\quad {name} & {rpm(CLIN['convnext_tiny'], ind, name, 'AUC')} & {rpm(CLIN['convnext_tiny'], tr, name, 'AUC')} & "
                    f"{cx(key, ds, 'test', 'er')} & {cx(key, ds, 'test', 'pg')} & {cx(key, ds, 'test', 'iou')} & "
                    f"{rpm(REL['convnext_tiny'], ind, name, 'occ_er_32')} \\\\")
write("convnext.tex", rows)
print("round 2 tables written")


# ============================ supplementary tables ===============================
def supp_clinical(arch, methods):
    rows = []
    for cond in (SZ, MO, S2M, M2S):
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{cond.replace('->', '$\\rightarrow$')}}}}} \\\\")
        for m in methods:
            rows.append(f"\\quad {m} & " + " & ".join(rpm(CLIN[arch], cond, m, k) for k in
                        ("AUC", "AUPRC", "sens_at_spec90", "spec_at_sens90", "Brier")) + " \\\\")
    return rows

def supp_delins(arch, methods):
    rows = []
    for lab, cond in (("Shenzhen", SZ), ("Montgomery", MO)):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for m in methods:
            rows.append(f"\\quad {m.replace('top-k', 'top-$k$')} & " + " & ".join(rpm(REL[arch], cond, m, k) for k in
                        ("del_cam", "del_rand", "del_gain", "ins_cam", "ins_rand", "ins_gain")) + " \\\\")
    return rows

RES_M = MAIN + ["Full (no clamp)", "AG-only (top-k outside)", "AG-only (saliency mass)", "Masked input"]
CNX_M = ["Focal", "AG-only", "Focal+Tversky", "Full"]
write("supp_clinical_resnet50.tex", [r.replace("top-k", "top-$k$") for r in supp_clinical("resnet50", RES_M)])
write("supp_clinical_convnext.tex", supp_clinical("convnext_tiny", CNX_M))
write("supp_delins_resnet50.tex", supp_delins("resnet50", RES_M))
write("supp_delins_convnext.tex", supp_delins("convnext_tiny", CNX_M))

SUPP = r"""\documentclass[10pt]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{booktabs,adjustbox,amsmath}
\usepackage[T1]{fontenc}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\arraystretch}{1.15}
\begin{document}
\begin{center}
{\Large\bfseries Supplementary Material}\\[4pt]
{\large Guidance versus Confinement: What Anatomical Supervision of Grad-CAM Does and Does Not Change}\\[4pt]
Numan Aslam, Ghulam Mustafa and Adnan N. Qureshi
\end{center}
\noindent All values are mean $\pm$ 1 SD across 8 random seeds. ``Transfer'' rows evaluate on the full other dataset.

%CLIN1
%CLIN2
%DEL1
%DEL2
\end{document}
"""
def block(caption, cols, header, name):
    body = open(OUT / name).read().rstrip()
    return ("\\begin{table}[!ht]\n\\centering\\footnotesize\n\\caption{" + caption + "}\n\\begin{adjustbox}{max width=\\textwidth}\n"
            "\\begin{tabular}{" + cols + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n")
ch = "\\textbf{Method} & \\textbf{AUC} & \\textbf{AUPRC} & \\textbf{Sens. at 90\\% spec.} & \\textbf{Spec. at 90\\% sens.} & \\textbf{Brier}"
dh = ("\\textbf{Method} & \\textbf{Deletion (CAM)} & \\textbf{Deletion (random)} & \\textbf{Deletion gain} & "
      "\\textbf{Insertion (CAM)} & \\textbf{Insertion (random)} & \\textbf{Insertion gain}")
dn = (" Deletion and insertion AUC are the areas under the TB-probability curve as pixels are removed from, or restored into, "
      "TB test images in order of decreasing Grad-CAM value or in random order. Deletion gain is random minus CAM and insertion gain is CAM minus random; "
      "positive values mean that the Grad-CAM ranks the pixels the classifier uses better than chance.")
supp = (SUPP.replace("%CLIN1", block("Classification metrics, ResNet-50.", "l c c c c c", ch, "supp_clinical_resnet50.tex"))
            .replace("%CLIN2", block("Classification metrics, ConvNeXt-Tiny.", "l c c c c c", ch, "supp_clinical_convnext.tex"))
            .replace("%DEL1", block("Deletion and insertion analysis on the in-domain test split, ResNet-50." + dn, "l c c c c c c", dh, "supp_delins_resnet50.tex"))
            .replace("%DEL2", block("Deletion and insertion analysis on the in-domain test split, ConvNeXt-Tiny.", "l c c c c c c", dh, "supp_delins_convnext.tex")))
(HERE / "supplementary.tex").write_text(supp)
print("wrote supplementary.tex")
