"""How much does COMET penalise word order vs meaning errors? Variants per sentence,
each scored against the FLORES reference, reported as the difference to the correct one."""
import json, glob, sys
from comet import download_model, load_from_checkpoint

REPO = "."  # run from the repository folder

def load(s):
    f = [x for x in sorted(glob.glob(f"{REPO}/results/pipeline/{s}/*.jsonl")) if sum(1 for _ in open(x)) == 712][-1]
    return {r["idx"]: r for r in map(json.loads, open(f))}

g1, g3, h1 = load("gemma-e2b-q4_waitk1"), load("gemma-e2b-q4_la-s3"), load("hymt2-q4_waitk1")

V = {  # C correct, L German word order, V wrong verb, N negation, E wrong entity/number
300: dict(
 C="The commission was Martelly's answer to the widespread anti-regime protests that began in October.",
 L="The commission was Martelly's answer to the widespread anti-regime protests, which in October began.",
 V="The commission was Martelly's answer to the widespread anti-regime protests that ended in October.",
 N="The commission was not Martelly's answer to the widespread anti-regime protests that began in October.",
 E="The commission was Martelly's answer to the widespread anti-regime protests that began in December."),
303: dict(
 C="Japan's Ministry of Economy, Trade and Industry (METI) said it was aware of 27 accidents involving the devices.",
 L="Japan's Ministry of Economy, Trade and Industry (METI) said that to it 27 accidents involving the devices known were.",
 V="Japan's Ministry of Economy, Trade and Industry (METI) said it had caused 27 accidents involving the devices.",
 N="Japan's Ministry of Economy, Trade and Industry (METI) said it was not aware of 27 accidents involving the devices.",
 E="Japan's Ministry of Economy, Trade and Industry (METI) said it was aware of 72 accidents involving the devices."),
310: dict(
 C="Rolando Mendoza shot at the tourists with his M16 rifle.",
 L="Rolando Mendoza shot with his M16 rifle at the tourists.",
 V="Rolando Mendoza handed his M16 rifle to the tourists.",
 N="Rolando Mendoza did not shoot at the tourists with his M16 rifle.",
 E="Rolando Mendoza shot at the soldiers with his M16 rifle."),
330: dict(
 C="Comets may have been a source of the Earth's water, along with organic matter that can form proteins and support life.",
 L="Comets were possibly a source for the water supply of the Earth, together with organic matter, that proteins form and life enable can.",
 V="Comets may have been a source of the Earth's water, along with organic matter that can destroy proteins and support life.",
 N="Comets cannot have been a source of the Earth's water, along with organic matter that can form proteins and support life.",
 E="Asteroids may have been a source of the Earth's water, along with organic matter that can form proteins and support life."),
350: dict(
 C="The cities of Harappa and Mohenjo-daro had a flush toilet in almost every house, connected to a sophisticated sewage system.",
 L="The cities of Harappa and Mohenjo-daro had in almost every house a flush toilet, which to a sophisticated sewage system connected was.",
 V="The cities of Harappa and Mohenjo-daro lacked a flush toilet in almost every house, connected to a sophisticated sewage system.",
 N="The cities of Harappa and Mohenjo-daro had a flush toilet in almost no house, connected to a sophisticated sewage system.",
 E="The cities of Harappa and Mohenjo-daro had a flush toilet in almost every temple, connected to a sophisticated sewage system."),
360: dict(
 C="Finally, there are many small cats (including free-roaming house cats) that eat much smaller prey such as insects, rodents, lizards and birds.",
 L="Finally gives it still many small cats (including free-roaming house cats), which much smaller prey such as insects, rodents, lizards and birds eat.",
 V="Finally, there are many small cats (including free-roaming house cats) that avoid much smaller prey such as insects, rodents, lizards and birds.",
 N="Finally, there are many small cats (including free-roaming house cats) that do not eat much smaller prey such as insects, rodents, lizards and birds.",
 E="Finally, there are many small cats (including free-roaming house cats) that eat much larger prey such as insects, rodents, lizards and birds."),
}
REAL = [(360, "Gemma wait-1", g1), (360, "Hy-MT2 wait-1", h1), (370, "Gemma wait-1", g1),
        (370, "Gemma LA s3", g3), (370, "Hy-MT2 wait-1", h1), (300, "Hy-MT2 wait-1", h1), (304, "Gemma wait-1", g1)]

rows, keys = [], []
for i, vs in V.items():
    for k, mt in vs.items():
        rows.append(dict(src=g1[i]["src"], mt=mt, ref=g1[i]["ref"])); keys.append((i, k))
for i, name, d in REAL:
    rows.append(dict(src=d[i]["src"], mt=d[i]["hyp"], ref=d[i]["ref"])); keys.append((i, name))

model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
scores = model.predict(rows, batch_size=8, gpus=0, num_workers=1, progress_bar=False).scores
S = dict(zip(keys, scores))

print("sentence  C      L      V      N      E     (differences to C)")
for i in V:
    c = S[(i, "C")]
    print(f"{i}     {c:.3f}  " + "  ".join(f"{S[(i, k)] - c:+.3f}" for k in "LVNE"))
print("\nmean      " + "       " + "  ".join(
    f"{sum(S[(i, k)] - S[(i, 'C')] for i in V) / len(V):+.3f}" for k in "LVNE"))
print("\nreal outputs")
for (i, name, d), key in zip(REAL, keys[-len(REAL):]):
    print(f"{i} {name:14} {S[key]:.3f}  {d[i]['hyp']}")
