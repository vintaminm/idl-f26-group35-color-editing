import argparse

from color_editing.data import contact_sheet, describe_sample, recolor_lab, recolor_lab_light, select_samples

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="/ocean/projects/cis260330p/shared/data/CelebAMask-HQ")
parser.add_argument("--out", required=True)
args = parser.parse_args()

imgs, labels = [], []
for region, n in (("hair", 4), ("cloth", 2)):
    for s in select_samples(args.data, region, split="val", limit=n):
        src = describe_sample(s)["source_color"]
        for t in (["red", "blue", "brown"] if region == "hair" else ["red", "pink"]):
            imgs += [s["image"], recolor_lab(s["image"], s["mask"], t), recolor_lab_light(s["image"], s["mask"], t)]
            labels += ["%d %s %s" % (s["image_id"], region, src), "B0 -> " + t, "light -> " + t]
contact_sheet(imgs, labels, width=200).save(args.out)
print("saved", len(imgs))
