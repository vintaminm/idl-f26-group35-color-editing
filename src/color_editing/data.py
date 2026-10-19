"""Small CelebAMask-HQ helpers; original images and masks are never modified."""

from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps


# Initial named-color prototypes, intended for visual review before full training.
COLOR_RGB = {
    "black": (25, 25, 25),
    "brown": (115, 75, 50),
    "blond": (210, 180, 110),
    "yellow": (210, 180, 110),
    "red": (175, 45, 40),
    "blue": (45, 80, 180),
    "green": (55, 120, 70),
    "purple": (110, 60, 150),
    "pink": (210, 110, 150),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
}
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])
_D65 = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _rgb_to_lab(rgb):
    """Convert an array of sRGB values in [0, 255] to CIELAB under D65."""
    srgb = np.asarray(rgb, dtype=np.float64) / 255.0
    linear = np.where(
        srgb <= 0.04045, srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    xyz = (linear @ _RGB_TO_XYZ.T) / _D65
    f = np.where(
        xyz > _DELTA ** 3, np.cbrt(xyz),
        xyz / (3 * _DELTA ** 2) + 4.0 / 29.0,
    )
    return np.stack([
        116 * f[..., 1] - 16,
        500 * (f[..., 0] - f[..., 1]),
        200 * (f[..., 1] - f[..., 2]),
    ], axis=-1)


def _lab_to_rgb(lab):
    """Convert CIELAB D65 to uint8 sRGB, clipping out-of-gamut values."""
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], axis=-1)
    xyz = np.where(
        f > _DELTA, f ** 3,
        3 * _DELTA ** 2 * (f - 4.0 / 29.0),
    ) * _D65
    linear = xyz @ _XYZ_TO_RGB.T
    srgb = np.where(
        linear <= 0.0031308, 12.92 * linear,
        1.055 * np.maximum(linear, 0) ** (1 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def split_ids(seed=35):
    """Return deterministic, disjoint image-ID splits (26k/2k/2k).

    Split image IDs before selecting regions, so hair and clothing from one
    photograph always belong to the same split. Requires no generated files.
    """
    ids = np.random.default_rng(seed).permutation(30000).tolist()
    return {"train": ids[:26000], "val": ids[26000:28000], "test": ids[28000:]}


def _mask_path(data_root, image_id, region):
    if region not in {"hair", "cloth"}:
        raise ValueError("region must be 'hair' or 'cloth'")
    if not isinstance(image_id, (int, np.integer)) or not 0 <= image_id < 30000:
        raise ValueError("image_id must be an integer between 0 and 29999")
    return (Path(data_root) / "CelebAMask-HQ-mask-anno" / str(image_id // 2000)
            / f"{image_id:05d}_{region}.png")


def _load_mask(path):
    with Image.open(path) as raw:
        mask = raw.convert("L").resize((512, 512), Image.Resampling.NEAREST)
    return Image.fromarray((np.asarray(mask) > 127).astype(np.uint8) * 255)


def load_sample(data_root, image_id, region="hair", dilation=5):
    """Read one aligned 512-square image and binary region mask.

    White selects pixels to edit. ``dilation`` is an odd filter width in pixels;
    0 or 1 disables dilation. Missing raw masks raise FileNotFoundError.
    """
    if not isinstance(dilation, int) or dilation < 0 or (dilation > 1 and dilation % 2 == 0):
        raise ValueError("dilation must be 0, 1, or a positive odd filter width")
    mask = _load_mask(_mask_path(data_root, image_id, region))
    image_path = Path(data_root) / "CelebA-HQ-img" / f"{image_id}.jpg"
    with Image.open(image_path) as raw:
        image = raw.convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    edit_mask = mask.filter(ImageFilter.MaxFilter(dilation)) if dilation > 1 else mask.copy()
    return {"image": image, "mask": mask, "edit_mask": edit_mask,
            "image_id": int(image_id), "region": region}


def select_samples(data_root, region, split="val", limit=2, seed=35,
                   min_mask_fraction=0.01, dilation=5, max_scan=2000):
    """Select a few qualifying samples within one fixed image-ID split.

    Inspect mask area before reading photographs. Missing region masks are
    skipped; a missing selected photograph remains an explicit dataset error.
    Raise rather than silently return fewer than the requested sample count.
    """
    partitions = split_ids(seed)
    if split not in partitions:
        raise ValueError("split must be 'train', 'val', or 'test'")
    if limit < 0 or max_scan < 0 or not 0 <= min_mask_fraction <= 1:
        raise ValueError("limit/max_scan must be nonnegative and mask fraction in [0, 1]")
    if region not in {"hair", "cloth"}:
        raise ValueError("region must be 'hair' or 'cloth'")
    if limit == 0:
        return []
    samples = []
    for image_id in partitions[split][:max_scan]:
        mask_path = _mask_path(data_root, image_id, region)
        if not mask_path.is_file():
            continue
        mask = _load_mask(mask_path)
        fraction = float(np.mean(np.asarray(mask) > 127))
        if fraction == 0 or fraction < min_mask_fraction:
            continue
        samples.append(load_sample(data_root, image_id, region, dilation))
        if len(samples) == limit:
            return samples
    raise ValueError(
        f"Found {len(samples)} of {limit} requested {region} samples in {split}. "
        "Review min_mask_fraction or increase max_scan."
    )


def describe_sample(sample):
    """Return JSON-safe region statistics and a provisional nearest-color label.

    Distance is Euclidean CIELAB (CIE76), not a learned classification score.
    Patterned clothes, shadows, and highlights can make this label unreliable.
    """
    rgb = np.asarray(sample["image"].convert("RGB"))
    selected = np.asarray(sample["mask"].convert("L")) > 127
    if rgb.shape[:2] != selected.shape:
        raise ValueError("image and mask sizes must match")
    if not np.any(selected):
        raise ValueError("cannot label an empty region")
    median = np.median(_rgb_to_lab(rgb[selected]), axis=0)
    excluded = "yellow" if sample["region"] == "hair" else "blond"
    names = [name for name in COLOR_RGB if name != excluded]
    palette_lab = _rgb_to_lab(np.asarray([COLOR_RGB[name] for name in names]))
    distances = np.linalg.norm(palette_lab - median, axis=1)
    nearest = int(np.argmin(distances))
    return {
        "image_id": int(sample["image_id"]), "region": sample["region"],
        "source_color": names[nearest], "median_lab": median.tolist(),
        "color_distance": float(distances[nearest]),
        "mask_fraction": float(np.mean(selected)),
        "label_method": "heuristic_median_lab_nearest_palette",
    }


def recolor_lab(image, mask, target_color):
    """B0: shift regional a/b medians, retaining L and within-region residuals.

    The target prototype contributes chroma only; this does not lighten black
    hair to blond. RGB gamut clipping can affect color/texture at extreme edits.
    Pixels outside the supplied mask remain exactly equal to the input RGB.
    """
    if target_color not in COLOR_RGB:
        raise ValueError(f"Unknown target color: {target_color}")
    rgb = np.asarray(image.convert("RGB"))
    selected = np.asarray(mask.convert("L")) > 127
    if rgb.shape[:2] != selected.shape:
        raise ValueError("image and mask sizes must match")
    if not np.any(selected):
        raise ValueError("cannot recolor an empty region")
    region_lab = _rgb_to_lab(rgb[selected])
    target_lab = _rgb_to_lab(np.asarray(COLOR_RGB[target_color]))
    region_lab[:, 1:] += target_lab[1:] - np.median(region_lab[:, 1:], axis=0)
    result = rgb.copy()
    result[selected] = _lab_to_rgb(region_lab)
    return Image.fromarray(result)


def contact_sheet(images, labels, width=256):
    """Make a labeled RGB preview with at most three columns; do not save it."""
    images, labels = list(images), list(labels)
    if not images or len(images) != len(labels):
        raise ValueError("provide equal nonempty image and label lists")
    if not isinstance(width, int) or width < 64:
        raise ValueError("width must be an integer of at least 64 pixels")
    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    cell_height = width + 42
    sheet = Image.new("RGB", (columns * width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x, y = (index % columns) * width, (index // columns) * cell_height
        preview = ImageOps.contain(image.convert("RGB"), (width, width))
        sheet.paste(preview, (x + (width - preview.width) // 2,
                              y + (width - preview.height) // 2))
        lines = textwrap.wrap(str(label), max(12, width // 7))[:2]
        draw.multiline_text((x + 5, y + width + 5), "\n".join(lines), fill="black")
    return sheet
