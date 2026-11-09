"""Run the independent color CNN from the project environment."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from color_editing.color_cnn import main


if __name__ == "__main__":
    main()
