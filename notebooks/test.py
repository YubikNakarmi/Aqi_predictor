from pathlib import Path
import sys

proj_root = Path.cwd().resolve().parent
idff_dir = proj_root / "scripts"
sys.path.append(str(idff_dir))

import data_hourly_preprocessing as data_clean_hourly