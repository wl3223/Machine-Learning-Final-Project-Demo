"""
Shared helper functions and decorators.
"""

import os
import random

import numpy as np


def set_reproducibility(seed=42):
	"""
	Configure deterministic seeds for common RNG sources used by the project.
	"""
	os.environ.setdefault("PYTHONHASHSEED", str(seed))
	random.seed(seed)
	np.random.seed(seed)

	try:
		import torch

		torch.manual_seed(seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(seed)
	except Exception:
		pass
