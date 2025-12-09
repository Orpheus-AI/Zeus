# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pathlib
import os
import re
import bittensor as bt
import numpy as np
from datetime import datetime

from zeus.data.base.sample import BaseSample
from zeus.utils.coordinates import get_bbox, slice_bbox


class DifficultyLoader:

    def __init__(
        self,
        data_folder: str = "weights/",  # part starting from the folder this script itself is located in
    ):
        data_folder = pathlib.Path(os.path.abspath(__file__)).parent / data_folder
        weight_files = data_folder.glob("*.npy")

        self.difficulty_matrices = {}

        for weight_file in weight_files:
            weight_matrix = np.load(weight_file)
            month = re.search(r"difficulty_(\d+)", weight_file.stem)
            if not month:
                bt.logging.error(
                    f"Could not parse weight file {weight_file.stem}! This means another month will be used for month {month}."
                )
                continue
            month = month.group(1)
            self.difficulty_matrices[int(month)] = weight_matrix

        if not self.difficulty_matrices:
            raise AssertionError(
                f"No difficulty matrices found in {data_folder}, scoring cannot be calculated!"
            )

    def get_difficulty_grid(self, sample: BaseSample) -> np.ndarray:
        """
        Returns the difficulties for each grid location in the given sample.
        """
        start_month = datetime.fromtimestamp(sample.start_timestamp).month

        # months don't differ that much. If the specified month is not found, use the first one in storage.
        difficulty_matrix = self.difficulty_matrices.get(
            start_month,
            self.difficulty_matrices[list(self.difficulty_matrices.keys())[0]],
        )

        difficulty_grid = slice_bbox(difficulty_matrix, sample.get_bbox())

        return difficulty_grid
