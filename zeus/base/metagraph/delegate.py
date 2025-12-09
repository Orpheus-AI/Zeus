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

from typing import List, Optional, Dict
import bittensor as bt

from zeus.validator.constants import MechanismType, MECHAGRAPH_SIZES
from zeus.base.metagraph.mechagraph import Mechagraph

class MetagraphDelegate(bt.Metagraph):
    """
    Interface between regular/actual Metagraph as on the blockchain,
    and the Zeus Mechagraph system.
    """

    def __init__(
            self,
            sync: bool = True,
            mechagraph_sizes: Dict[MechanismType, int] = MECHAGRAPH_SIZES,
            *args,
            **kwargs
    ):
        kwargs["sync"] = False # don't sync until mechagraphs are created
        super().__init__(*args, **kwargs)
        self.mechagraphs = [
            Mechagraph(
                size=size,
                parent=self,
                mechanism=mtype,
            )
            for mtype, size in mechagraph_sizes.items()
        ]
        if sync:
            self.sync()

    def sync(
        self,
        *args,
        **kwargs
    ):
        """
        Delegate syncing to mechagraphs after syncing self.
        """
        super().sync(*args, **kwargs)
        for subgraph in self.mechagraphs:
            subgraph._sync(block=self.block.item(), lite=self.lite)
