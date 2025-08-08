from typing import Union, Any, AsyncGenerator, Callable, List, Tuple
import asyncio
import torch
import bittensor as bt

class MoEDendrite(bt.Dendrite):
    """
    Custom dendrite to get to implement a Mixture-of-Experts Dendrite.
    It queries a certain amount of axons, and filters their output based on a predefined filter.
    All outputs that pass the filter are averaged using the miner's rank as a weighting factor..

    This dendrite has a desired timeout, after which it will cut off as soon as it has responses.
    Note this dendrite has to be async for the above to make sense,
    And that it does not support streaming.
    """

    def __init__(
            self,
            cutoff_percent: float = 0.6,
            *args,
            **kwargs,
    ):
        """
        Once more than desired_timeout seconds have passed and we have responses,
        we use those. If there are no responses, we wait until first one
        """
        super().__init__(*args, **kwargs)
        self.cutoff_percent = cutoff_percent

    async def forward(
        self,
        uids: List[int],
        metagraph: bt.Metagraph,
        synapse: "bt.Synapse",
        filter: Callable[[Union["bt.Synapse", Any]], bool],
        timeout: float = 12,
        deserialize=True,
    ) -> Tuple[int, Union["AsyncGenerator[Any, Any]", "bt.Synapse"]]:
        """
        See parent class bt.Dendrite for original documentation. 
        Does not support streaming behaviour, since this dendrite is already greedy.
        Is necessarily asynchronous.

        If deserialize is True, the filter is applied on the output.
        If deserialize is False, the filter is applied on the whole synapse.

        Returns: A single Synapse, the first one that responded and passes filter
        """

        async def query_uids_eager(
            filter: Callable[["bt.Synapse"], bool],
        ) -> Union["AsyncGenerator[Any, Any]", "bt.Synapse"]:
            
            async def single_uid_response(
                target_uid: int,
            ) -> Union["AsyncGenerator[Any, Any]", "bt.Synapse"]:
                result = await self.call(
                    target_axon=metagraph.axons[target_uid],
                    synapse=synapse.model_copy(),  # type: ignore
                    timeout=timeout,
                    deserialize=deserialize,
                )
                return target_uid, result
            
            
            responses = []

            for task in asyncio.as_completed(
                [single_uid_response(target_uid) for target_uid in uids]
            ):
                task
                uid, result = await task
                if filter(result):
                    responses.append((uid, result))
                    if len(responses) >= len(uids) * self.cutoff_percent:
                        return responses
                    
            return responses

        # Get responses eagerily, so return first one that succeeds
        response = await query_uids_eager(filter)
        return response