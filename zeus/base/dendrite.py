from typing import Union, Any, AsyncGenerator, Dict, Tuple, List, Awaitable, TypeVar
T = TypeVar("T")
import time
import asyncio

import aiohttp
import bittensor as bt


class ZeusDendrite(bt.Dendrite):
    """
    A heavily improved Dendrite which has far better concurrency,
    And makes request timeouts independent of synapse cryptography,
    ensuring a far more fair subnet.

    During testing, it now takes less than a second for a request to reach a miner.
    All 125 requests are send in a fraction of a second,
    as opposed to a multi-second (3+) delay between the first and last miner.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # just do this once initially
        connector = aiohttp.TCPConnector(
            limit=200, 
            verify_ssl=False, 
            loop=asyncio.get_event_loop(),
            limit_per_host=125,
            )
        self._session = aiohttp.ClientSession(connector=connector)

    async def forward(
        self,
        axons: Union[list[Union[bt.AxonInfo, bt.Axon]], Union[bt.AxonInfo, bt.Axon]],
        synapse: bt.Synapse = bt.Synapse(),
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
    ) -> list[Union[AsyncGenerator[Any, Any], bt.Synapse, bt.StreamingSynapse]]:
        """
        Modified forward call to decompose request creation and sending,
        So that request timing is only applied to the later part.
        NOTE: does not support streaming for now as unused.
        """
        is_list = True
        # If a single axon is provided, wrap it in a list for uniform processing
        if not isinstance(axons, list):
            is_list = False
            axons = [axons]

        async def query_all_axons() -> Union[AsyncGenerator[Any, Any], bt.Synapse]:
            """
            Decompose calling into two steps. Can be run sync or async,
            as per the parent class.
            """

            async def sync_gather(*calls: List[Awaitable[T]]) -> List[T]:
                """Wrapper to handle any async inputs collection synchronously"""
                return [await call for call in calls]
            
            # whether to process calls in parallel or synchronously
            gather_func = asyncio.gather if run_async else sync_gather
            
            # First get all modified synapses and arguments to the post-calls
            calls = await gather_func(
                *[
                    self.prepare_call(
                        target_axon=target_axon,
                        synapse=synapse.model_copy(),
                        timeout=timeout, 
                    ) 
                    for target_axon in axons
                ]
            )

            # actually execute the calls, internal timing starts here
            return await gather_func(
                *[
                    self.call(post_args=post_args, synapse=synapse, deserialize=deserialize) 
                    for (synapse, post_args) in calls
                ]
            )

        # Get responses for all axons.
        responses = await query_all_axons()
        # Return the single response if only one axon was targeted, else return all responses
        return responses[0] if len(responses) == 1 and not is_list else responses  # type: ignore
    
    async def prepare_call(
        self,
        target_axon: Union[bt.AxonInfo, bt.Axon],
        timeout: float = 12.0,
        synapse: bt.Synapse = bt.Synapse(),
    ) -> Tuple[bt.Synapse, Dict[str, Any]]:
        """
        First half of default BitTensor dendrite.call function.
        Modifies the synapse in place, and returns all precomputed arguments for post request

        Returns:
        - Synapyse: modified in place to include hotkey signing
        - Post request arguments: A dict representing all precomputed HTTP post arguments
        """
        target_axon = (
            target_axon.info() if isinstance(target_axon, bt.Axon) else target_axon
        )

        # Build request endpoint from the synapse class
        request_name = synapse.__class__.__name__
        url = self._get_endpoint_url(target_axon, request_name=request_name)
        # Preprocess synapse for making a request
        synapse = self.preprocess_synapse_for_request(target_axon, synapse, timeout)

        # precompute the arguments to the call
        return (
            synapse, 
            {
            "url": url,
            "headers": synapse.to_headers(),
            "json": synapse.model_dump(),
            "timeout": aiohttp.ClientTimeout(total=timeout),
            }
        )

    async def call(
        self,
        post_args: Dict[str, Any],
        synapse: bt.Synapse,
        deserialize: bool = True,
    ) -> bt.Synapse:
        """
        Second half of default BitTensor dendrite.call function.
        Sends to actual request and handles its output, timing accordingly.

        Returns:
        - Synapse or deserialisation result
        """

        # Record start time
        start_time = time.time()

        try:
            # Log outgoing request
            self._log_outgoing_request(synapse)

            # Make the HTTP POST request
            async with (await self.session).post(**post_args) as response:
                # Extract the JSON response from the server
                json_response = await response.json()
                # Process the server response and fill synapse
                self.process_server_response(response, json_response, synapse)

            # Set process time and log the response
            synapse.dendrite.process_time = str(time.time() - start_time)  # type: ignore

        except Exception as e:
            synapse = self.process_error_message(synapse, synapse.__class__.__name__, e)

        finally:
            self._log_incoming_response(synapse)

            # Log synapse event history
            self.synapse_history.append(bt.Synapse.from_headers(synapse.to_headers()))

            # Return the updated synapse object after deserializing if requested
            return synapse.deserialize() if deserialize else synapse
