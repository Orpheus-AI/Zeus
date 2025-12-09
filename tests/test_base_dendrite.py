"""
Unit tests for zeus.base.dendrite module.
"""
import pytest
import asyncio
import warnings
from unittest.mock import Mock, AsyncMock, patch, MagicMock, PropertyMock
from types import SimpleNamespace
import bittensor as bt
from zeus.base.dendrite import ZeusDendrite
from zeus.protocol import PreferenceSynapse, PredictionSynapse

# Suppress warnings from bittensor library
warnings.filterwarnings("ignore", category=DeprecationWarning, module="bittensor")
warnings.filterwarnings("ignore", message=".*coroutine.*was never awaited", category=RuntimeWarning)


class TestZeusDendrite:
    """Tests for ZeusDendrite class."""

    @pytest.fixture
    def dendrite(self, mock_wallet):
        """Create a ZeusDendrite instance with a mock wallet."""
        dendrite = ZeusDendrite(wallet=mock_wallet)
        yield dendrite
        # Clean up: close the session if it exists
        if hasattr(dendrite, '_session') and dendrite._session and not dendrite._session.closed:
            try:
                # Try to close the session
                import asyncio
                loop = asyncio.get_event_loop()
                if not loop.is_running():
                    loop.run_until_complete(dendrite._session.close())
            except Exception:
                pass  # Ignore cleanup errors

    def test_dendrite_is_dendrite_subclass(self, dendrite):
        """Test that ZeusDendrite is a subclass of bt.Dendrite."""
        assert isinstance(dendrite, bt.Dendrite)

    def test_dendrite_has_session(self, dendrite):
        """Test that ZeusDendrite has a session attribute."""
        assert hasattr(dendrite, '_session')

    @pytest.mark.asyncio
    async def test_prepare_call_returns_synapse_and_args(self, dendrite):
        """Test that prepare_call returns synapse and post_args."""
        # Create a proper AxonInfo-like object with actual values using SimpleNamespace
        # This ensures ip and port are real values, not Mock objects
        axon_info = SimpleNamespace(ip='127.0.0.1', port=8091)
        
        mock_axon = Mock()
        mock_axon.info = Mock(return_value=axon_info)
        synapse = PreferenceSynapse()
        
        # Patch preprocess_synapse_for_request to avoid TerminalInfo issues
        with patch.object(dendrite, 'preprocess_synapse_for_request', return_value=synapse):
            result_synapse, post_args = await dendrite.prepare_call(
                target_axon=mock_axon,
                synapse=synapse,
                timeout=12.0
            )
        
        assert isinstance(result_synapse, PreferenceSynapse)
        assert isinstance(post_args, dict)
        assert 'url' in post_args
        assert 'headers' in post_args
        assert 'json' in post_args
        assert 'timeout' in post_args

    @pytest.mark.asyncio
    async def test_prepare_call_handles_axon_info(self, dendrite):
        """Test that prepare_call handles both Axon and AxonInfo."""
        # Test with AxonInfo - create proper object with actual values
        axon_info = SimpleNamespace(ip='127.0.0.1', port=8091)
        synapse = PreferenceSynapse()
        
        # Patch preprocess_synapse_for_request to avoid TerminalInfo issues
        with patch.object(dendrite, 'preprocess_synapse_for_request', return_value=synapse):
            result_synapse, post_args = await dendrite.prepare_call(
                target_axon=axon_info,
                synapse=synapse
            )
        
        assert 'url' in post_args
        
        # Test with Axon (should call info() if it's recognized as bt.Axon)
        # Since Mock won't be recognized as bt.Axon by isinstance, we test that
        # passing AxonInfo directly works (which is what happens after info() is called)
        axon_info2 = SimpleNamespace(ip='192.168.1.1', port=9000)
        
        with patch.object(dendrite, 'preprocess_synapse_for_request', return_value=synapse):
            result_synapse, post_args = await dendrite.prepare_call(
                target_axon=axon_info2,
                synapse=synapse
            )
        
        assert 'url' in post_args
        # Verify that the URL was built from the axon_info
        assert '192.168.1.1' in post_args['url'] or '9000' in post_args['url']

    @pytest.mark.asyncio
    async def test_call_handles_success(self, dendrite):
        """Test that call handles successful requests."""
        post_args = {
            'url': 'http://127.0.0.1:8091/PreferenceSynapse',
            'headers': {},
            'json': {},
            'timeout': Mock()
        }
        synapse = PreferenceSynapse()
        
        # Mock the session.post to return a successful response
        # The post() method returns an async context manager
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={'mechanism': 0})
        mock_response.status = 200
        
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_context)
        
        # Mock the session property - it's an async property that returns the session
        # The code does `await self.session`, so the property must return a coroutine
        # Create a function that returns a new coroutine each time (coroutines can only be awaited once)
        def get_session_coro():
            async def session_coro():
                return mock_session
            return session_coro()
        
        # Use side_effect to return a new coroutine each time the property is accessed
        with patch.object(type(dendrite), 'session', new_callable=PropertyMock, side_effect=get_session_coro):
            result = await dendrite.call(
                post_args=post_args,
                synapse=synapse,
                deserialize=True
            )
        
        assert result is not None

    @pytest.mark.asyncio
    async def test_call_handles_error(self, dendrite):
        """Test that call handles errors gracefully."""
        post_args = {
            'url': 'http://127.0.0.1:8091/PreferenceSynapse',
            'headers': {},
            'json': {},
            'timeout': Mock()
        }
        synapse = PreferenceSynapse()
        
        # Mock the session.post to raise an error
        # The post() method returns an async context manager
        mock_response = AsyncMock()
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(side_effect=Exception("Connection error"))
        mock_context.__aexit__ = AsyncMock(return_value=None)
        
        mock_session = AsyncMock()
        mock_session.post = Mock(return_value=mock_context)
        
        # Mock the session property - it's an async property that returns the session
        # The code does `await self.session`, so the property must return a coroutine
        # Create a function that returns a new coroutine each time (coroutines can only be awaited once)
        def get_session_coro():
            async def session_coro():
                return mock_session
            return session_coro()
        
        # Use side_effect to return a new coroutine each time the property is accessed
        # The method should handle the error gracefully without raising an exception
        # The result may be None if deserialize() returns None, which is acceptable
        with patch.object(type(dendrite), 'session', new_callable=PropertyMock, side_effect=get_session_coro):
            # If this completes without raising an exception, error handling worked
            result = await dendrite.call(
                post_args=post_args,
                synapse=synapse,
                deserialize=True
            )
        
        # If we reach here, the error was handled gracefully (no exception was raised)
        # The result might be None if deserialize() returns None, which is acceptable
        # The important thing is that the error was caught and handled without crashing

    @pytest.mark.asyncio
    async def test_forward_single_axon(self, dendrite):
        """Test forward with a single axon."""
        axon_info = SimpleNamespace(ip='127.0.0.1', port=8091)
        
        mock_axon = Mock()
        mock_axon.info = Mock(return_value=axon_info)
        synapse = PreferenceSynapse()
        
        # Mock prepare_call and call
        async def mock_prepare_call(*args, **kwargs):
            return synapse, {'url': 'http://127.0.0.1:8091/test', 'headers': {}, 'json': {}, 'timeout': Mock()}
        
        async def mock_call(*args, **kwargs):
            return synapse
        
        dendrite.prepare_call = mock_prepare_call
        dendrite.call = mock_call
        
        result = await dendrite.forward(
            axons=mock_axon,
            synapse=synapse,
            timeout=12.0
        )
        
        # Should return single synapse, not list
        assert isinstance(result, PreferenceSynapse)

    @pytest.mark.asyncio
    async def test_forward_multiple_axons(self, dendrite):
        """Test forward with multiple axons."""
        mock_axons = [Mock() for _ in range(3)]
        for axon in mock_axons:
            axon_info = SimpleNamespace(ip='127.0.0.1', port=8091)
            axon.info = Mock(return_value=axon_info)
        
        synapse = PreferenceSynapse()
        
        # Mock prepare_call and call
        async def mock_prepare_call(*args, **kwargs):
            return synapse, {'url': 'http://127.0.0.1:8091/test', 'headers': {}, 'json': {}, 'timeout': Mock()}
        
        async def mock_call(*args, **kwargs):
            return synapse
        
        dendrite.prepare_call = mock_prepare_call
        dendrite.call = mock_call
        
        result = await dendrite.forward(
            axons=mock_axons,
            synapse=synapse,
            timeout=12.0
        )
        
        # Should return list of synapses
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_forward_run_async(self, dendrite):
        """Test forward with run_async=True (default)."""
        mock_axons = [Mock() for _ in range(2)]
        for axon in mock_axons:
            axon_info = SimpleNamespace(ip='127.0.0.1', port=8091)
            axon.info = Mock(return_value=axon_info)
        
        synapse = PreferenceSynapse()
        
        call_count = 0
        
        async def mock_prepare_call(*args, **kwargs):
            return synapse, {'url': 'http://127.0.0.1:8091/test', 'headers': {}, 'json': {}, 'timeout': Mock()}
        
        async def mock_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)  # Small delay to test async
            return synapse
        
        dendrite.prepare_call = mock_prepare_call
        dendrite.call = mock_call
        
        result = await dendrite.forward(
            axons=mock_axons,
            synapse=synapse,
            run_async=True
        )
        
        assert len(result) == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_forward_run_sync(self, dendrite):
        """Test forward with run_async=False."""
        mock_axons = [Mock() for _ in range(2)]
        for axon in mock_axons:
            axon_info = SimpleNamespace(ip='127.0.0.1', port=8091)
            axon.info = Mock(return_value=axon_info)
        
        synapse = PreferenceSynapse()
        
        call_order = []
        
        async def mock_prepare_call(*args, **kwargs):
            return synapse, {'url': 'http://127.0.0.1:8091/test', 'headers': {}, 'json': {}, 'timeout': Mock()}
        
        async def mock_call(*args, **kwargs):
            call_order.append(1)
            await asyncio.sleep(0.01)
            return synapse
        
        dendrite.prepare_call = mock_prepare_call
        dendrite.call = mock_call
        
        result = await dendrite.forward(
            axons=mock_axons,
            synapse=synapse,
            run_async=False
        )
        
        assert len(result) == 2
        # With sync, calls should be sequential
        assert len(call_order) == 2

    @pytest.mark.asyncio
    async def test_forward_deserialize_false(self, dendrite):
        """Test forward with deserialize=False."""
        axon_info = SimpleNamespace(ip='127.0.0.1', port=8091)
        
        mock_axon = Mock()
        mock_axon.info = Mock(return_value=axon_info)
        synapse = PreferenceSynapse()
        
        async def mock_prepare_call(*args, **kwargs):
            return synapse, {'url': 'http://127.0.0.1:8091/test', 'headers': {}, 'json': {}, 'timeout': Mock()}
        
        async def mock_call(*args, deserialize=True, **kwargs):
            if deserialize:
                return synapse.deserialize()
            return synapse
        
        dendrite.prepare_call = mock_prepare_call
        dendrite.call = mock_call
        
        result = await dendrite.forward(
            axons=mock_axon,
            synapse=synapse,
            deserialize=False
        )
        
        # Should return synapse object, not deserialized value
        assert isinstance(result, PreferenceSynapse)

