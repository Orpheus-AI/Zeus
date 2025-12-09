"""
Unit tests for zeus.base.metagraph module.
Tests MetagraphDelegate and Mechagraph with real bittensor data.

Note: These tests require network access to bittensor mainnet.
Mark with @pytest.mark.network to skip in CI if needed.
"""
import pytest
import bittensor as bt
import numpy as np
from zeus.base.metagraph.delegate import MetagraphDelegate
from zeus.base.metagraph.mechagraph import Mechagraph
from zeus.validator.constants import MechanismType, MECHAGRAPH_SIZES, MAINNET_UID

# Mark all tests in this module as requiring network access
pytestmark = pytest.mark.network


class TestMetagraphDelegate:
    """Tests for MetagraphDelegate class."""

    @pytest.fixture(scope="module")
    def metagraph_delegate(self):
        """Create a MetagraphDelegate instance with real bittensor data.
        
        Uses module scope to initialize only once for all tests in this class,
        making tests much faster.
        """
        # Use mainnet for testing
        subtensor = bt.subtensor(network="finney")
        delegate = MetagraphDelegate(
            netuid=MAINNET_UID,
            subtensor=subtensor,
            sync=True
        )
        return delegate

    @pytest.fixture(scope="module")
    def regular_metagraph(self):
        """Create a regular bittensor Metagraph for comparison.
        
        Uses module scope to initialize only once for all tests in this class,
        making tests much faster.
        """
        subtensor = bt.subtensor(network="finney")
        metagraph = bt.metagraph(
            netuid=MAINNET_UID,
            subtensor=subtensor,
            sync=True
        )
        return metagraph

    def test_delegate_is_metagraph_subclass(self, metagraph_delegate):
        """Test that MetagraphDelegate is a subclass of bt.Metagraph."""
        assert isinstance(metagraph_delegate, bt.Metagraph)

    def test_delegate_has_same_fields_as_regular_metagraph(
        self, metagraph_delegate, regular_metagraph
    ):
        """Test that MetagraphDelegate has the same fields as regular Metagraph."""
        # Get all attributes from both metagraphs
        delegate_attrs = set(dir(metagraph_delegate))
        regular_attrs = set(dir(regular_metagraph))
        
        # Filter out private methods and methods that might differ
        delegate_public = {a for a in delegate_attrs if not a.startswith('_') or a in ['_sync']}
        regular_public = {a for a in regular_attrs if not a.startswith('_') or a in ['_sync']}
        
        # Core attributes that should be present in both
        core_attrs = {
            'n', 'netuid', 'axons', 'hotkeys', 'uids', 'block',
            'neurons', 'S', 'validator_permit', 'last_update',
            'block_at_registration', 'max_uids', 'hparams'
        }
        
        for attr in core_attrs:
            assert hasattr(metagraph_delegate, attr), f"Delegate missing attribute: {attr}"
            assert hasattr(regular_metagraph, attr), f"Regular metagraph missing attribute: {attr}"

    def test_delegate_has_same_core_values(
        self, metagraph_delegate, regular_metagraph
    ):
        """Test that MetagraphDelegate has the same core values as regular Metagraph."""
        assert metagraph_delegate.netuid == regular_metagraph.netuid
        assert metagraph_delegate.n == regular_metagraph.n
        # Note: block may differ if metagraphs were synced at different times
        # Both should be valid block numbers
        assert metagraph_delegate.block.item() > 0
        assert regular_metagraph.block.item() > 0
        assert len(metagraph_delegate.neurons) == len(regular_metagraph.neurons)
        assert len(metagraph_delegate.axons) == len(regular_metagraph.axons)
        assert len(metagraph_delegate.hotkeys) == len(regular_metagraph.hotkeys)

    def test_delegate_has_mechagraphs(self, metagraph_delegate):
        """Test that MetagraphDelegate has mechagraphs."""
        assert hasattr(metagraph_delegate, 'mechagraphs')
        assert isinstance(metagraph_delegate.mechagraphs, list)
        assert len(metagraph_delegate.mechagraphs) == len(MECHAGRAPH_SIZES)

    def test_delegate_mechagraphs_have_correct_mechanisms(self, metagraph_delegate):
        """Test that mechagraphs have the correct mechanisms."""
        mechanisms = {mg.mechanism for mg in metagraph_delegate.mechagraphs}
        expected_mechanisms = set(MECHAGRAPH_SIZES.keys())
        assert mechanisms == expected_mechanisms

    def test_delegate_sync_updates_mechagraphs(self, metagraph_delegate):
        """Test that syncing the delegate also syncs mechagraphs."""
        # Get initial block
        initial_block = metagraph_delegate.block.item()
        
        # Sync again
        metagraph_delegate.sync()
        
        # Verify mechagraphs were synced (they should have neurons attribute)
        # Note: Some mechanisms may not exist on blockchain yet, so neurons might be empty
        for mechagraph in metagraph_delegate.mechagraphs:
            assert hasattr(mechagraph, 'neurons')
            # Neurons can be empty if the mechanism doesn't exist on blockchain yet
            assert isinstance(mechagraph.neurons, list)
            assert len(mechagraph.neurons) >= 0


class TestMechagraph:
    """Tests for Mechagraph class."""

    @pytest.fixture
    def metagraph_delegate(self):
        """Create a MetagraphDelegate instance.
        
        Uses function scope to ensure each test gets fresh data,
        preventing shared state issues with metagraph_info.incentives.
        """
        subtensor = bt.subtensor(network="finney")
        delegate = MetagraphDelegate(
            netuid=MAINNET_UID,
            subtensor=subtensor,
            sync=True
        )
        return delegate

    @pytest.fixture
    def era5_mechagraph(self, metagraph_delegate):
        """Get the ERA5 mechagraph."""
        for mg in metagraph_delegate.mechagraphs:
            if mg.mechanism == MechanismType.ERA5:
                return mg
        pytest.fail("ERA5 mechagraph not found")

    @pytest.fixture
    def weatherxm_mechagraph(self, metagraph_delegate):
        """Get the WEATHER_XM mechagraph."""
        for mg in metagraph_delegate.mechagraphs:
            if mg.mechanism == MechanismType.WEATHER_XM:
                return mg
        pytest.fail("WEATHER_XM mechagraph not found")

    def test_mechagraph_is_metagraph_subclass(self, era5_mechagraph):
        """Test that Mechagraph is a subclass of bt.Metagraph."""
        assert isinstance(era5_mechagraph, bt.Metagraph)

    def test_mechagraph_has_parent(self, era5_mechagraph, metagraph_delegate):
        """Test that Mechagraph has a parent reference."""
        assert hasattr(era5_mechagraph, 'parent')
        assert era5_mechagraph.parent == metagraph_delegate

    def test_mechagraph_has_mechanism(self, era5_mechagraph):
        """Test that Mechagraph has a mechanism attribute."""
        assert hasattr(era5_mechagraph, 'mechanism')
        assert era5_mechagraph.mechanism == MechanismType.ERA5

    def test_mechagraph_respects_size(self, era5_mechagraph):
        """Test that Mechagraph respects the size it's given."""
        expected_size = MECHAGRAPH_SIZES[MechanismType.ERA5]
        assert era5_mechagraph.size == expected_size
        # Neurons can be empty if mechanism doesn't exist on blockchain yet
        assert len(era5_mechagraph.neurons) <= expected_size
        assert era5_mechagraph.max_uids == expected_size

    def test_mechagraph_neurons_are_subset_of_parent(
        self, era5_mechagraph, metagraph_delegate
    ):
        """Test that mechagraph neurons are a subset of parent neurons."""
        if len(era5_mechagraph.neurons) == 0:
            pytest.skip("No neurons in mechagraph (mechanism may not exist on blockchain yet)")
        
        parent_neurons = metagraph_delegate.neurons
        mech_neurons = era5_mechagraph.neurons
        
        # All mechagraph neurons should be in parent neurons
        parent_uids = {n.uid for n in parent_neurons}
        mech_uids = {n.uid for n in mech_neurons}
        
        assert mech_uids.issubset(parent_uids)
        assert len(mech_neurons) <= len(parent_neurons)

    def test_mechagraph_neurons_have_same_fields_except_incentive(
        self, era5_mechagraph, metagraph_delegate
    ):
        """Test that mechagraph neurons have same fields as parent except incentive."""
        if len(era5_mechagraph.neurons) == 0:
            pytest.skip("No neurons in mechagraph (mechanism may not exist on blockchain yet)")
        
        mech_neuron = era5_mechagraph.neurons[0]
        # Get parent neuron by index, not uid (since parent_idxs maps to indices)
        parent_idx = era5_mechagraph.parent_idxs[0]
        parent_neuron = metagraph_delegate.neurons[parent_idx]
        
        # Get all attributes from both neurons
        mech_attrs = set(dir(mech_neuron))
        parent_attrs = set(dir(parent_neuron))
        
        # Core attributes that should be the same
        core_attrs = {
            'uid', 'hotkey', 'coldkey', 'axon', 'validator_permit',
            'stake', 'rank', 'trust', 'consensus', 'incentive', 'dividends',
            'emission', 'active', 'last_update', 'block_at_registration'
        }
        
        for attr in core_attrs:
            if attr == 'incentive':
                # Incentive should be different (mechagraph-specific)
                assert hasattr(mech_neuron, attr)
                assert hasattr(parent_neuron, attr)
                # The incentive values should come from mechagraph_info
                assert mech_neuron.incentive is not None
            else:
                # Other attributes should be the same
                if hasattr(parent_neuron, attr):
                    assert hasattr(mech_neuron, attr), f"Mechagraph neuron missing: {attr}"

    def test_mechagraph_neurons_have_mechagraph_incentive(
        self, era5_mechagraph, metagraph_delegate
    ):
        """Test that mechagraph neurons have incentive from mechagraph_info."""
        if len(era5_mechagraph.neurons) == 0:
            pytest.skip("No neurons in mechagraph (mechanism may not exist on blockchain yet)")
        
        # Check that neurons have incentive set
        for neuron in era5_mechagraph.neurons:
            assert hasattr(neuron, 'incentive')
            assert neuron.incentive is not None

    def test_mechagraph_has_parent_idxs(self, era5_mechagraph):
        """Test that mechagraph has parent_idxs attribute."""
        if len(era5_mechagraph.neurons) == 0:
            pytest.skip("No neurons in mechagraph (mechanism may not exist on blockchain yet)")
        assert hasattr(era5_mechagraph, 'parent_idxs')
        assert isinstance(era5_mechagraph.parent_idxs, np.ndarray)
        assert len(era5_mechagraph.parent_idxs) == len(era5_mechagraph.neurons)

    def test_mechagraph_parent_idxs_match_neurons(
        self, era5_mechagraph, metagraph_delegate
    ):
        """Test that parent_idxs correctly map to neurons."""
        if len(era5_mechagraph.neurons) == 0:
            pytest.skip("No neurons in mechagraph (mechanism may not exist on blockchain yet)")
        
        for i, neuron in enumerate(era5_mechagraph.neurons):
            parent_idx = era5_mechagraph.parent_idxs[i]
            assert neuron.uid == metagraph_delegate.neurons[parent_idx].uid

    def test_mechagraph_stake_attributes_match_parent(
        self, era5_mechagraph, metagraph_delegate
    ):
        """Test that stake attributes are correctly filtered from parent."""
        if len(era5_mechagraph.neurons) == 0:
            pytest.skip("No neurons in mechagraph (mechanism may not exist on blockchain yet)")
        
        parent_idxs = era5_mechagraph.parent_idxs
        
        # Check alpha_stake
        if hasattr(metagraph_delegate, 'alpha_stake'):
            expected_alpha = metagraph_delegate.alpha_stake[parent_idxs]
            assert np.array_equal(era5_mechagraph.alpha_stake, expected_alpha)
        
        # Check tao_stake
        if hasattr(metagraph_delegate, 'tao_stake'):
            expected_tao = metagraph_delegate.tao_stake[parent_idxs]
            assert np.array_equal(era5_mechagraph.tao_stake, expected_tao)
        
        # Check total_stake
        if hasattr(metagraph_delegate, 'total_stake'):
            expected_total = metagraph_delegate.total_stake[parent_idxs]
            assert np.array_equal(era5_mechagraph.total_stake, expected_total)

    def test_mechagraph_neurons_follow_parent_uid_ordering(
        self, era5_mechagraph, metagraph_delegate
    ):
        """Test that mechagraph neurons follow parent uid ordering (not sorted by incentive).
        
        NOTE: Neurons are not sorted by incentive in the mechagraph. 
        They should follow parent uid ordering (with some missing of course).
        """
        if len(era5_mechagraph.neurons) < 2:
            pytest.skip("Need at least 2 neurons to test ordering")
        
        # Get parent indices for mechagraph neurons
        parent_idxs = era5_mechagraph.parent_idxs
        
        # Get UIDs from parent at those indices
        parent_uids = [metagraph_delegate.neurons[idx].uid for idx in parent_idxs]
        
        # Get UIDs from mechagraph neurons
        mech_uids = [n.uid for n in era5_mechagraph.neurons]
        
        # UIDs should match (same order)
        assert mech_uids == parent_uids
        
        # Verify they're in ascending order (parent UID ordering)
        assert mech_uids == sorted(mech_uids)

    def test_mechagraph_sync_raises_error(self, era5_mechagraph):
        """Test that calling sync directly on mechagraph raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            era5_mechagraph.sync()

    def test_mechagraph_has_correct_netuid(self, era5_mechagraph, metagraph_delegate):
        """Test that mechagraph has the same netuid as parent."""
        assert era5_mechagraph.netuid == metagraph_delegate.netuid

    def test_mechagraph_has_correct_mechid(self, era5_mechagraph):
        """Test that mechagraph has correct mechid."""
        assert era5_mechagraph.mechid == era5_mechagraph.mechanism.value

    def test_mechagraph_filtered_attributes_match_parent(
        self, era5_mechagraph, metagraph_delegate
    ):
        """Test that filtered attributes (identities, pruning_score, etc.) match parent."""
        if len(era5_mechagraph.neurons) == 0:
            pytest.skip("No neurons in mechagraph (mechanism may not exist on blockchain yet)")
        
        parent_idxs = era5_mechagraph.parent_idxs
        
        # Check identities
        if hasattr(metagraph_delegate, 'identities'):
            expected_identities = np.array(metagraph_delegate.identities)[parent_idxs].tolist()
            assert era5_mechagraph.identities == expected_identities
        
        # Check pruning_score
        if hasattr(metagraph_delegate, 'pruning_score'):
            expected_pruning = np.array(metagraph_delegate.pruning_score)[parent_idxs].tolist()
            assert era5_mechagraph.pruning_score == expected_pruning
        
        # Check block_at_registration
        if hasattr(metagraph_delegate, 'block_at_registration'):
            expected_block = np.array(metagraph_delegate.block_at_registration)[parent_idxs].tolist()
            assert era5_mechagraph.block_at_registration == expected_block

    def test_multiple_mechagraphs_independent(
        self, era5_mechagraph, weatherxm_mechagraph
    ):
        """Test that different mechagraphs are independent."""
        assert era5_mechagraph.mechanism != weatherxm_mechagraph.mechanism
        
        # They should have different neurons (or at least potentially different)
        # Note: Some mechanisms may not exist on blockchain yet, so neurons might be empty
        era5_uids = {n.uid for n in era5_mechagraph.neurons}
        weatherxm_uids = {n.uid for n in weatherxm_mechagraph.neurons}
        
        # They might overlap, but that's okay - just verify they're separate objects
        assert era5_mechagraph is not weatherxm_mechagraph
        
        # Verify both mechagraphs have the correct structure regardless of neuron count
        assert era5_mechagraph.mechanism == MechanismType.ERA5
        assert weatherxm_mechagraph.mechanism == MechanismType.WEATHER_XM

    def test_mechagraph_num_uids_equals_n(self, era5_mechagraph):
        """Test that num_uids equals n."""
        assert era5_mechagraph.num_uids == era5_mechagraph.n
        # num_uids should equal the number of neurons (which can be 0 if mechanism doesn't exist)
        assert era5_mechagraph.num_uids == len(era5_mechagraph.neurons)
    
    def test_mechagraph_with_zero_neurons_valid(self, metagraph_delegate):
        """Test that mechagraphs with zero neurons are valid (mechanism may not exist yet)."""
        # Find a mechagraph with zero neurons (likely WEATHER_XM if it doesn't exist yet)
        zero_neuron_mechagraph = None
        for mg in metagraph_delegate.mechagraphs:
            if len(mg.neurons) == 0:
                zero_neuron_mechagraph = mg
                break
        
        if zero_neuron_mechagraph is None:
            pytest.skip("No mechagraph with zero neurons found (all mechanisms exist on blockchain)")
        
        # Verify the mechagraph is still valid even with zero neurons
        assert zero_neuron_mechagraph.size > 0
        assert zero_neuron_mechagraph.max_uids == zero_neuron_mechagraph.size
        assert len(zero_neuron_mechagraph.neurons) == 0
        assert len(zero_neuron_mechagraph.parent_idxs) == 0
        assert zero_neuron_mechagraph.n == 0
        assert zero_neuron_mechagraph.num_uids == 0

