# Deep Analysis: Improving Miner Prediction Quality

## Critical Findings from Codebase Analysis

### 1. Current State: You're Matching the Baseline

**Your current miner code** (lines 77-110 in `neurons/miner.py`):
- Simply calls OpenMeteo API
- Returns predictions in same format as baseline
- **Result**: Score ≈ 0.0 (no improvement over baseline)

**The validator's baseline** (`zeus/data/loaders/openmeteo.py`):
- Uses identical OpenMeteo API calls
- Same conversion logic
- **This is what you need to beat**

### 2. Scoring Mechanism Deep Dive

#### Reward Calculation Flow

```python
# From zeus/validator/reward.py

# Step 1: Calculate RMSE
miner_rmse = sqrt(mean((prediction - ground_truth)^2))
baseline_rmse = sqrt(mean((openmeteo_pred - ground_truth)^2))

# Step 2: Calculate relative improvement
improvement = (baseline_rmse - miner_rmse) / baseline_rmse
# Range: -1.0 (100% worse) to +0.8 (80% better)

# Step 3: Apply difficulty weighting
gamma = 1 / (difficulty_factor + challenge_age)
quality_score = improvement^gamma  # Gamma correction

# Step 4: Combine with efficiency
final_score = 0.8 * quality_score + 0.2 * efficiency_score
```

#### Key Insights:

1. **You MUST beat OpenMeteo** to get positive rewards
   - If `miner_rmse >= baseline_rmse`: score ≤ 0
   - Need `miner_rmse < baseline_rmse` to earn rewards

2. **Small improvements matter**
   - 5% better RMSE → ~0.05 improvement → ~0.04 quality score
   - 10% better RMSE → ~0.10 improvement → ~0.08 quality score
   - 20% better RMSE → ~0.20 improvement → ~0.16 quality score

3. **Difficulty matters**
   - Easy regions (low variance): gamma < 1 → rewards compressed
   - Hard regions (high variance): gamma > 1 → rewards expanded
   - Future predictions: harder → more lenient scoring

4. **Speed penalty is small but real**
   - < 0.4s: perfect efficiency score
   - > 0.4s: efficiency score decreases
   - Default timeout: 10 seconds

### 3. Critical Code Issues Found

#### Issue 1: No Error Handling ⚠️

**Current code** (lines 88-90):
```python
responses = self.openmeteo_api.weather_api(
    "https://api.open-meteo.com/v1/forecast", params=params, method="POST"
)
```

**Problem**: If API fails, miner crashes → penalty (score = 0)

**Fix**:
```python
try:
    responses = self.openmeteo_api.weather_api(...)
except Exception as e:
    bt.logging.error(f"OpenMeteo API failed: {e}")
    # Return fallback predictions (e.g., persistence or climatology)
    return self._get_fallback_prediction(synapse)
```

#### Issue 2: No Shape Validation ⚠️

**Current code** doesn't validate output shape before returning.

**Problem**: Wrong shape → penalty (score = 0)

**Fix**:
```python
# After line 109
expected_shape = (synapse.requested_hours, coordinates.shape[0], coordinates.shape[1])
if output.shape != expected_shape:
    bt.logging.error(f"Shape mismatch! Expected {expected_shape}, got {output.shape}")
    # Fix or return error
    output = self._fix_shape(output, expected_shape)
```

#### Issue 3: No NaN/Inf Checking ⚠️

**Problem**: NaN or Inf values → penalty (score = 0)

**Fix**:
```python
# After line 109
if not torch.isfinite(output).all():
    bt.logging.warning("Output contains NaN/Inf, replacing with fallback")
    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
    # Or use better fallback strategy
```

#### Issue 4: Inefficient API Calls ⚠️

**Problem**: Every request calls external API → slow → lower efficiency score

**Fix**: Add caching
```python
from functools import lru_cache
import hashlib

def _get_cache_key(self, synapse):
    key_data = f"{synapse.variable}_{synapse.start_time}_{synapse.end_time}_{synapse.locations}"
    return hashlib.md5(key_data.encode()).hexdigest()

# In forward():
cache_key = self._get_cache_key(synapse)
if cache_key in self.prediction_cache:
    cached_pred, cache_time = self.prediction_cache[cache_key]
    if time.time() - cache_time < 3600:  # 1 hour cache
        return cached_pred
```

### 4. Why OpenMeteo Can Be Beaten

#### OpenMeteo's Limitations:

1. **Generic model**: Not optimized for specific regions
2. **No historical learning**: Doesn't learn from ERA5 patterns
3. **API latency**: Network calls add delay
4. **Limited variables**: May not use all available ERA5 data

#### Your Advantages:

1. **Can train on ERA5**: Access to 80+ years of historical data
2. **Region-specific models**: Train models for specific geographic areas
3. **Ensemble methods**: Combine multiple prediction sources
4. **Faster inference**: Local models vs API calls
5. **Variable-specific optimization**: Different models for different variables

### 5. Specific Improvement Strategies

#### Strategy 1: Simple Post-Processing (Quick Win) ⭐⭐⭐

**Idea**: Apply corrections to OpenMeteo predictions based on learned biases.

**Implementation**:
```python
class BiasCorrector:
    def __init__(self):
        # Load learned bias corrections per variable/region
        self.bias_map = self._load_bias_corrections()
    
    def correct(self, prediction, variable, region):
        bias = self.bias_map.get((variable, region), 0.0)
        return prediction + bias

# In forward():
output = converter.om_to_era5(output)
output = self.bias_corrector.correct(output, synapse.variable, self._get_region(coordinates))
```

**Expected improvement**: 2-5% RMSE reduction

#### Strategy 2: Ensemble with Persistence ⭐⭐⭐

**Idea**: Combine OpenMeteo with simple persistence (last known value).

**Implementation**:
```python
async def forward(self, synapse):
    # Get OpenMeteo prediction
    openmeteo_pred = await self._get_openmeteo_prediction(synapse)
    
    # Get persistence prediction (if historical data available)
    persistence_pred = await self._get_persistence_prediction(synapse)
    
    # Weighted ensemble
    if persistence_pred is not None:
        # Simple weighted average
        prediction = 0.7 * openmeteo_pred + 0.3 * persistence_pred
    else:
        prediction = openmeteo_pred
    
    return prediction
```

**Expected improvement**: 1-3% RMSE reduction

#### Strategy 3: Variable-Specific Models ⭐⭐⭐⭐

**Idea**: Different models for different variables.

**Why**: 
- Temperature: Strong diurnal patterns → LSTM/RNN
- Precipitation: Highly non-Gaussian → Specialized loss functions
- Wind: Vector field → Predict u/v together

**Implementation**:
```python
class VariableSpecificPredictor:
    def __init__(self):
        self.temp_model = self._load_temperature_model()
        self.precip_model = self._load_precipitation_model()
        self.wind_model = self._load_wind_model()
    
    def predict(self, synapse):
        if synapse.variable == "2m_temperature":
            return self.temp_model.predict(synapse)
        elif synapse.variable == "total_precipitation":
            return self.precip_model.predict(synapse)
        # ... etc
```

**Expected improvement**: 5-15% RMSE reduction

#### Strategy 4: Train on ERA5 Historical Data ⭐⭐⭐⭐⭐

**Idea**: Train ML models on ERA5 data to learn patterns OpenMeteo misses.

**Steps**:
1. Download ERA5 data: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
2. Preprocess: Normalize, create train/val/test splits
3. Train model: CNN, RNN, Transformer, or Graph Neural Network
4. Deploy: Load model in miner, use for predictions

**Model Options**:
- **Lightweight CNN**: Fast inference, good for short-term
- **LSTM/GRU**: Good for temporal patterns
- **Transformer**: State-of-the-art for sequences
- **GraphCast**: Best accuracy, slower inference

**Expected improvement**: 10-30% RMSE reduction

#### Strategy 5: Spatial-Temporal Modeling ⭐⭐⭐⭐

**Idea**: Model spatial correlations between nearby locations.

**Why**: Weather is spatially correlated - nearby locations have similar weather.

**Implementation**:
```python
class SpatialTemporalModel:
    def predict(self, coordinates, time_range):
        # Use Graph Neural Network or ConvLSTM
        # to model spatial-temporal relationships
        pass
```

**Expected improvement**: 5-20% RMSE reduction

### 6. Implementation Priority

#### Phase 1: Immediate Fixes (This Week)
1. ✅ Add error handling
2. ✅ Add shape validation
3. ✅ Add NaN/Inf checking
4. ✅ Add basic caching

**Expected impact**: Prevents penalties, improves reliability

#### Phase 2: Quick Wins (1-2 Weeks)
1. ✅ Implement bias correction
2. ✅ Add persistence ensemble
3. ✅ Optimize response time

**Expected impact**: 2-5% RMSE improvement

#### Phase 3: ML Models (1-3 Months)
1. ✅ Set up ERA5 data pipeline
2. ✅ Train baseline model (CNN/RNN)
3. ✅ Deploy and test
4. ✅ Iterate and improve

**Expected impact**: 10-30% RMSE improvement

### 7. Code Structure Recommendations

#### Improved Miner Architecture

```python
class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Prediction sources
        self.openmeteo_client = openmeteo_requests.Client()
        self.ml_model = self._load_ml_model()  # Optional
        self.bias_corrector = BiasCorrector()
        
        # Caching
        self.prediction_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'errors': 0
        }
    
    async def forward(self, synapse):
        self.stats['total_requests'] += 1
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(synapse)
            if cache_key in self.prediction_cache:
                cached_pred, cache_time = self.prediction_cache[cache_key]
                if time.time() - cache_time < self.cache_ttl:
                    self.stats['cache_hits'] += 1
                    synapse.predictions = cached_pred.tolist()
                    return synapse
            
            # Get predictions
            predictions = await self._get_predictions(synapse)
            
            # Validate and fix
            predictions = self._validate_and_fix(predictions, synapse)
            
            # Cache
            self.prediction_cache[cache_key] = (predictions, time.time())
            
            # Set response
            synapse.predictions = predictions.tolist()
            synapse.version = zeus_version
            
            return synapse
            
        except Exception as e:
            self.stats['errors'] += 1
            bt.logging.error(f"Prediction failed: {e}")
            # Return fallback
            return self._get_fallback_response(synapse)
    
    async def _get_predictions(self, synapse):
        """Get predictions from multiple sources and ensemble"""
        # Get OpenMeteo prediction
        openmeteo_pred = await self._get_openmeteo_prediction(synapse)
        
        # Get ML model prediction (if available)
        ml_pred = None
        if self.ml_model is not None:
            try:
                ml_pred = await self._get_ml_prediction(synapse)
            except Exception as e:
                bt.logging.warning(f"ML model failed: {e}")
        
        # Ensemble
        if ml_pred is not None:
            prediction = 0.6 * openmeteo_pred + 0.4 * ml_pred
        else:
            prediction = openmeteo_pred
        
        # Apply bias correction
        prediction = self.bias_corrector.correct(
            prediction, 
            synapse.variable,
            self._get_region(synapse.locations)
        )
        
        return prediction
    
    def _validate_and_fix(self, prediction, synapse):
        """Validate shape and values, fix if needed"""
        coordinates = torch.Tensor(synapse.locations)
        expected_shape = (synapse.requested_hours, coordinates.shape[0], coordinates.shape[1])
        
        # Fix shape
        if prediction.shape != expected_shape:
            prediction = self._fix_shape(prediction, expected_shape)
        
        # Fix NaN/Inf
        if not torch.isfinite(prediction).all():
            prediction = torch.nan_to_num(
                prediction, 
                nan=0.0, 
                posinf=prediction[torch.isfinite(prediction)].max() if torch.isfinite(prediction).any() else 1e6,
                neginf=prediction[torch.isfinite(prediction)].min() if torch.isfinite(prediction).any() else -1e6
            )
        
        return prediction
```

### 8. Testing Your Improvements

#### Local Testing

```python
# Create test synapse
synapse = TimePredictionSynapse(
    locations=[[40.0, -74.0], [41.0, -75.0]],  # NYC area
    start_time=time.time(),
    end_time=time.time() + 3600 * 24,  # 24 hours
    requested_hours=24,
    variable="2m_temperature"
)

# Test prediction
miner = Miner()
result = await miner.forward(synapse)

# Validate
assert result.predictions is not None
assert len(result.predictions) == 24  # 24 hours
assert len(result.predictions[0]) == 2  # 2 locations
```

#### Testnet Deployment

1. Deploy to testnet (netuid 301)
2. Monitor WandB logs
3. Compare your RMSE vs baseline RMSE
4. Iterate based on results

### 9. Key Metrics to Track

| Metric | Target | How to Measure |
|--------|--------|----------------|
| RMSE Improvement | > 5% | Compare your RMSE vs baseline RMSE |
| Response Time | < 0.4s | Track `dendrite.process_time` |
| Cache Hit Rate | > 50% | Track cache hits vs misses |
| Error Rate | < 1% | Track exceptions |
| Shape Penalties | 0 | Monitor validator logs |

### 10. Common Mistakes to Avoid

1. ❌ **Wrong units**: Temperature in Celsius instead of Kelvin
2. ❌ **Wrong shape**: Missing time dimension or wrong order
3. ❌ **NaN values**: Not checking for NaN/Inf
4. ❌ **Slow responses**: Not optimizing inference
5. ❌ **No error handling**: Crashes on API failures
6. ❌ **Overfitting**: Training on validation data
7. ❌ **Ignoring variable differences**: Same model for all variables

## Conclusion

To improve prediction quality:

1. **Fix immediate issues**: Error handling, shape validation, NaN checking
2. **Add quick wins**: Bias correction, caching, persistence ensemble
3. **Train ML models**: Use ERA5 historical data to beat OpenMeteo
4. **Optimize**: Speed matters (20% of score)
5. **Test thoroughly**: Use testnet before mainnet

**The key insight**: Even small improvements (5-10% better RMSE) can yield significant rewards due to the relative scoring system. Start with quick wins, then invest in ML models for larger gains.

