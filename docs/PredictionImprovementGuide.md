# Miner Prediction Quality Improvement Guide

## Executive Summary

Your current miner implementation simply calls OpenMeteo's API, which means you're **matching the baseline** (score ≈ 0). To earn rewards, you need to **beat OpenMeteo's RMSE**. This guide analyzes the scoring mechanism and provides actionable strategies to improve prediction quality.

## Understanding the Scoring System

### Reward Formula Breakdown

Your score is calculated as:
```
Score = 0.8 × Quality_Score + 0.2 × Efficiency_Score
```

**Quality Score (80% weight):**
- Based on **relative improvement** over OpenMeteo baseline
- Formula: `Improvement = (RMSE_baseline - RMSE_yours) / RMSE_baseline`
- Capped between -100% (worse) and +80% (better) than baseline
- Adjusted by difficulty: harder regions get more lenient scoring
- Gamma correction applied based on challenge difficulty

**Efficiency Score (20% weight):**
- Based on response time
- Threshold: < 0.4 seconds = "perfect"
- Capped at 2× median response time

### Key Insights

1. **You MUST beat OpenMeteo** to get positive rewards
2. **Even small improvements** (5-10% better RMSE) can yield significant rewards
3. **Response speed matters** but quality dominates (80/20 split)
4. **Difficulty weighting** means some regions are easier to score well on

## Current Implementation Analysis

### What Your Miner Does Now

```python
# Current flow:
1. Receives grid of (lat, lon) coordinates
2. Calls OpenMeteo API with those coordinates
3. Converts units to ERA5 format
4. Returns predictions
```

**Problem**: This is identical to what validators use as baseline, so you'll score ≈ 0.

### Variables You Need to Predict

| Variable | Weight | Description | Unit |
|----------|--------|------------|------|
| `2m_temperature` | 15% | Temperature 2m above ground | Kelvin |
| `total_precipitation` | 15% | Total precipitation | m/h |
| `100m_u_component_of_wind` | 20% | Eastward wind at 100m | m/s |
| `100m_v_component_of_wind` | 20% | Northward wind at 100m | m/s |
| `2m_dewpoint_temperature` | 20% | Dewpoint at 2m | Kelvin |
| `surface_pressure` | 10% | Surface pressure | Pascal |

**Note**: Wind components require special handling - OpenMeteo provides speed/direction, you need to convert to u/v components.

## Strategies to Improve Predictions

### 1. **Train ML Models on ERA5 Historical Data** ⭐⭐⭐⭐⭐

**Why**: OpenMeteo uses physics-based models. ML models trained on ERA5 can learn patterns that beat physics models.

**Approach**:
- Download ERA5 data from Google Cloud: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- Train models like:
  - **GraphCast** (Google DeepMind) - State-of-the-art for weather
  - **FourCastNet** - Fast, efficient weather model
  - **Pangu-Weather** (Huawei) - High accuracy
  - **Lightweight CNN/RNN** - Faster inference, good for short-term

**Implementation Tips**:
```python
# Load historical ERA5 data
import xarray as xr
dataset = xr.open_zarr("gs://gcp-public-data-arco-era5/...")

# Train on past data, predict future
# Use sliding window approach for temporal patterns
```

### 2. **Ensemble Multiple Models** ⭐⭐⭐⭐

**Why**: Combining predictions from multiple models often beats individual models.

**Approach**:
- Combine OpenMeteo + your trained model + simple baselines
- Use weighted average (weight by historical performance)
- Or use stacking with meta-learner

**Example**:
```python
prediction = 0.4 * openmeteo_pred + 0.5 * your_model_pred + 0.1 * persistence
```

### 3. **Spatial-Temporal Modeling** ⭐⭐⭐⭐

**Why**: Weather has strong spatial and temporal correlations.

**Approach**:
- Use **Graph Neural Networks** to model spatial relationships
- Use **LSTM/Transformer** for temporal patterns
- Consider **ConvLSTM** or **3D CNNs** for spatiotemporal modeling

**Key Insight**: Nearby locations and recent hours are highly correlated.

### 4. **Post-Processing Corrections** ⭐⭐⭐

**Why**: Simple corrections can improve predictions without retraining.

**Approaches**:
- **Bias correction**: Learn systematic errors and correct them
- **Kalman filtering**: Smooth predictions using uncertainty estimates
- **Anomaly detection**: Flag and correct unrealistic predictions

### 5. **Variable-Specific Strategies** ⭐⭐⭐

Different variables need different approaches:

**Temperature**:
- Strong diurnal cycle - model time-of-day patterns
- Altitude effects - use elevation data if available
- Land-sea contrast - different models for ocean vs land

**Precipitation**:
- Highly non-Gaussian (many zeros, few large values)
- Use specialized loss functions (e.g., focal loss)
- Consider probabilistic predictions

**Wind**:
- Vector field - predict u and v together
- Use physical constraints (divergence, vorticity)
- Wind patterns are smoother than precipitation

**Pressure**:
- More stable, easier to predict
- Strong spatial correlations
- Can use simpler models

### 6. **Leverage Additional Data Sources** ⭐⭐⭐

**Why**: More data = better predictions.

**Sources**:
- **Current weather observations** (if available)
- **Satellite imagery** (cloud cover, etc.)
- **Topography data** (elevation affects weather)
- **Ocean data** (SST for coastal regions)

### 7. **Optimize for Speed** ⭐⭐

**Why**: 20% of score is efficiency, and faster = more queries handled.

**Strategies**:
- Use lighter models (MobileNet-style architectures)
- Model quantization (FP16 or INT8)
- Batch processing
- Caching common predictions
- GPU acceleration

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. ✅ Add error handling to current implementation
2. ✅ Implement simple post-processing (bias correction)
3. ✅ Add caching for repeated queries
4. ✅ Optimize response time

### Phase 2: Baseline ML Model (2-4 weeks)
1. Download ERA5 training data
2. Train simple CNN/RNN model
3. Implement ensemble with OpenMeteo
4. Test on validation set

### Phase 3: Advanced Models (1-3 months)
1. Implement GraphCast or FourCastNet
2. Add spatial-temporal modeling
3. Variable-specific model architectures
4. Continuous retraining pipeline

## Code Structure Recommendations

### Suggested Miner Architecture

```python
class Miner(BaseMinerNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        
        # Multiple prediction sources
        self.openmeteo_client = openmeteo_requests.Client()
        self.ml_model = self.load_trained_model()
        self.ensemble_weights = self.load_ensemble_weights()
        
        # Caching for speed
        self.prediction_cache = {}
        
    async def forward(self, synapse):
        # 1. Check cache
        cache_key = self._get_cache_key(synapse)
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # 2. Get predictions from multiple sources
        openmeteo_pred = await self._get_openmeteo_prediction(synapse)
        ml_pred = await self._get_ml_prediction(synapse)
        
        # 3. Ensemble
        prediction = self._ensemble_predictions(
            openmeteo_pred, ml_pred, synapse.variable
        )
        
        # 4. Post-process
        prediction = self._post_process(prediction, synapse)
        
        # 5. Cache and return
        self.prediction_cache[cache_key] = prediction
        return prediction
```

## Critical Implementation Details

### 1. Shape Handling
Your output **MUST** match expected shape: `[time, lat, lon]`
- `time`: Number of requested hours
- `lat`: Number of latitude points in grid
- `lon`: Number of longitude points in grid

**Common mistakes**:
- Wrong order of dimensions
- Missing squeeze for single-variable outputs
- Incorrect time dimension

### 2. Unit Conversions
Each variable has specific units (see converter.py):
- Temperature: Kelvin (not Celsius!)
- Precipitation: m/h (not mm/h!)
- Wind: m/s components
- Pressure: Pascal (not hPa!)

### 3. Coordinate Handling
- Coordinates come as `[lat, lon, 2]` tensor
- You need to reshape to grid format
- Handle edge cases (poles, date line)

### 4. Error Handling
```python
try:
    prediction = your_model_predict(...)
except Exception as e:
    # Fallback to OpenMeteo
    bt.logging.warning(f"Model failed: {e}, using fallback")
    prediction = openmeteo_prediction
```

## Performance Benchmarks

### Target Metrics

| Metric | Baseline (OpenMeteo) | Good Miner | Excellent Miner |
|--------|---------------------|------------|-----------------|
| RMSE Improvement | 0% | +5-10% | +15-25% |
| Response Time | ~1-2s | <0.5s | <0.3s |
| Score | 0.0 | 0.3-0.5 | 0.6-0.8 |

### Measuring Your Performance

1. **Test locally** with sample data
2. **Deploy to testnet** (netuid 301) first
3. **Monitor WandB** logs for your performance
4. **Compare** your RMSE vs baseline RMSE

## Resources

### Datasets
- ERA5: `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- OpenMeteo API: https://open-meteo.com/en/docs
- Copernicus CDS: https://cds.climate.copernicus.eu/

### Model Implementations
- GraphCast: https://github.com/google-deepmind/graphcast
- FourCastNet: https://github.com/NVlabs/FourCastNet
- Pangu-Weather: https://github.com/198808xc/Pangu-Weather

### Papers
- GraphCast: "GraphCast: Learning skillful medium-range global weather forecasting"
- FourCastNet: "FourCastNet: A Global Data-driven High-resolution Weather Model"
- Pangu-Weather: "Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast"

## Next Steps

1. **Start with Phase 1** - Quick wins that don't require training
2. **Set up data pipeline** - Download and preprocess ERA5 data
3. **Train baseline model** - Simple CNN/RNN to beat OpenMeteo
4. **Iterate** - Continuously improve based on validation performance
5. **Deploy** - Test on testnet before mainnet

## Common Pitfalls to Avoid

1. ❌ **Forgetting unit conversions** - Will cause shape penalties
2. ❌ **Wrong tensor shapes** - Must match exactly
3. ❌ **Slow response times** - Optimize inference
4. ❌ **Overfitting to validation** - Use proper train/val/test splits
5. ❌ **Ignoring variable-specific needs** - Different variables need different approaches
6. ❌ **Not handling edge cases** - Poles, date line, small grids

## Conclusion

To improve prediction quality:
1. **Train ML models** on ERA5 historical data
2. **Ensemble** multiple prediction sources
3. **Optimize** for both accuracy and speed
4. **Test thoroughly** before deploying

The key is to **beat OpenMeteo's RMSE** - even small improvements (5-10%) can yield significant rewards due to the relative scoring system.

