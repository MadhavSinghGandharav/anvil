use ndarray::{Array2, ArrayView2};

/// Quantile transformer — maps each feature to a uniform `[0, 1]` distribution.
///
/// For each feature, `n_quantiles` reference points are computed during `fit`.
/// During `transform`, each value is mapped to its quantile rank via
/// linear interpolation between the two nearest reference points.
///
/// # Notes
///
/// - Robust to outliers — values outside the training range are clipped to `[0, 1]`
/// - Quantile levels are spaced as `(i + 0.5) / k` to avoid boundary artifacts
pub struct QuantileTransformer {

    /// Per-feature quantile reference values computed during `fit`
    q_values: Option<Vec<Vec<f64>>>,

    /// Number of quantile reference points
    n_quantiles: usize,
}

impl QuantileTransformer {

    /// Create transformer with the given number of quantile reference points
    pub fn new(n_quantiles: usize) -> Self {
        Self {
            q_values: None,
            n_quantiles,
        }
    }

    /// Returns builder
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Fits the transformer by computing per-feature quantile reference values
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - dataset contains zero samples
    pub fn fit(&mut self, features: ArrayView2<f64>) {

        assert!(
            features.nrows() > 0,
            "Cannot fit with zero samples"
        );

        let n_samples  = features.nrows();
        let n_features = features.ncols();

        // effective number of quantiles — cannot exceed n_samples
        let k = self.n_quantiles.min(n_samples);

        // quantile levels spaced to avoid 0 and 1 boundary artifacts
        let quantile_levels: Vec<f64> = (0..k)
            .map(|i| (i as f64 + 0.5) / k as f64)
            .collect();

        let mut all_q_values = Vec::with_capacity(n_features);
        let mut col_buffer   = Vec::with_capacity(n_samples);

        for col in 0..n_features {

            col_buffer.clear();

            // copy column — column() iterator handles strided access
            col_buffer.extend(features.column(col).iter().copied());

            // sort column for quantile computation
            col_buffer.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            all_q_values.push(find_q_values(&col_buffer, &quantile_levels));
        }

        self.q_values = Some(all_q_values);
    }

    /// Transforms features by mapping each value to its quantile rank
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - transformer not fitted
    /// - feature dimension mismatch
    pub fn transform(&self, features: ArrayView2<f64>) -> Array2<f64> {

        let q_values = self.q_values.as_ref().expect("Transformer not fitted");

        assert_eq!(
            features.ncols(),
            q_values.len(),
            "Feature dimension mismatch"
        );

        let n_samples  = features.nrows();
        let n_features = features.ncols();

        let mut output = Array2::<f64>::zeros((n_samples, n_features));

        for row in 0..n_samples {
            for col in 0..n_features {
                output[[row, col]] = transform_value(features[[row, col]], &q_values[col]);
            }
        }

        output
    }

    /// Fits the transformer and transforms the features in one step
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - dataset contains zero samples
    pub fn fit_transform(&mut self, features: ArrayView2<f64>) -> Array2<f64> {
        self.fit(features);
        self.transform(features)
    }
}

/// Builder for configuring [`QuantileTransformer`]
pub struct Builder {
    n_quantiles: usize,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - n_quantiles = 1000
    fn default() -> Self {
        Self {
            n_quantiles: 1000,
        }
    }
}

impl Builder {

    /// Set number of quantile reference points
    pub fn n_quantiles(mut self, value: usize) -> Self {
        self.n_quantiles = value;
        self
    }

    /// Build transformer
    pub fn build(self) -> QuantileTransformer {
        QuantileTransformer {
            q_values: None,
            n_quantiles: self.n_quantiles,
        }
    }
}

/// Computes quantile reference values from a sorted slice
fn find_q_values(sorted: &[f64], quantile_levels: &[f64]) -> Vec<f64> {

    let n = sorted.len();

    if n == 0 {
        return vec![];
    }

    if n == 1 {
        return vec![sorted[0]; quantile_levels.len()];
    }

    let mut out = Vec::with_capacity(quantile_levels.len());

    for &q in quantile_levels {

        let p = q.clamp(0.0, 1.0) * (n - 1) as f64;
        let i = p.floor() as usize;
        let f = p - i as f64;

        let val = if i + 1 < n {
            sorted[i] + (sorted[i + 1] - sorted[i]) * f
        } else {
            sorted[i]
        };

        out.push(val);
    }

    out
}

/// Maps a single value to its quantile rank via binary search and linear interpolation
fn transform_value(x: f64, qv: &[f64]) -> f64 {

    let k = qv.len();

    // clip values outside training range
    if x <= qv[0]     { return 0.0; }
    if x >= qv[k - 1] { return 1.0; }

    // binary search for the bracketing interval
    let idx = match qv.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
        Ok(i)  => i,
        Err(i) => i,
    };

    let i  = idx - 1;
    let v1 = qv[i];
    let v2 = qv[i + 1];
    let q1 = i as f64       / (k - 1) as f64;
    let q2 = (i + 1) as f64 / (k - 1) as f64;

    // linear interpolation between bracketing quantile levels
    q1 + (x - v1) / (v2 - v1) * (q2 - q1)
}
