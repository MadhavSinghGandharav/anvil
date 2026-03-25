use ndarray::{Array1, Array2, ArrayView2, Zip};
use std::cmp::Ordering::Greater;

/// Robust scaler — median centering and IQR scaling.
///
/// Transforms each feature by subtracting its median and dividing
/// by its interquartile range (IQR):
///
/// ```text
/// x' = (x - median) / IQR
/// ```
///
/// where `IQR = Q75 - Q25`.
///
/// # Notes
///
/// - Robust to outliers compared to `StandardScaler`
/// - If a feature has zero IQR (constant), its IQR is set to `1e-9`
///   to avoid division by zero — the feature is left unchanged
pub struct RobustScaler {

    /// Per-feature median computed during `fit`
    median: Option<Array1<f64>>,

    /// Per-feature interquartile range (Q75 - Q25) computed during `fit`
    iqr: Option<Array1<f64>>,
}

/// Computes a quantile using `select_nth_unstable` — O(n) average
///
/// Uses linear interpolation between adjacent values.
#[inline]
fn quantile(col: &mut [f64], q: f64) -> f64 {

    let n = col.len();
    let p = q * (n - 1) as f64;
    let i = p.floor() as usize;
    let f = p - i as f64;

    let (_, val, right) = col.select_nth_unstable_by(i, |a, b| {
        a.partial_cmp(b).unwrap_or(Greater)
    });

    if f == 0.0 || i >= n - 1 {
        return *val;
    }

    // find minimum of right partition — next value after i
    let next = right.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));

    // linear interpolation
    *val + (next - *val) * f
}

impl RobustScaler {

    /// Create scaler with default configuration
    pub fn new() -> Self {
        Self {
            median: None,
            iqr:    None,
        }
    }

    /// Fits the scaler by computing per-feature median and IQR
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

        let n_features = features.ncols();
        let n_samples  = features.nrows();

        // accumulate column buffers for quantile computation
        let mut col_buffer: Vec<Vec<f64>> = (0..n_features)
            .map(|_| Vec::with_capacity(n_samples))
            .collect();

        for row in features.outer_iter() {
            for (j, &v) in row.iter().enumerate() {
                col_buffer[j].push(v);
            }
        }

        let mut median = Array1::<f64>::zeros(n_features);
        let mut iqr    = Array1::<f64>::zeros(n_features);

        // compute per-feature quantiles — O(n) per quantile via select_nth_unstable
        for j in 0..n_features {
            let col = &mut col_buffer[j];
            let q25 = quantile(col, 0.25);
            let q50 = quantile(col, 0.50);
            let q75 = quantile(col, 0.75);
            median[j] = q50;
            iqr[j]    = (q75 - q25).max(1e-9);
        }

        self.median = Some(median);
        self.iqr    = Some(iqr);
    }

    /// Transforms features by applying median centering and IQR scaling
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - scaler not fitted
    /// - feature dimension mismatch
    pub fn transform(&self, features: ArrayView2<f64>) -> Array2<f64> {

        let median = self.median.as_ref().expect("Scaler not fitted");
        let iqr    = self.iqr.as_ref().expect("Scaler not fitted");

        assert_eq!(
            features.ncols(),
            median.len(),
            "Feature dimension mismatch"
        );

        let mut data = features.to_owned();

        for mut row in data.outer_iter_mut() {
            Zip::from(&mut row)
                .and(median)
                .and(iqr)
                .for_each(|x, &m, &s| {
                    *x = (*x - m) / s;
                });
        }

        data
    }

    /// Fits the scaler and transforms the features in one step
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
