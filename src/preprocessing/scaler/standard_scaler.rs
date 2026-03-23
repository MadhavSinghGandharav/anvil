use ndarray::{Array1, Array2, ArrayView2, Zip};

/// Standard scaler — zero mean, unit variance normalization.
///
/// Transforms each feature by subtracting its mean and dividing
/// by its standard deviation:
///
/// ```text
/// x' = (x - μ) / σ
/// ```
///
/// # Notes
///
/// - If a feature has zero variance (constant), its standard deviation
///   is set to `1.0` to avoid division by zero — the feature is left unchanged
pub struct StandardScaler {

    /// Per-feature mean computed during `fit`
    mean: Option<Array1<f64>>,

    /// Per-feature standard deviation computed during `fit`
    std: Option<Array1<f64>>,
}

impl StandardScaler {

    /// Create scaler with default configuration
    pub fn new() -> Self {
        Self {
            mean: None,
            std:  None,
        }
    }

    /// Fits the scaler by computing per-feature mean and standard deviation
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
        let n_samples  = features.nrows() as f64;

        let mut sum    = Array1::<f64>::zeros(n_features);
        let mut sum_sq = Array1::<f64>::zeros(n_features);

        // accumulate per-feature sums and squared sums
        for row in features.outer_iter() {
            Zip::from(&mut sum)
                .and(&mut sum_sq)
                .and(&row)
                .for_each(|s, sq, &v| {
                    *s  += v;
                    *sq += v * v;
                });
        }

        // compute mean
        sum /= n_samples;

        // compute std — constant features get std = 1.0 to avoid division by zero
        Zip::from(&mut sum_sq)
            .and(&sum)
            .for_each(|sq, &m| {
                let variance = *sq / n_samples - m * m;
                *sq = if variance < 1e-10 { 1.0 } else { variance.sqrt() };
            });

        self.mean = Some(sum);
        self.std  = Some(sum_sq);
    }

    /// Transforms features by applying zero mean and unit variance scaling
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - scaler not fitted
    /// - feature dimension mismatch
    pub fn transform(&self, features: ArrayView2<f64>) -> Array2<f64> {

        let mean = self.mean.as_ref().expect("Scaler not fitted");
        let std  = self.std.as_ref().expect("Scaler not fitted");

        assert_eq!(
            features.ncols(),
            mean.len(),
            "Feature dimension mismatch"
        );

        let mut data = features.to_owned();

        for mut row in data.outer_iter_mut() {
            Zip::from(&mut row)
                .and(mean)
                .and(std)
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
