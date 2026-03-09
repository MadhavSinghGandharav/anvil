use crate::core::DenseMatrix;

/// Standardize features by removing the mean and scaling to unit variance.
///
/// The transformation applied is:
///
/// ```text
/// z = (x - μ) / σ
/// ```
///
/// where:
///
/// - `μ` = feature mean
/// - `σ` = feature standard deviation
///
/// # Notes
///
/// - Statistics are computed during [`fit`].
/// - Variance smoothing is applied to avoid division by zero.
/// - The scaler must be fitted before calling [`transform`].
///
/// # Example
///
/// ```ignore
/// let mut scaler = StandardScaler::new();
///
/// scaler.fit(&x);
///
/// let x_scaled = scaler.transform(&x);
/// ```
pub struct StandardScaler {
    mean: Option<Vec<f64>>,
    std: Option<Vec<f64>>,
}

impl StandardScaler {

    /// Create a new `StandardScaler`.
    ///
    /// The scaler is initially unfitted.
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }

    /// Return the computed feature means.
    ///
    /// # Panics
    ///
    /// Panics if the scaler has not been fitted.
    pub fn mean(&self) -> &[f64] {
        self.mean
            .as_ref()
            .expect("StandardScaler not fitted")
    }

    /// Return the computed feature standard deviations.
    ///
    /// # Panics
    ///
    /// Panics if the scaler has not been fitted.
    pub fn std(&self) -> &[f64] {
        self.std
            .as_ref()
            .expect("StandardScaler not fitted")
    }

    /// Compute feature-wise mean and standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if the dataset is empty.
    pub fn fit(&mut self, features: &DenseMatrix) {

        let n_features = features.n_cols();
        let n_samples = features.n_rows();

        assert!(
            n_samples > 0,
            "Cannot fit scaler on empty dataset"
        );

        let mut sum = vec![0.0; n_features];
        let mut sum_sq = vec![0.0; n_features];

        for i in 0..n_samples {

            let row = features.row(i);

            for (j, &x) in row.iter().enumerate() {
                sum[j] += x;
                sum_sq[j] += x * x;
            }
        }

        for i in 0..n_features {

            let mean = sum[i] / n_samples as f64;

            sum[i] = mean;

            sum_sq[i] =
                (sum_sq[i] / n_samples as f64)
                - mean * mean;
        }

        // variance smoothing
        let mut max_var = f64::NEG_INFINITY;

        for &v in sum_sq.iter() {
            if v > max_var {
                max_var = v;
            }
        }

        let eps = (max_var * 1e-9).max(1e-9);

        for v in sum_sq.iter_mut() {
            *v = (*v + eps).sqrt();
        }

        self.mean = Some(sum);
        self.std = Some(sum_sq);
    }

    /// Transform features using the fitted statistics.
    ///
    /// Applies:
    ///
    /// ```text
    /// z = (x - mean) / std
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - the scaler has not been fitted
    /// - feature dimension mismatch
    pub fn transform(&self, features: &DenseMatrix) -> DenseMatrix {

        let mean = self.mean.as_ref().expect("StandardScaler not fitted");
        let std = self.std.as_ref().expect("StandardScaler not fitted");

        assert!(
            features.n_cols() == mean.len(),
            "Feature dimension mismatch"
        );

        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        let mut scaled = DenseMatrix::zeros(n_samples, n_features);

        for i in 0..n_samples {

            let row = features.row(i);

            for j in 0..n_features {

                *scaled.get_mut(i, j) =
                    (row[j] - mean[j]) / std[j];
            }
        }

        scaled
    }

    /// Fit the scaler and transform the data.
    ///
    /// Equivalent to calling [`fit`] followed by [`transform`].
    pub fn fit_transform(&mut self, features: &DenseMatrix) -> DenseMatrix {

        self.fit(features);

        self.transform(features)
    }
}
