use crate::core::DenseMatrix;

/// Scale features to the range `[0, 1]`.
///
/// The transformation applied is:
///
/// ```text
/// x_scaled = (x - min) / (max - min)
/// ```
///
/// where:
///
/// - `min` = minimum value of the feature
/// - `max` = maximum value of the feature
///
/// # Notes
///
/// - Statistics are computed during [`fit`].
/// - If a feature has constant value (`max == min`),
///   the scaled output will be `0`.
/// - The scaler must be fitted before calling [`transform`].
///
/// # Example
///
/// ```ignore
/// let mut scaler = MinMaxScaler::new();
///
/// scaler.fit(&x);
///
/// let x_scaled = scaler.transform(&x);
/// ```
pub struct MinMaxScaler {
    min: Option<Vec<f64>>,
    max: Option<Vec<f64>>,
}

impl MinMaxScaler {

    /// Create a new `MinMaxScaler`.
    ///
    /// The scaler is initially unfitted.
    pub fn new() -> Self {
        Self {
            min: None,
            max: None,
        }
    }

    /// Return the computed feature minimum values.
    ///
    /// # Panics
    ///
    /// Panics if the scaler has not been fitted.
    pub fn min(&self) -> &[f64] {
        self.min
            .as_ref()
            .expect("MinMaxScaler not fitted")
    }

    /// Return the computed feature maximum values.
    ///
    /// # Panics
    ///
    /// Panics if the scaler has not been fitted.
    pub fn max(&self) -> &[f64] {
        self.max
            .as_ref()
            .expect("MinMaxScaler not fitted")
    }

    /// Compute feature-wise minimum and maximum values.
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

        let mut min = vec![f64::INFINITY; n_features];
        let mut max = vec![f64::NEG_INFINITY; n_features];

        for i in 0..n_samples {

            let row = features.row(i);

            for j in 0..n_features {

                if row[j] < min[j] {
                    min[j] = row[j];
                }

                if row[j] > max[j] {
                    max[j] = row[j];
                }
            }
        }

        self.min = Some(min);
        self.max = Some(max);
    }

    /// Transform features using the fitted statistics.
    ///
    /// Applies:
    ///
    /// ```text
    /// x_scaled = (x - min) / (max - min)
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - the scaler has not been fitted
    /// - feature dimension mismatch
    pub fn transform(&self, features: &DenseMatrix) -> DenseMatrix {

        let min = self.min.as_ref().expect("MinMaxScaler not fitted");
        let max = self.max.as_ref().expect("MinMaxScaler not fitted");

        assert!(
            features.n_cols() == min.len(),
            "Feature dimension mismatch"
        );

        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        let mut scaled = DenseMatrix::zeros(n_samples, n_features);

        for i in 0..n_samples {

            let row = features.row(i);

            for j in 0..n_features {

                let range = max[j] - min[j];

                *scaled.get_mut(i, j) =
                    if range == 0.0 {
                        0.0
                    } else {
                        (row[j] - min[j]) / range
                    };
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
