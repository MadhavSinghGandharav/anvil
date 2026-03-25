use ndarray::{Array1, Array2, ArrayView2, Zip};

/// Min-Max scaler — scales each feature to the `[0, 1]` range.
///
/// Transforms each feature using its observed minimum and maximum:
///
/// ```text
/// x' = (x - min) / (max - min)
/// ```
///
/// # Notes
///
/// - If a feature has zero range (constant), its range is set to `1.0`
///   to avoid division by zero — the feature is left unchanged
pub struct MinMaxScaler {

    /// Per-feature minimum computed during `fit`
    min: Option<Array1<f64>>,

    /// Per-feature range (max - min) computed during `fit`
    range: Option<Array1<f64>>,
}

impl MinMaxScaler {

    /// Create scaler with default configuration
    pub fn new() -> Self {
        Self {
            min:   None,
            range: None,
        }
    }

    /// Fits the scaler by computing per-feature minimum and maximum
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

        let mut min = Array1::from_elem(n_features, f64::INFINITY);
        let mut max = Array1::from_elem(n_features, f64::NEG_INFINITY);

        // accumulate per-feature min and max
        for row in features.outer_iter() {
            Zip::from(&row)
                .and(&mut min)
                .and(&mut max)
                .for_each(|&v, mn, mx| {
                    if v < *mn { *mn = v; }
                    if v > *mx { *mx = v; }
                });
        }

        // compute range — constant features get range = 1.0 to avoid division by zero
        Zip::from(&mut max)
            .and(&min)
            .for_each(|mx, &mn| {
                let r = *mx - mn;
                *mx = if r < 1e-10 { 1.0 } else { r };
            });

        self.min   = Some(min);
        self.range = Some(max);
    }

    /// Transforms features by scaling each to the `[0, 1]` range
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - scaler not fitted
    /// - feature dimension mismatch
    pub fn transform(&self, features: ArrayView2<f64>) -> Array2<f64> {

        let min   = self.min.as_ref().expect("Scaler not fitted");
        let range = self.range.as_ref().expect("Scaler not fitted");

        assert_eq!(
            features.ncols(),
            min.len(),
            "Feature dimension mismatch"
        );

        let mut data = features.to_owned();

        for mut row in data.outer_iter_mut() {
            Zip::from(&mut row)
                .and(min)
                .and(range)
                .for_each(|x, &mn, &r| {
                    *x = (*x - mn) / r;
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
