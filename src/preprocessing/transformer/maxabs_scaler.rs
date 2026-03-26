use ndarray::{Array1, Array2, ArrayView2, Zip};

/// MaxAbs scaler — scales each feature to the `[-1, 1]` range.
///
/// Transforms each feature using its maximum absolute value:
///
/// ```text
/// x' = x / max(|x|)
/// ```
///
/// # Notes
///
/// - Preserves sparsity (zero stays zero)
/// - If a feature has max_abs = 0, it is set to `1.0`
///   to avoid division by zero
pub struct MaxAbsScaler {
    /// Per-feature max absolute value computed during `fit`
    max_abs: Option<Array1<f64>>,
}

impl MaxAbsScaler {
    pub fn new() -> Self {
        Self {
            max_abs: None,
        }
    }

    pub fn fit(&mut self, features: ArrayView2<f64>) {
        assert!(features.nrows() > 0, "Cannot fit with zero samples");

        let n_features = features.ncols();

        let mut max_abs = Array1::from_elem(n_features, f64::NEG_INFINITY);

        // compute max abs per feature
        for row in features.outer_iter() {
            Zip::from(&row)
                .and(&mut max_abs)
                .for_each(|&v, mx| {
                    let val = v.abs();
                    if val > *mx {
                        *mx = val;
                    }
                });
        }

        // handle zero columns
        max_abs
            .iter_mut()
            .for_each(|x| if *x == 0.0 { *x = 1.0 });

        self.max_abs = Some(max_abs);
    }

    pub fn transform(&self, features: ArrayView2<f64>) -> Array2<f64> {
        let max_abs = self.max_abs.as_ref().expect("Scaler not fitted");

        assert_eq!(
            features.ncols(),
            max_abs.len(),
            "Feature dimension mismatch"
        );

        let mut data = features.to_owned();

        for mut row in data.outer_iter_mut() {
            Zip::from(&mut row)
                .and(max_abs)
                .for_each(|x, &mx| {
                    *x /= mx;
                });
        }

        data
    }

    pub fn fit_transform(&mut self, features: ArrayView2<f64>) -> Array2<f64> {
        self.fit(features);
        self.transform(features)
    }
}
