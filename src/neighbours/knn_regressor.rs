
use crate::{
    core::DenseMatrix,
    neighbours::{
        Algorithm, DistanceMetric, Weight, algorithm::BruteForce, metrics::Euclidean
    },
};

/// K-Nearest Neighbors Regressor.
///
/// This struct implements a configurable KNN regression model.
/// It supports:
/// - Custom number of neighbors (`k`)
/// - Different neighbor search algorithms
/// - Pluggable distance metrics
/// - Uniform or distance-based weighting
///
/// The model must be fitted using [`fit`] before calling [`predict`].
pub struct KNNRegressor<M: DistanceMetric> {
    /// Number of nearest neighbors to consider.
    k: usize,

    /// Neighbor search strategy (e.g., brute force).
    algorithm: Algorithm,

    /// Distance metric used to compute similarity.
    metric: M,

    /// Weighting strategy for neighbor aggregation.
    weights: Weight,

    /// Training feature matrix (set after `fit`).
    features: Option<DenseMatrix>,

    /// Training target values (set after `fit`).
    targets: Option<Vec<f64>>,
}

/// Builder for configuring a [`KNNRegressor`].
///
/// Allows setting:
/// - Number of neighbors
/// - Algorithm
/// - Metric
/// - Weighting strategy
pub struct Builder<M: DistanceMetric> {
    k: usize,
    algorithm: Algorithm,
    metric: M,
    weights: Weight,
}

/// Default builder configuration using Euclidean distance
/// and brute-force neighbor search.
impl Default for Builder<Euclidean> {
    fn default() -> Self {
        Self {
            k: 5,
            algorithm: Algorithm::Brute,
            metric: Euclidean,
            weights: Weight::Uniform,
        }
    }
}

impl KNNRegressor<Euclidean> {
    /// Creates a new `KNNRegressor` with default configuration.
    pub fn new() -> Self {
        Builder::default().build()
    }

    /// Returns a builder initialized with default configuration.
    pub fn builder() -> Builder<Euclidean> {
        Builder::default()
    }
}

impl<M: DistanceMetric> KNNRegressor<M> {
    /// Fits the model using training data.
    ///
    /// # Arguments
    /// * `features` - Training feature matrix
    /// * `targets` - Target values corresponding to each row
    ///
    /// # Panics
    /// Panics if number of rows in `features`
    /// does not match length of `targets`.
    pub fn fit(&mut self, features: &DenseMatrix, targets: &[f64]) {
        assert_eq!(features.n_rows(), targets.len());

        self.features = Some(features.clone());
        self.targets = Some(targets.to_vec());
    }

    /// Predicts target values for the given input matrix.
    ///
    /// For each row in `x`:
    /// 1. Finds `k` nearest neighbors.
    /// 2. Applies the selected weighting strategy.
    /// 3. Computes weighted mean of neighbor targets.
    ///
    /// # Arguments
    /// * `x` - Input feature matrix
    ///
    /// # Returns
    /// Vector of predicted values (one per row).
    ///
    /// # Panics
    /// Panics if the model has not been fitted.
    pub fn predict(&self, x: &DenseMatrix) -> Vec<f64> {
        let features = self.features.as_ref().expect("model not fitted");
        let targets = self.targets.as_ref().expect("model not fitted");

        assert_eq!(features.n_cols(), x.n_cols());

        let mut predictions = Vec::with_capacity(x.n_rows());

        for row in 0..x.n_rows() {
            let row_slice = x.row(row);

            let neighbours = match self.algorithm {
                Algorithm::Brute => {
                    BruteForce::neighbours(features, row_slice, self.k, &self.metric)
                }
            };

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for (dist, idx) in neighbours {
                let weight = match self.weights {
                    Weight::Uniform => 1.0,
                    Weight::Distance => {
                        if dist == 0.0 {
                            1.0
                        } else {
                            1.0 / dist
                        }
                    }
                };

                numerator += weight * targets[idx];
                denominator += weight;
            }

            predictions.push(numerator / denominator);
        }

        predictions
    }
}

impl<M: DistanceMetric> Builder<M> {
    /// Builds a `KNNRegressor` instance using the configured parameters.
    ///
    /// The returned model is unfitted and must call `fit()` before `predict()`.
    pub fn build(self) -> KNNRegressor<M> {
        KNNRegressor {
            k: self.k,
            algorithm: self.algorithm,
            metric: self.metric,
            weights: self.weights,
            features: None,
            targets: None,
        }
    }

    /// Sets the number of neighbors (`k`).
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the neighbor search algorithm.
    pub fn algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Sets the distance metric.
    pub fn metric(mut self, metric: M) -> Self {
        self.metric = metric;
        self
    }

    /// Sets the weighting strategy.
    pub fn weights(mut self, weights: Weight) -> Self {
        self.weights = weights;
        self
    }
}
