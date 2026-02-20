use std::collections::HashMap;

use crate::{
    core::DenseMatrix,
    neighbours::{
        Algorithm, DistanceMetric, Weight, algorithm::BruteForce, metrics::Euclidean
    },
};

/// K-Nearest Neighbors (KNN) Classifier.
///
/// A configurable implementation of the KNN classification algorithm.
///
/// # Features
/// - Custom number of neighbors (`k`)
/// - Pluggable neighbor search algorithms (e.g., brute force)
/// - Custom distance metrics
/// - Uniform or distance-based weighting
///
/// # Workflow
/// 1. Create the model using [`new`] or [`builder`].
/// 2. Call [`fit`] with training data.
/// 3. Call [`predict`] to obtain class predictions.
///
/// # Notes
/// - The model must be fitted before calling [`predict`].
/// - Training data is cloned and stored internally during fitting.
pub struct KNNClassifier<M: DistanceMetric> {
    /// Number of nearest neighbors considered for voting.
    k: usize,

    /// Strategy used to search for nearest neighbors.
    algorithm: Algorithm,

    /// Distance metric used to compute similarity between samples.
    metric: M,

    /// Weighting strategy applied during neighbor aggregation.
    weights: Weight,

    /// Training feature matrix (available after calling `fit`).
    features: Option<DenseMatrix>,

    /// Training class labels (available after calling `fit`).
    targets: Option<Vec<usize>>,
}

/// Builder for configuring a [`KNNClassifier`].
///
/// Allows customization of:
/// - Number of neighbors (`k`)
/// - Neighbor search algorithm
/// - Distance metric
/// - Weighting strategy
///
/// The resulting model is unfitted and must call [`KNNClassifier::fit`]
/// before predictions can be made.
pub struct Builder<M: DistanceMetric> {
    k: usize,
    algorithm: Algorithm,
    metric: M,
    weights: Weight,
}

/// Default builder configuration using:
/// - `k = 5`
/// - Brute-force neighbor search
/// - Euclidean distance metric
/// - Uniform weighting
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

impl KNNClassifier<Euclidean> {
    /// Creates a new classifier with default configuration.
    ///
    /// Equivalent to calling `KNNClassifier::builder().build()`.
    pub fn new() -> Self {
        Builder::default().build()
    }

    /// Returns a builder initialized with default configuration.
    ///
    /// This allows customization before building the model.
    pub fn builder() -> Builder<Euclidean> {
        Builder::default()
    }
}

impl<M: DistanceMetric> KNNClassifier<M> {
    /// Fits the classifier using training data.
    ///
    /// # Arguments
    /// - `features`: Training feature matrix (rows = samples).
    /// - `targets`: Class labels corresponding to each sample.
    ///
    /// # Panics
    /// Panics if the number of rows in `features`
    /// does not match the length of `targets`.
    ///
    /// # Behavior
    /// Training data is cloned and stored internally.
    pub fn fit(&mut self, features: &DenseMatrix, targets: &[usize]) {
        assert_eq!(features.n_rows(), targets.len());

        self.features = Some(features.clone());
        self.targets = Some(targets.to_vec());
    }

    /// Predicts class labels for the given input matrix.
    ///
    /// For each input sample:
    /// 1. Finds the `k` nearest neighbors.
    /// 2. Computes weights based on the selected strategy.
    /// 3. Aggregates votes per class.
    /// 4. Returns the class with the highest total weight.
    ///
    /// # Arguments
    /// - `x`: Input feature matrix.
    ///
    /// # Returns
    /// A vector of predicted class labels (one per input row).
    ///
    /// # Panics
    /// Panics if the model has not been fitted.
    pub fn predict(&self, x: &DenseMatrix) -> Vec<usize> {
        let features = self.features.as_ref().expect("model not fitted");
        let targets = self.targets.as_ref().expect("model not fitted");

        assert_eq!(features.n_cols(), x.n_cols());

        let mut predictions: Vec<usize> = Vec::with_capacity(x.n_rows());

        for row in 0..x.n_rows() {
            let row_slice = x.row(row);

            let neighbours = match self.algorithm {
                Algorithm::Brute => {
                    BruteForce::neighbours(features, row_slice, self.k, &self.metric)
                }
            };

            let mut map: HashMap<usize, f64> = HashMap::new();

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

                let label = targets[idx];
                *map.entry(label).or_insert(0.0) += weight;
            }

            let mut max_score = f64::NEG_INFINITY;
            let mut best_label = 0;

            for (&label, &score) in map.iter() {
                if score > max_score {
                    max_score = score;
                    best_label = label;
                }
            }

            predictions.push(best_label);
        }

        predictions
    }
}

impl<M: DistanceMetric> Builder<M> {
    /// Builds an unfitted `KNNClassifier` instance.
    ///
    /// The returned model must call [`KNNClassifier::fit`]
    /// before predictions can be performed.
    pub fn build(self) -> KNNClassifier<M> {
        KNNClassifier {
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
