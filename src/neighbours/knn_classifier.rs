use std::collections::HashMap;

use crate::{
    core::DenseMatrix,
    neighbours::{
        DistanceMetric,
        NeighbourSearch,
        Weight,
        brute::BruteForce,
        metrics::Euclidean,
    },
};

/// K-Nearest Neighbors (KNN) Classifier.
///
/// A fully generic and zero-cost implementation of the KNN classification algorithm.
///
/// # Type Parameters
///
/// - `M` — Distance metric used to compute similarity.
/// - `N` — Nearest neighbor search strategy (e.g., [`BruteForce`], [`KDTree`]).
///
/// This design avoids dynamic dispatch and runtime branching by using
/// compile-time generics.
///
/// # Features
///
/// - Custom number of neighbors (`k`)
/// - Pluggable neighbor search algorithms
/// - Pluggable distance metrics
/// - Uniform or distance-based weighting
///
/// # Workflow
///
/// 1. Create a model using [`KNNClassifier::new`] or [`KNNClassifier::builder`].
/// 2. Call [`fit`] with training data.
/// 3. Call [`predict`] to obtain class predictions.
///
/// # Example
///
/// ```ignore
/// use crate::neighbours::{KNNClassifier, KDTree};
///
/// let mut model = KNNClassifier::new()
///     .algorithm(KDTree::new(30))
///     .k(5)
///     .build();
///
/// model.fit(&x_train, &y_train);
/// let predictions = model.predict(&x_test);
/// ```
pub struct KNNClassifier<M, N>
where
    M: DistanceMetric,
    N: NeighbourSearch<M>,
{
    /// Number of nearest neighbors considered.
    k: usize,

    /// Neighbor search strategy.
    searcher: N,

    /// Distance metric used for similarity computation.
    metric: M,

    /// Weighting strategy for aggregation.
    weights: Weight,

    /// Training class labels (set during `fit`).
    targets: Option<Vec<usize>>,
}

//
// ─────────────────────────────── Builder ─────────────────────────────
//

/// Builder for configuring a [`KNNClassifier`].
///
/// Allows customization of:
///
/// - Number of neighbors (`k`)
/// - Search algorithm
/// - Distance metric
/// - Weighting strategy
///
/// By default:
///
/// - `k = 5`
/// - Algorithm = [`BruteForce`]
/// - Metric = [`Euclidean`]
/// - Weighting = [`Weight::Uniform`]
pub struct Builder<M, N>
where
    M: DistanceMetric,
    N: NeighbourSearch<M>,
{
    k: usize,
    searcher: N,
    metric: M,
    weights: Weight,
}

/// Default builder using Euclidean distance and brute-force search.
impl Default for Builder<Euclidean, BruteForce> {
    fn default() -> Self {
        Self {
            k: 5,
            searcher: BruteForce::new(),
            metric: Euclidean,
            weights: Weight::Uniform,
        }
    }
}

/// Entry point for creating a default classifier.
///
/// Equivalent to:
///
/// ```ignore
/// Builder::default().build()
/// ```
impl KNNClassifier<Euclidean, BruteForce> {
    /// Creates a classifier with default configuration.
    pub fn new() -> KNNClassifier<Euclidean, BruteForce> {
        Builder::default().build()
    }

    /// Returns a configurable builder initialized with defaults.
    pub fn builder() -> Builder<Euclidean, BruteForce> {
        Builder::default()
    }
}

impl<M, N> Builder<M, N>
where
    M: DistanceMetric,
    N: NeighbourSearch<M>,
{
    /// Sets number of neighbors (`k`).
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets weighting strategy.
    pub fn weights(mut self, weights: Weight) -> Self {
        self.weights = weights;
        self
    }

    /// Changes the neighbor search strategy.
    ///
    /// This changes the type parameter `N`.
    pub fn algorithm<N2>(self, searcher: N2) -> Builder<M, N2>
    where
        N2: NeighbourSearch<M>,
    {
        Builder {
            k: self.k,
            searcher,
            metric: self.metric,
            weights: self.weights,
        }
    }

    /// Changes the distance metric.
    ///
    /// This changes the type parameter `M`.
    pub fn metric<M2>(self, metric: M2) -> Builder<M2, N>
    where
        M2: DistanceMetric,
        N: NeighbourSearch<M2>,
    {
        Builder {
            k: self.k,
            searcher: self.searcher,
            metric,
            weights: self.weights,
        }
    }

    /// Builds an unfitted [`KNNClassifier`].
    ///
    /// You must call [`KNNClassifier::fit`] before [`KNNClassifier::predict`].
    pub fn build(self) -> KNNClassifier<M, N> {
        KNNClassifier {
            k: self.k,
            searcher: self.searcher,
            metric: self.metric,
            weights: self.weights,
            targets: None,
        }
    }
}

//
// ───────────────────────────── Classifier Impl ─────────────────────────────
//

impl<M, N> KNNClassifier<M, N>
where
    M: DistanceMetric,
    N: NeighbourSearch<M>,
{
    /// Fits the classifier using training data.
    ///
    /// # Arguments
    ///
    /// * `features` — Training feature matrix.
    /// * `targets` — Class labels corresponding to each row.
    ///
    /// # Panics
    ///
    /// Panics if number of rows in `features`
    /// does not match length of `targets`.
    pub fn fit(&mut self, features: &DenseMatrix, targets: &[usize]) {
        assert_eq!(features.n_rows(), targets.len());

        self.targets = Some(targets.to_vec());
        self.searcher.fit(features.clone());
    }

    /// Predicts class labels for input samples.
    ///
    /// For each row in `x`:
    ///
    /// 1. Finds the `k` nearest neighbors.
    /// 2. Applies selected weighting strategy.
    /// 3. Aggregates votes per class.
    /// 4. Returns the label with highest total weight.
    ///
    /// # Panics
    ///
    /// Panics if model has not been fitted.
    pub fn predict(&self, x: &DenseMatrix) -> Vec<usize> {
        let targets = self.targets.as_ref().expect("model not fitted");


        let mut predictions = Vec::with_capacity(x.n_rows());

        for row in 0..x.n_rows() {
            let query = x.row(row);

            let neighbours = self.searcher.neighbours(
                query,
                self.k,
                &self.metric,
            );

            let mut votes: HashMap<usize, f64> = HashMap::new();

            for (dist, idx) in neighbours {
                let weight = match self.weights {
                    Weight::Uniform => 1.0,
                    Weight::Distance => {
                        if dist == 0.0 { 1.0 } else { 1.0 / dist }
                    }
                };

                let label = targets[idx];
                *votes.entry(label).or_insert(0.0) += weight;
            }

            let best_label = votes
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(label, _)| label)
                .unwrap();

            predictions.push(best_label);
        }

        predictions
    }
}
