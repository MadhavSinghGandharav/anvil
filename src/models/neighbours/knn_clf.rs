use std::collections::HashMap;

use ndarray::{Array1, ArrayView2,ArrayView1};

use crate::neighbours::{NeighbourSearch,Weight, metric::Euclidean}; 
use crate::neighbours::brute_force::BruteForce;

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
pub struct KNNClassifier<N>
where
    N: NeighbourSearch,
{
    /// Number of nearest neighbors considered.
    k: usize,

    /// Neighbor search strategy.
    searcher: N,

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
pub struct Builder<N>
where
    N: NeighbourSearch,
{
    k: usize,
    searcher: N,
    weights: Weight,
}

/// Default builder using Euclidean distance and brute-force search.
impl Default for Builder<BruteForce<Euclidean>> {
    fn default() -> Self {
        Self {
            k: 5,
            searcher: BruteForce::new(),
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
impl KNNClassifier<BruteForce<Euclidean>> {
    /// Creates a classifier with default configuration.
    pub fn new() -> KNNClassifier<BruteForce<Euclidean>> {
        Builder::default().build()
    }

    /// Returns a configurable builder initialized with defaults.
    pub fn builder() -> Builder<BruteForce<Euclidean>> {
        Builder::default()
    }
}

impl<N> Builder <N>
where
    N: NeighbourSearch,
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
    pub fn algorithm<S: NeighbourSearch>(self, searcher: S) -> Builder<S>{
        Builder {
            k: self.k,
            searcher,
            weights: self.weights,
        }
    }

    /// Builds an unfitted [`KNNClassifier`].
    ///
    /// You must call [`KNNClassifier::fit`] before [`KNNClassifier::predict`].
    pub fn build(self) -> KNNClassifier<N> {
        KNNClassifier {
            k: self.k,
            searcher: self.searcher,
            weights: self.weights,
            targets: None,
        }
    }
}

//
// ───────────────────────────── Classifier Impl ─────────────────────────────
//

impl<N> KNNClassifier<N>
where
    N: NeighbourSearch,
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
    pub fn fit(&mut self, features: ArrayView2<f64>, targets: ArrayView1<usize>) {
        assert_eq!(features.nrows(), targets.len());

        self.targets = Some(targets.to_vec());
        self.searcher.build(features.to_owned());
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
    pub fn predict(&self, x: ArrayView2<f64>) -> Array1<usize> {
        let targets = self.targets.as_ref().expect("model not fitted");

        let samples = x.nrows();
        let mut predictions = Array1::zeros(samples);
        let mut votes: HashMap<usize, f64> = HashMap::new();

        for (i,row) in x.outer_iter().enumerate() {

            let neighbours = self.searcher.query(
                row,
                self.k,
            );
            votes.clear();
            
            for (idx,dist) in neighbours {
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
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(&label, _)| label)
                .unwrap();

            predictions[i] = best_label;
        }

        predictions
    }
}
