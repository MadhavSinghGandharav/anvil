use ndarray::{Array1, ArrayView1, ArrayView2};
use crate::neighbours::{NeighbourSearch, Weight};
use crate::neighbours::metric::Euclidean;
use crate::neighbours::brute_force::BruteForce;

/// K-Nearest Neighbors regressor.
///
/// A generic, zero-cost implementation of KNN regression.
///
/// The type parameter `N` determines the neighbour search strategy
/// used at compile time — no dynamic dispatch, no runtime branching.
///
/// # Notes
///
/// - Supports uniform and distance-based weighting
/// - Pluggable search algorithms (`BruteForce`, `KDTree`, ...)
/// - `BruteForce<Euclidean>` is the default configuration
pub struct KNNRegressor<N>
where
    N: NeighbourSearch,
{
    /// Number of nearest neighbours considered
    k: usize,

    /// Neighbour search strategy
    searcher: N,

    /// Weighting strategy for prediction aggregation
    weights: Weight,

    /// Training target values (set during `fit`)
    targets: Option<Vec<f64>>,
}

/// Builder for configuring [`KNNRegressor`]
pub struct Builder<N>
where
    N: NeighbourSearch,
{
    k: usize,
    searcher: N,
    weights: Weight,
}

impl Default for Builder<BruteForce<Euclidean>> {
    /// Default configuration
    ///
    /// - k = 5
    /// - algorithm = BruteForce
    /// - metric = Euclidean
    /// - weights = Uniform
    fn default() -> Self {
        Self {
            k: 5,
            searcher: BruteForce::new(),
            weights: Weight::Uniform,
        }
    }
}

impl KNNRegressor<BruteForce<Euclidean>> {

    /// Create model with default configuration
    pub fn new() -> Self {
        Builder::default().build()
    }

    /// Returns builder
    pub fn builder() -> Builder<BruteForce<Euclidean>> {
        Builder::default()
    }
}

impl<N> Builder<N>
where
    N: NeighbourSearch,
{
    /// Set number of neighbours
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set weighting strategy
    pub fn weights(mut self, weights: Weight) -> Self {
        self.weights = weights;
        self
    }

    /// Set neighbour search algorithm
    pub fn algorithm<S: NeighbourSearch>(self, searcher: S) -> Builder<S> {
        Builder {
            k: self.k,
            searcher,
            weights: self.weights,
        }
    }

    /// Build model
    pub fn build(self) -> KNNRegressor<N> {
        KNNRegressor {
            k: self.k,
            searcher: self.searcher,
            weights: self.weights,
            targets: None,
        }
    }
}

impl<N> KNNRegressor<N>
where
    N: NeighbourSearch,
{
    /// Fits the regressor using training data
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - features and targets size mismatch
    pub fn fit(&mut self, features: ArrayView2<f64>, targets: ArrayView1<f64>) {

        assert_eq!(
            features.nrows(),
            targets.len(),
            "Number of samples and target values must match"
        );

        self.targets = Some(targets.to_vec());
        self.searcher.build(features.to_owned());
    }

    /// Predict target values
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - model not fitted
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {

        let targets = self.targets.as_ref().expect("Model not fitted");

        let mut predictions = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {

            // find k nearest neighbours
            let neighbours = self.searcher.query(row, self.k);

            let mut numerator   = 0.0;
            let mut denominator = 0.0;

            // accumulate weighted target values
            for (idx, dist) in neighbours {

                let weight = match self.weights {
                    Weight::Uniform  => 1.0,
                    Weight::Distance => if dist == 0.0 { 1.0 } else { 1.0 / dist },
                };

                numerator   += weight * targets[idx];
                denominator += weight;
            }

            // weighted average of neighbour targets
            predictions[i] = numerator / denominator;
        }

        predictions
    }
}
