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

/// K-Nearest Neighbors (KNN) Regressor.
///
/// A fully generic and zero-cost implementation of the KNN regression algorithm.
///
/// # Type Parameters
///
/// - `M` — Distance metric used to compute similarity.
/// - `N` — Nearest neighbor search strategy (e.g., [`BruteForce`], [`KDTree`]).
///
/// This implementation uses compile-time generics to avoid dynamic dispatch
/// and runtime branching.
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
/// 1. Create a model using [`KNNRegressor::new`] or [`KNNRegressor::builder`].
/// 2. Call [`fit`] with training data.
/// 3. Call [`predict`] to obtain predictions.
///
/// # Example
///
/// ```ignore
/// use crate::neighbours::{KNNRegressor, KDTree};
///
/// let mut model = KNNRegressor::new()
///     .algorithm(KDTree::new(30))
///     .k(5)
///     .build();
///
/// model.fit(&x_train, &y_train);
/// let preds = model.predict(&x_test);
/// ```
pub struct KNNRegressor<M, N>
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

    /// Weighting strategy applied during aggregation.
    weights: Weight,


    /// Training target values (set during `fit`).
    targets: Option<Vec<f64>>,
}

//
// ───────────────────────────── Builder ─────────────────────────────
//

/// Builder for configuring a [`KNNRegressor`].
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

/// Entry point for creating a default regressor.
impl KNNRegressor<Euclidean, BruteForce> {
    /// Creates a regressor with default configuration.
    pub fn new() -> KNNRegressor<Euclidean, BruteForce> {
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

    /// Changes neighbor search strategy.
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

    /// Changes distance metric.
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

    /// Builds an unfitted [`KNNRegressor`].
    ///
    /// You must call [`KNNRegressor::fit`] before [`KNNRegressor::predict`].
    pub fn build(self) -> KNNRegressor<M, N> {
        KNNRegressor {
            k: self.k,
            searcher: self.searcher,
            metric: self.metric,
            weights: self.weights,
            targets: None,
        }
    }
}

//
// ───────────────────────────── Regressor Impl ─────────────────────────────
//

impl<M, N> KNNRegressor<M, N>
where
    M: DistanceMetric,
    N: NeighbourSearch<M>,
{
    /// Fits the regressor using training data.
    ///
    /// # Arguments
    ///
    /// * `features` — Training feature matrix.
    /// * `targets` — Target values corresponding to each row.
    ///
    /// # Panics
    ///
    /// Panics if number of rows in `features`
    /// does not match length of `targets`.
    pub fn fit(&mut self, features: &DenseMatrix, targets: &[f64]) {
        assert_eq!(features.n_rows(), targets.len());

        self.targets = Some(targets.to_vec());

        self.searcher.fit(features.clone());
    }

    /// Predicts target values for input samples.
    ///
    /// For each row in `x`:
    ///
    /// 1. Finds the `k` nearest neighbors.
    /// 2. Applies selected weighting strategy.
    /// 3. Computes weighted mean of neighbor targets.
    ///
    /// # Panics
    ///
    /// Panics if model has not been fitted.
    pub fn predict(&self, x: &DenseMatrix) -> Vec<f64> {
        let targets = self.targets.as_ref().expect("model not fitted");


        let mut predictions = Vec::with_capacity(x.n_rows());

        for row in 0..x.n_rows() {
            let query = x.row(row);

            let neighbours = self.searcher.neighbours(
                query,
                self.k,
                &self.metric,
            );

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for (dist, idx) in neighbours {
                let weight = match self.weights {
                    Weight::Uniform => 1.0,
                    Weight::Distance => {
                        if dist == 0.0 { 1.0 } else { 1.0 / dist }
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
