//! K-Nearest Neighbours (KNN) Regression module.
//!
//! This module provides a KNN regressor that predicts the target value of a sample
//! by interpolating the values of its $k$ nearest neighbours.

use ndarray::{Array1, ArrayView1, ArrayView2};

use crate::{
    core::{Estimator, Regressor, AnvilError},
    neighbours::{NeighbourSearch, Weight},
    neighbours::metric::Euclidean,
    neighbours::brute_force::BruteForce,
};

/// K-Nearest Neighbours regressor.
///
/// The target is predicted by local interpolation of the targets associated 
/// of the nearest neighbours in the training set.
///
/// # Examples
///
/// ```
/// use anvil::models::KNNRegressor;
/// use anvil::neighbours::Weight;
///
/// let model = KNNRegressor::builder()
///     .k(5)
///     .weights(Weight::Uniform)
///     .build()
///     .unwrap();
/// ```
pub struct KNNRegressor<N>
where
    N: NeighbourSearch,
{
    k: usize,
    searcher: N,
    weights: Weight,
    targets: Option<Vec<f64>>,
}

/// A builder for configuring and creating a [`KNNRegressor`].
pub struct Builder<N>
where
    N: NeighbourSearch,
{
    k: usize,
    searcher: N,
    weights: Weight,
}

impl Default for Builder<BruteForce<Euclidean>> {
    fn default() -> Self {
        Self {
            k: 5,
            searcher: BruteForce::new(),
            weights: Weight::Uniform,
        }
    }
}

impl<N> Builder<N>
where
    N: NeighbourSearch,
{
    /// Sets the number of neighbours to use for queries.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Sets the weighting function used in prediction.
    /// 
    /// * `Uniform`: All points in each neighbourhood are weighted equally.
    /// * `Distance`: Weights points by the inverse of their distance.
    pub fn weights(mut self, weights: Weight) -> Self {
        self.weights = weights;
        self
    }

    /// Sets the underlying search algorithm (e.g., BruteForce).
    pub fn algorithm<S: NeighbourSearch>(self, searcher: S) -> Builder<S> {
        Builder {
            k: self.k,
            searcher,
            weights: self.weights,
        }
    }

    /// Consumes the builder and returns a [`KNNRegressor`].
    ///
    /// # Errors
    ///
    /// Returns [`AnvilError::InvalidParam`] if `k` is 0.
    pub fn build(self) -> Result<KNNRegressor<N>, AnvilError> {
        
        if self.k == 0 {
            return Err(AnvilError::InvalidParam {
                param: "k",
                reason: "must be > 0".into(),
            });
        }

        Ok(KNNRegressor {
            k: self.k,
            searcher: self.searcher,
            weights: self.weights,
            targets: None,
        })
    }
}

impl KNNRegressor<BruteForce<Euclidean>> {
    /// Returns a new instance of [`KNNRegressor`] with default settings.
    pub fn new() -> Result<Self, AnvilError> {
        Builder::default().build()
    }

    /// Returns a [`Builder`] with default BruteForce and Euclidean settings.
    pub fn builder() -> Builder<BruteForce<Euclidean>> {
        Builder::default()
    }
}

impl<N> Estimator<f64> for KNNRegressor<N>
where
    N: NeighbourSearch,
{
    /// Fits the KNN regressor by building the search index with training data.
    ///
    /// # Errors
    ///
    /// * `AnvilError::DimensionMismatch`: If `x` and `y` sample counts do not match.
    /// * `AnvilError::EmptyDataset`: If `x` is empty.
    /// * `AnvilError::InvalidParam`: If `k` is 0 or greater than the number of samples.
    fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<(), AnvilError> {

        let n_samples = x.nrows();

        if n_samples != y.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: y.len(),
            });
        }

        if n_samples == 0 {
            return Err(AnvilError::EmptyDataset { target: "X" });
        }

        if self.k == 0 {
            return Err(AnvilError::InvalidParam {
                param: "k",
                reason: "k must be > 0".into(),
            });
        }

        if self.k > n_samples {
            return Err(AnvilError::InvalidParam {
                param: "k",
                reason: "k cannot be greater than number of samples".into(),
            });
        }

        self.targets = Some(y.to_vec());
        self.searcher.build(x.to_owned());

        Ok(())
    }
}

impl<N> Regressor for KNNRegressor<N>
where
    N: NeighbourSearch,
{
    /// Predicts target values for the provided test samples.
    ///
    /// # Errors
    ///
    /// * `AnvilError::NotFitted`: If the model has not been trained.
    /// * `AnvilError::InvalidParam`: If the sum of weights is zero.
    fn predict(
        &self,
        x: ArrayView2<f64>,
    ) -> Result<Array1<f64>, AnvilError> {

        let targets = self.targets.as_ref().ok_or(AnvilError::NotFitted)?;

        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in x.outer_iter().enumerate() {

            let neighbours = self.searcher.query(row, self.k);

            let mut numerator = 0.0;
            let mut denominator = 0.0;

            for (idx, dist) in neighbours {

                let weight = match self.weights {
                    Weight::Uniform => 1.0,
                    Weight::Distance => {
                        if dist == 0.0 { 1.0 } else { 1.0 / dist }
                    }
                };

                numerator += weight * targets[idx];
                denominator += weight;
            }

            if denominator == 0.0 {
                return Err(AnvilError::InvalidParam {
                    param: "weights",
                    reason: "sum of weights is zero".into(),
                });
            }

            predictions[i] = numerator / denominator;
        }

        Ok(predictions)
    }
}
