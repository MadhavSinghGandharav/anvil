//! Nearest Neighbor algorithms and KNN models.
//!
//! This module provides:
//!
//! - Distance metrics (`DistanceMetric`)
//! - Neighbour search strategies (`NeighbourSearch`)
//! - KNN Classifier and Regressor
//! - Weighting strategies
//!
//! # Design
//!
//! The library is built around two core abstractions:
//!
//! - [`DistanceMetric`] — defines how distance between samples is computed.
//! - [`NeighbourSearch`] — defines how nearest neighbors are located.
//!
//! This separation allows combining different metrics and search strategies
//! without runtime overhead (fully generic, zero dynamic dispatch).
//!
//! # Example
//!
//! ```ignore
//! use crate::neighbours::{KNNClassifier, KDTree, Euclidean};
//!
//! let mut model = KNNClassifier::new()
//!     .algorithm(KDTree::new(30))
//!     .k(5)
//!     .build();
//!
//! model.fit(&x_train, &y_train);
//! let preds = model.predict(&x_test);
//! ```

mod knn_regressor;
mod knn_classifier;
mod metrics;

pub mod brute;
pub mod kd_tree;

use crate::core::DenseMatrix;

/// Trait defining a distance metric.
///
/// Any distance metric used in KNN must implement this trait.
///
/// # Requirements
///
/// - Should satisfy metric properties (non-negative, symmetric, triangle inequality)
///   if used with tree-based search structures like [`KDTree`].
/// - Must be cheap to compute since it is called in hot loops.
///
/// # Example
///
/// ```ignore
/// struct MyMetric;
///
/// impl DistanceMetric for MyMetric {
///     fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
///         // custom logic
///         0.0
///     }
/// }
/// ```
pub trait DistanceMetric {
    /// Computes distance between two feature slices.
    ///
    /// # Panics
    /// May panic if slice lengths differ.
    fn distance(&self, a: &[f64], b: &[f64]) -> f64;
}

/// Trait representing a nearest neighbor search strategy.
///
/// Implementations include:
///
/// - [`BruteForce`]
/// - [`KDTree`]
///
/// The searcher is responsible for:
/// 1. Building any internal structure during [`fit`].
/// 2. Returning the `k` nearest neighbors for a query point.
///
/// # Performance Notes
///
/// - Implementations are generic over the metric to allow
///   full compiler inlining (no dynamic dispatch).
/// - `fit` is called once during model training.
/// - `neighbours` may be called many times during prediction.
pub trait NeighbourSearch<M: DistanceMetric> {
    /// Builds internal search structure from training features.
    fn fit(&mut self, features: DenseMatrix);

    /// Returns the `k` nearest neighbors for a query point.
    ///
    /// # Arguments
    /// * `query` — Feature vector to query.
    /// * `features` — Training feature matrix.
    /// * `k` — Number of neighbors.
    /// * `metric` — Distance metric.
    ///
    /// # Returns
    /// A vector of `(distance, index)` pairs.
    fn neighbours(
        &self,
        query: &[f64],
        k: usize,
        metric: &M,
    ) -> Vec<(f64, usize)>;
}

/// Weighting strategy applied during neighbor aggregation.
///
/// Used by both [`KNNClassifier`] and [`KNNRegressor`].
#[derive(Clone, Copy, Debug)]
pub enum Weight {
    /// All neighbors contribute equally.
    Uniform,

    /// Each neighbor contributes weight = 1 / distance.
    ///
    /// If distance is zero, weight is treated as 1.0
    /// to avoid division by zero.
    Distance,
}

//
// ───────────────────────────── Re-exports ─────────────────────────────
//

/// Euclidean (L2) distance metric.
pub use metrics::Euclidean;

/// Manhattan (L1) distance metric.
pub use metrics::Manhattan;

/// Brute-force nearest neighbor search.
pub use brute::BruteForce;

/// KD-Tree based nearest neighbor search.
pub use kd_tree::KDTree;

/// K-Nearest Neighbors Regressor model.
pub use knn_regressor::KNNRegressor;

/// K-Nearest Neighbors Classifier model.
pub use knn_classifier::KNNClassifier;
