use crate::{
    core::{DenseMatrix, utils::dot},
    optim::{Optimizer, SGD},
};
use rand::seq::SliceRandom;

/// Stochastic Gradient Descent regressor.
///
/// `SGDRegressor` implements linear regression using
/// mini-batch stochastic gradient descent.
///
/// The optimizer is generic and supplied at build time.
/// Model parameters (`weights` and `bias`) are initialized
/// during [`fit`].
///
/// # Model
///
/// Prediction function:
///
/// `ŷ = w^T x + b`
pub struct SGDRegressor {
    /// Learned feature weights.
    ///
    /// Initialized during [`fit`].
    weights: Option<Vec<f64>>,

    /// Learned bias (intercept).
    ///
    /// Stored as a length-1 array so it can be updated
    /// using the same optimizer interface as weights.
    bias: Option<[f64; 1]>,

    /// Number of training epochs.
    epochs: usize,

    /// Mini-batch size used for SGD updates.
    batch_size: usize,

    /// Optimizer used for parameter updates.
    optimizer: Box<dyn Optimizer>,
}

/// Builder for configuring [`SGDRegressor`].
///
/// Allows setting:
///
/// - `epochs`
/// - `batch_size`
/// - `optimizer`
pub struct Builder {
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

impl Default for Builder {
    /// Creates a builder with default hyperparameters.
    ///
    /// Defaults:
    ///
    /// - `epochs = 100`
    /// - `batch_size = 1`
    /// - `optimizer = SGD(learning_rate = 0.01)`
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 1,
            optimizer: Box::new(SGD::new(0.01)),
        }
    }
}

impl SGDRegressor {

    /// Creates an `SGDRegressor` using default settings.
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// SGDRegressor::builder().build()
    /// ```
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a builder used to configure the model.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns learned weights.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been fitted.
    pub fn weights(&self) -> &[f64] {
        self.weights.as_ref().expect("model not fitted")
    }

    /// Returns the learned bias term.
    ///
    /// # Panics
    ///
    /// Panics if the model has not been fitted.
    pub fn bias(&self) -> f64 {
        self.bias.as_ref().expect("model not fitted")[0]
    }

    /// Fits the model using mini-batch SGD.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - `features.n_rows() != target.len()`
    /// - `batch_size == 0`
    pub fn fit(&mut self, features: &DenseMatrix, target: &[f64]) {

        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        assert_eq!(
            n_samples,
            target.len(),
            "Number of samples and target values must match"
        );

        assert!(
            self.batch_size > 0,
            "batch_size must be greater than 0"
        );

        // initialize parameters
        let mut weights = vec![0.0; n_features];
        let mut bias = [0.0];

        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let mut gradient = vec![0.0; n_features];
        let mut bias_gradient = [0.0];

        for _ in 0..self.epochs {

            // shuffle samples each epoch
            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {

                gradient.fill(0.0);
                bias_gradient[0] = 0.0;

                for &idx in batch {

                    let row = features.row(idx);

                    // prediction
                    let y_pred = dot(row, &weights) + bias[0];

                    // error
                    let error = target[idx] - y_pred;

                    // accumulate gradients
                    for (g, &x) in gradient.iter_mut().zip(row) {
                        *g += -error * x;
                    }

                    bias_gradient[0] += -error;
                }

                // average gradients
                let inv_bs = 1.0 / batch.len() as f64;

                for g in &mut gradient {
                    *g *= inv_bs;
                }

                bias_gradient[0] *= inv_bs;

                // update parameters
                self.optimizer.step(&mut weights, &gradient);
                self.optimizer.step(&mut bias, &bias_gradient);
            }
        }

        self.weights = Some(weights);
        self.bias = Some(bias);
    }

    /// Predicts target values for the given feature matrix.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - model is not fitted
    /// - feature dimension mismatch
    pub fn predict(&self, features: &DenseMatrix) -> Vec<f64>{

        let weights = self.weights.as_ref().expect("Model not fitted");
        let bias = self.bias.as_ref().expect("Model not fitted")[0];

        assert!(
            weights.len() == features.n_cols(),
            "Feature dimension mismatch"
        );

        let mut preds = Vec::with_capacity(features.n_rows());

        for i in 0..features.n_rows(){
            let row = features.row(i);
            let pred = dot(row, weights) + bias;
            preds.push(pred);
        }

        preds
    }
}

impl Builder {

    /// Sets number of training epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets mini-batch size.
    ///
    /// `batch_size = 1` corresponds to pure SGD.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        self.batch_size = batch_size;
        self
    }

    /// Sets a custom optimizer.
    ///
    /// Example:
    ///
    /// ```ignore
    /// .optimizer(Adam::new(0.001))
    /// ```
    pub fn optimizer<O:Optimizer + 'static>(mut self, optimizer: O) -> Builder {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Builds the `SGDRegressor`.
    ///
    /// Returned model is **untrained**.
    /// Parameters are initialized during [`fit`].
    pub fn build(self) -> SGDRegressor {
        SGDRegressor {
            weights: None,
            bias: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
        }
    }
}
