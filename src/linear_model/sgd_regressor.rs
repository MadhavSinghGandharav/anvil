
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
/// The prediction function is:
///
/// `ŷ = w^T x + b`
///
/// where:
/// - `w` is the weight vector
/// - `b` is the bias (intercept)
///
/// # Training
///
/// During training:
/// - Data is shuffled every epoch.
/// - Gradients are accumulated per mini-batch.
/// - The optimizer updates both weights and bias.
///
/// # Example
///
/// ```ignore
/// let model = SGDRegressor::builder()
///     .epochs(200)
///     .build();
/// ```
pub struct SGDRegressor {
    /// Learned feature weights.
    ///
    /// Initialized inside [`fit`].
    weights: Vec<f64>,

    /// Learned bias (intercept).
    ///
    /// Stored as a length-1 array so it can be updated
    /// using the same optimizer interface as weights.
    bias: [f64; 1],

    /// Number of training epochs.
    epochs: usize,

    /// Mini-batch size.
    batch_size: usize,

    /// Optimizer used for parameter updates.
    optimizer: Box<dyn Optimizer>,
}

/// Builder for [`SGDRegressor`].
///
/// Allows configuration of:
/// - number of epochs
/// - batch size
/// - optimizer
///
/// # Defaults
///
/// - `epochs = 100`
/// - `batch_size = 1` (pure SGD)
/// - `optimizer = SGD::new(0.01)`
pub struct Builder {
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

impl Default for Builder {
    /// Creates a builder with default hyperparameters.
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
 /// Returns a builder for configuring an `SGDRegressor`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = SGDRegressor::builder()
    ///     .epochs(500)
    ///     .batch_size(32)
    ///     .build();
    /// ```

    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl SGDRegressor {

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias[0]
    }

    /// Fits the model using mini-batch SGD.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `features.n_rows() != target.len()`
    ///
    /// # Algorithm
    ///
    /// For each epoch:
    /// 1. Shuffle sample indices.
    /// 2. Iterate over mini-batches.
    /// 3. Accumulate gradients.
    /// 4. Average gradients.
    /// 5. Update weights and bias using the optimizer.
    pub fn fit(&mut self, features: &DenseMatrix, target: &[f64]) {
        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        assert_eq!(n_samples, target.len());

        // Initialize parameters
        self.weights = vec![0.0; n_features];

        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let mut gradient = vec![0.0; n_features];
        let mut bias_gradient = [0.0];

        for _ in 0..self.epochs {
            // Shuffle data each epoch
            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {
                gradient.fill(0.0);
                bias_gradient[0] = 0.0;

                for &idx in batch {
                    let row = features.row(idx);

                    // Prediction
                    let y_pred = dot(row, &self.weights) + self.bias[0];

                    // Error
                    let error = target[idx] - y_pred;

                    // Weight gradient accumulation
                    for j in 0..n_features {
                        gradient[j] +=  -error * row[j];
                    }

                    // Bias gradient accumulation
                    bias_gradient[0] += -error;
                }

                // Average gradients
                let inv_bs = 1.0 / batch.len() as f64;

                for g in &mut gradient {
                    *g *= inv_bs;
                }

                bias_gradient[0] *= inv_bs;

                // Update parameters
                self.optimizer.step(&mut self.weights, &gradient);
                self.optimizer.step(&mut self.bias, &bias_gradient);
            }
        }
    }
}

impl Builder {
    /// Sets the number of training epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the mini-batch size.
    ///
    /// `batch_size = 1` corresponds to pure SGD.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets a custom optimizer.
    ///
    /// Allows replacing the default `SGD` optimizer with
    /// another implementation of [`Optimizer`].
    pub fn optimizer<O:Optimizer + 'static>(mut self, optimizer: O) -> Builder {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Builds the `SGDRegressor`.
    ///
    /// The returned model is **untrained**.
    /// Parameters are initialized during [`fit`].
    pub fn build(self) -> SGDRegressor {
        SGDRegressor {
            weights: Vec::new(),
            bias: [0.0],
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
        }
    }
}
