use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rand::seq::SliceRandom;

use crate::optim::{Optimizer, SGD};

/// Linear regression model trained using mini-batch Stochastic Gradient Descent.
///
/// The model learns parameters `w` and `b` such that:
///
/// `ŷ = wᵀx + b`
pub struct SGDRegressor {
    /// Learned parameters `[bias, weights...]`
    params: Option<Array1<f64>>,

    /// Number of training epochs
    epochs: usize,

    /// Mini-batch size
    batch_size: usize,

    /// Optimizer used for parameter updates
    optimizer: Box<dyn Optimizer>,
}

/// Builder for configuring [`SGDRegressor`]
pub struct Builder {
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - epochs = 100
    /// - batch_size = 1
    /// - optimizer = SGD(0.01)
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 1,
            optimizer: Box::new(SGD::new(0.01)),
        }
    }
}

impl Builder {

    /// Set number of epochs
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Set mini-batch size
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        self.batch_size = batch_size;
        self
    }

    /// Set optimizer
    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    /// Build model
    pub fn build(self) -> SGDRegressor {
        SGDRegressor {
            params: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
        }
    }
}

impl SGDRegressor {

    /// Create model with default configuration
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns builder
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns learned weights
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted
    pub fn weights(&self) -> ArrayView1<'_, f64> {
        let params = self.params.as_ref().expect("Model not fitted");
        params.slice(s![1..])
    }

    /// Returns learned bias
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted
    pub fn bias(&self) -> f64 {
        self.params.as_ref().expect("Model not fitted")[0]
    }

    /// Fits the model using mini-batch stochastic gradient descent
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - features and target size mismatch
    /// - batch_size == 0
    pub fn fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<f64>) {

        let n_samples = features.nrows();
        let n_features = features.ncols();

        assert_eq!(
            n_samples,
            target.len(),
            "Number of samples and target values must match"
        );

        assert!(
            self.batch_size > 0,
            "batch_size must be greater than 0"
        );

        // initialize parameters [bias, weights...]
        let mut params = Array1::zeros(n_features + 1);

        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let mut gradient = Array1::<f64>::zeros(n_features + 1);

        for _ in 0..self.epochs {

            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {

                gradient.fill(0.0);

                // slice once per batch
                let weights = params.slice(s![1..]);
                let bias = params[0];

                let mut grad_w = gradient.slice_mut(s![1..]);
                let mut grad_b = 0.0;

                for &idx in batch {

                    let row = features.row(idx);

                    // prediction
                    let y_pred = row.dot(&weights) + bias;

                    // error
                    let error = target[idx] - y_pred;
                    let grad = -error;

                    // bias gradient
                    grad_b += grad;

                    // weight gradients
                    for (g, &x) in grad_w.iter_mut().zip(row.iter()) {
                        *g += grad * x;
                    }
                }
                gradient[0] = grad_b;

                // average gradient
                let inv_bs = 1.0 / batch.len() as f64;
                gradient *= inv_bs; 

                self.optimizer
                    .step(params.as_slice_mut().unwrap(), gradient.as_slice().unwrap());
            }
        }

        self.params = Some(params);
    }

    /// Predict target values
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - model not fitted
    /// - feature dimension mismatch
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {

        let params = self.params.as_ref().expect("Model not fitted");

        let weights = params.slice(s![1..]);
        let bias = params[0];

        assert!(
            weights.len() == features.ncols(),
            "Feature dimension mismatch"
        );

        let mut preds = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {
            preds[i] = row.dot(&weights) + bias;
        }

        preds
    }
}
