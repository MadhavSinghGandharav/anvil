use core::f64;

use crate::{
    core::{DenseMatrix, utils::dot},
    optim::{Optimizer, SGD}, preprocessing::LabelEncoder,
};
use rand::seq::SliceRandom;

/// Linear binary classifier trained using the Perceptron algorithm.
///
/// The model learns a linear decision boundary of the form:
///
/// `f(x) = w^T x + b`
///
/// where:
/// - `w` is the weight vector
/// - `b` is the bias (intercept)
///
/// A prediction is made using:
///
/// `ŷ = sign(w^T x + b)`
///
/// # Loss Function
///
/// This implementation optimizes the perceptron loss:
///
/// `L = max(0, -y f(x))`
///
/// For each training sample:
///
/// - If `y * f(x) <= 0`, the parameters are updated.
/// - Otherwise, no update is performed.
///
/// # Notes
///
/// - Target labels must be encoded as `-1.0` or `1.0`.
/// - Convergence is guaranteed only if the dataset is linearly separable.
/// - When `batch_size = 1`, this reduces to the classical online perceptron.
/// - For `batch_size > 1`, this performs mini-batch subgradient descent.
///
/// # Example
///
/// ```ignore
/// let model = Perceptron::builder()
///     .epochs(200)
///     .batch_size(1)
///     .build();
/// ```
pub struct Perceptron<T: Optimizer> {
    /// Learned feature weights.
    ///
    /// Initialized during [`fit`].
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
    optimizer: T,

    classes: [usize;2]
}

/// Builder for configuring a [`Perceptron`] classifier.
///
/// Allows customization of:
/// - number of training epochs
/// - mini-batch size
/// - optimization algorithm
///
/// # Defaults
///
/// - `epochs = 100`
/// - `batch_size = 1` (pure online perceptron)
/// - `optimizer = SGD::new(0.01)`
pub struct Builder<T: Optimizer> {
    epochs: usize,
    batch_size: usize,
    optimizer: T,
}

impl Default for Builder<SGD> {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 1,
            optimizer: SGD::new(1.0),
        }
    }
}

impl Perceptron<SGD> {
    /// Creates a perceptron classifier using default hyperparameters.
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// Perceptron::builder().build()
    /// ```
    pub fn new() -> Self {
        Self::builder().build()
    }

    /// Returns a builder for configuring a [`Perceptron`] classifier.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = Perceptron::builder()
    ///     .epochs(500)
    ///     .batch_size(32)
    ///     .build();
    /// ```
    pub fn builder() -> Builder<SGD> {
        Builder::default()
    }
}

impl<T: Optimizer> Perceptron<T> {

    /// Returns a reference to the learned weight vector.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Returns the learned bias value.
    pub fn bias(&self) -> f64 {
        self.bias[0]
    }
    pub fn classes(&self) -> &[usize;2]{
        &self.classes
    }

    /// Predicts the class label for a single sample.
    ///
    /// Returns:
    /// - `1.0` if `w^T x + b >= 0`
    /// - `-1.0` otherwise
    pub fn predict(&self, x: &[f64]) -> usize {
        let fx = dot(x, &self.weights) + self.bias[0];
        if fx >= 0.0 { self.classes[1] } else { self.classes[0] }
        
    }

    /// Trains the perceptron classifier using mini-batch stochastic
    /// subgradient descent.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `features.n_rows() != target.len()`
    ///
    /// # Training Procedure
    ///
    /// For each epoch:
    ///
    /// 1. Shuffle sample indices.
    /// 2. Iterate over mini-batches.
    /// 3. For each sample:
    ///     - Compute `f(x) = w^T x + b`.
    ///     - If `y * f(x) <= 0`, accumulate gradients:
    ///         - `∂L/∂w = -y x`
    ///         - `∂L/∂b = -y`
    /// 4. Average gradients across the batch.
    /// 5. Update parameters using the optimizer.
    ///
    /// # Target Format
    ///
    /// Target labels must be encoded as:
    ///
    /// `-1.0` or `1.0`
    fn update_target(&mut self, target: &[usize]) -> Vec<f64> {
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target);

        assert_eq!(encoder.classes().len(), 2);

        self.classes = [
            encoder.classes()[0],
            encoder.classes()[1],
        ];

            encoded
                .into_iter()
                .map(|i| if i == 0 { -1.0 } else { 1.0 })
                .collect()
    }    

    pub fn fit(&mut self, features: &DenseMatrix, target: &[usize]) {
        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        assert_eq!(n_samples, target.len());


        // Initialize parameters
        self.weights = vec![0.0; n_features];
        let target = self.update_target(&target); 
    
        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let mut gradient = vec![0.0; n_features];
        let mut bias_gradient = [0.0];
        
        for _ in 0..self.epochs {
            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {
                gradient.fill(0.0);
                bias_gradient[0] = 0.0;

                for &idx in batch {
                    let row = features.row(idx);
                    let fx = dot(row, &self.weights) + self.bias[0];

                    if target[idx] * fx <= 0.0 {
                        for j in 0..n_features {
                            gradient[j] += -target[idx] * row[j];
                        }
                        bias_gradient[0] += -target[idx];
                    }
                }

                let inv_bs = 1.0 / batch.len() as f64;

                for g in &mut gradient {
                    *g *= inv_bs;
                }
                bias_gradient[0] *= inv_bs;

                self.optimizer.step(&mut self.weights, &gradient);
                self.optimizer.step(&mut self.bias, &bias_gradient);
            }
        }
    }
}

impl<T: Optimizer> Builder<T> {

    /// Sets the number of training epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Sets the mini-batch size.
    ///
    /// `batch_size = 1` corresponds to pure online perceptron.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets a custom optimizer.
    pub fn optimizer(mut self, optimizer: T) -> Builder<T> {
        self.optimizer = optimizer;
        self
    }

    /// Builds the untrained [`Perceptron`] model.
    ///
    /// Parameters are initialized during [`fit`].
    pub fn build(self) -> Perceptron<T> {
        Perceptron {
            weights: Vec::new(),
            bias: [0.0],
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0;2]
            
        }
    }
}
