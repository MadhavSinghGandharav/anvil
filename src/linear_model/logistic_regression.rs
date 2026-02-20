use crate::{
    core::{DenseMatrix, utils::dot},
    optim::{Optimizer, SGD},
    preprocessing::LabelEncoder,
};

use rand::seq::SliceRandom;

/// Logistic Regression classifier for binary classification.
///
/// The model learns a linear decision function:
///
///     f(x) = w^T x + b
///
/// and optimizes the **logistic loss**:
///
///     L = log(1 + exp(-y f(x)))
///
/// where:
/// - `y ∈ {-1, 1}`
/// - `w` is the weight vector
/// - `b` is the bias
///
/// # Optimization
///
/// Training is performed using mini-batch gradient descent with a
/// user-specified optimizer.
///
/// The gradient of the loss for a single sample is:
///
///     ∂L/∂w = -y x / (1 + exp(y f(x)))
///     ∂L/∂b = -y     / (1 + exp(y f(x)))
///
/// # Notes
///
/// - Target labels are automatically encoded internally.
/// - Only binary classification is supported.
/// - Convergence is guaranteed because logistic loss is convex.
/// - When `batch_size = 1`, this reduces to stochastic gradient descent.
///
/// # Example
///
/// ```ignore
/// let model = LogisticRegression::builder()
///     .epochs(200)
///     .batch_size(32)
///     .build();
/// ```
pub struct LogisticRegression {
    weights: Vec<f64>,
    bias: [f64; 1],
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
    classes: [usize; 2],
}

/// Builder for configuring a [`LogisticRegression`] classifier.
///
/// # Defaults
///
/// - `epochs = 100`
/// - `batch_size = 1`
/// - `optimizer = SGD::new(1.0)`
pub struct Builder {
    epochs: usize,
    batch_size: usize,
    optimizer: Box<dyn Optimizer>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 1,
            optimizer: Box::new(SGD::new(1.0)),
        }
    }
}

#[inline]
 fn sigmoid(z: f64) -> f64 {
        if z >= 0.0 {
            let exp_neg = (-z).exp();
            1.0 / (1.0 + exp_neg)
        } else {
            let exp_pos = z.exp();
            exp_pos / (1.0 + exp_pos)
        }
    }

impl LogisticRegression {
    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl LogisticRegression {
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn bias(&self) -> f64 {
        self.bias[0]
    }

    pub fn classes(&self) -> &[usize; 2] {
        &self.classes
    }

    /// Predict probability of the positive class.
    ///
    /// Returns:
    ///     σ(w^T x + b)
    ///
    /// where:
    ///     σ(z) = 1 / (1 + exp(-z))
    pub fn predict_proba(&self, x: &[f64]) -> f64 {
        let fx = dot(x, &self.weights) + self.bias[0];
        sigmoid(fx)
    }

    /// Predict class label.
    ///
    /// Returns:
    /// - `classes[1]` if probability ≥ 0.5
    /// - `classes[0]` otherwise
    pub fn predict(&self, x: &[f64]) -> usize {
        let prob = self.predict_proba(x);
        if prob >= 0.5 {
            self.classes[1]
        } else {
            self.classes[0]
        }
    }

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

    /// Train the logistic regression model.
    ///
    /// Performs mini-batch gradient descent.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `features.n_rows() != target.len()`
    pub fn fit(&mut self, features: &DenseMatrix, target: &[usize]) {
        let n_samples = features.n_rows();
        let n_features = features.n_cols();

        assert_eq!(n_samples, target.len());

        self.weights = vec![0.0; n_features];
        self.bias = [0.0];

        let target = self.update_target(target);

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

                    let yz = target[idx] * fx;

                    // σ(-yz) = 1 / (1 + exp(yz))
                    let sig = sigmoid(-yz);

                    for j in 0..n_features {
                        gradient[j] += -target[idx] * row[j] * sig;
                    }

                    bias_gradient[0] += -target[idx] * sig;
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

impl Builder {
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Builder {
        self.optimizer = Box::new(optimizer);
        self
    }

    pub fn build(self) -> LogisticRegression {
        LogisticRegression {
            weights: Vec::new(),
            bias: [0.0],
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0; 2],
        }
    }
}
