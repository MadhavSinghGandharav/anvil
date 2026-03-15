
use crate::{
    optim::{Optimizer, SGD},
    preprocessing::LabelEncoder,
};
use ndarray::{Array1, ArrayView1, ArrayView2, s};
use rand::seq::SliceRandom;

/// Linear binary classifier trained using the **Perceptron algorithm**.
///
/// The perceptron learns a **linear decision boundary** of the form:
///
/// f(x) = wᵀx + b
///
/// A prediction is made using the sign of the decision function.
///
/// # Training Rule
///
/// For each sample `(x, y)` where `y ∈ {-1, 1}`:
///
/// if y * (wᵀx + b) ≤ 0:
///     w ← w + η y x
///     b ← b + η y
///
/// where `η` is the learning rate.
///
/// # Notes
///
/// - Supports **binary classification only**
/// - Target labels are internally converted to `{-1,1}`
/// - Uses mini-batch stochastic subgradient descent
pub struct Perceptron {

    /// Model parameters `[bias, weights...]`
    params: Option<Array1<f64>>,

    /// Number of training epochs
    epochs: usize,

    /// Mini-batch size
    batch_size: usize,

    /// Optimizer used for parameter updates
    optimizer: Box<dyn Optimizer>,

    /// Original class labels
    classes: [usize; 2],
}

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

impl Builder {

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be greater than 0");
        self.batch_size = batch_size;
        self
    }

    pub fn optimizer<O: Optimizer + 'static>(mut self, optimizer: O) -> Self {
        self.optimizer = Box::new(optimizer);
        self
    }

    pub fn build(self) -> Perceptron {
        Perceptron {
            params: None,
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.optimizer,
            classes: [0; 2],
        }
    }
}

impl Perceptron {

    pub fn new() -> Self {
        Self::builder().build()
    }

    pub fn builder() -> Builder {
        Builder::default()
    }

    pub fn weights(&self) -> ArrayView1<'_, f64> {
        let params = self.params.as_ref().expect("Model not fitted");
        params.slice(s![1..])
    }

    pub fn bias(&self) -> f64 {
        self.params.as_ref().expect("Model not fitted")[0]
    }

    pub fn classes(&self) -> &[usize; 2] {
        &self.classes
    }

    /// Convert labels to {-1,1}
    fn update_target(&mut self, target: &[usize]) -> Vec<f64> {

        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(target);

        assert_eq!(
            encoder.classes().len(),
            2,
            "Perceptron supports only binary classification"
        );

        self.classes = [encoder.classes()[0], encoder.classes()[1]];

        encoded
            .into_iter()
            .map(|i| if i == 0 { -1.0 } else { 1.0 })
            .collect()
    }

    pub fn fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<usize>) {

        let n_samples = features.nrows();
        let n_features = features.ncols();

        assert_eq!(
            n_samples,
            target.len(),
            "Number of samples and target values must match"
        );

        let mut params = Array1::<f64>::zeros(n_features + 1);

        let mut rng = rand::rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        let target = self.update_target(target.as_slice().unwrap());

        let mut gradient = Array1::<f64>::zeros(n_features + 1);

        for _ in 0..self.epochs {

            indices.shuffle(&mut rng);

            for batch in indices.chunks(self.batch_size) {

                gradient.fill(0.0);

                let weights = params.slice(s![1..]);
                let bias = params[0];
                
                let mut grad_w = gradient.slice_mut(s![1..]);
                let mut grad_b = 0.0;

                for &idx in batch {

                    let row = features.row(idx);

                    let y = target[idx];
                    let y_pred = row.dot(&weights) + bias;

                    if y * y_pred <= 0.0 {


                        for (g, &x) in grad_w.iter_mut().zip(row.iter()) {
                            *g -= y * x;
                        }

                        grad_b -= y;
                    }
                }
                gradient[0] = grad_b;

                let inv_bs = 1.0 / batch.len() as f64;
                gradient *= inv_bs;

                self.optimizer
                    .step(params.as_slice_mut().unwrap(), gradient.as_slice().unwrap());
            }
        }

        self.params = Some(params);
    }

    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<usize> {

        let params = self.params.as_ref().expect("Model not fitted");

        let weights = params.slice(s![1..]);
        let bias = params[0];

        assert!(
            weights.len() == features.ncols(),
            "Feature dimension mismatch"
        );

        let mut preds = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {

            let fx = row.dot(&weights) + bias;

            if fx >= 0.0 {
                preds[i] = self.classes[1];
            } else {
                preds[i] = self.classes[0];
            }
        }

        preds
    }
}

