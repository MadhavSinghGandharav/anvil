use crate::optim::Optimizer;

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Updates parameters using the rule:
///
/// `w = w - lr * gradient`
///
/// where:
/// - `lr` is the learning rate
/// - `gradient` is the gradient of the loss with respect to parameters
///
/// # Notes
///
/// - This implementation performs in-place updates.
/// - No momentum or adaptive behavior is included.
/// - Suitable for simple linear models and baseline experiments.
pub struct SGD {
    /// Learning rate (step size).
    ///
    /// Controls how large each parameter update is.
    lr: f64,
}

impl SGD {
    /// Creates a new `SGD` optimizer with the given learning rate.
    ///
    /// # Parameters
    ///
    /// - `lr`: Learning rate (should typically be > 0).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let optimizer = SGD::new(0.01);
    /// ```
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
}

impl Optimizer for SGD {
    /// Performs a single parameter update step.
    ///
    /// Applies:
    ///
    /// `weights[i] -= lr * gradient[i]`
    ///
    /// for each parameter.
    ///
    /// # Panics
    ///
    /// This function assumes:
    /// - `weights.len() == gradient.len()`
    ///
    /// Behavior is undefined if lengths differ.
    ///
    /// # Complexity
    ///
    /// O(n), where `n` is the number of parameters.
    ///
    /// # Performance Notes
    ///
    /// - In-place update.
    /// - No allocations.
    /// - Fully vector-length dependent.

    fn step(&mut self, weights: &mut [f64], gradient: &[f64]) {

        for (w, g) in weights.iter_mut().zip(gradient) {
            *w -= self.lr * g;
        }
    }
}
