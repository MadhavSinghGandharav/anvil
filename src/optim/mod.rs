mod sgd;

/// Re-export of the default Stochastic Gradient Descent optimizer.
///
/// This allows users to import `SGD` directly from the
/// `optim` module:
///
/// ```ignore
/// use crate::optim::SGD;
/// ```
pub use sgd::SGD;

/// Trait defining the interface for optimization algorithms.
///
/// An `Optimizer` is responsible for updating model parameters
/// in-place using their corresponding gradients.
///
/// # Contract
///
/// Implementations must:
/// - Update `weights` in-place.
/// - Assume `weights.len() == gradient.len()`.
///
/// Dimension validation is expected to be handled by the caller
/// (typically the model).
///
/// # Design Notes
///
/// - The trait uses `&mut self` to allow optimizers to store
///   internal state (e.g., momentum buffers, Adam moments).
/// - The interface operates on slices to remain independent of
///   specific container types.
/// - No allocation should occur inside `step`.
///
/// # Example
///
/// ```ignore
/// impl Optimizer for MyOptimizer {
///     fn step(&mut self, weights: &mut [f64], gradient: &[f64]) {
///         for i in 0..weights.len() {
///             weights[i] -= 0.01 * gradient[i];
///         }
///     }
/// }
/// ```
pub trait Optimizer {
    /// Performs a single optimization step.
    ///
    /// Applies an update rule to `weights` using `gradient`.
    ///
    /// # Panics
    ///
    /// Behavior is undefined if:
    /// - `weights.len() != gradient.len()`
    ///
    /// Implementations may assume matching lengths for performance reasons.
    ///
    /// # Complexity
    ///
    /// O(n), where `n` is the number of parameters.
    fn step(&mut self, weights: &mut [f64], gradient: &[f64]);
}
