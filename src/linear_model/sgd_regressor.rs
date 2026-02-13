/// Stochastic Gradient Descent regressor.
///
/// `SGDRegressor` implements linear regression using
/// stochastic gradient descent optimization.
///
/// The model learns weights and bias during [`fit`].
/// Hyperparameters such as learning rate and number of epochs
/// can be configured using the builder pattern.
///
/// # Example
///
/// ```ignore
/// let model = SGDRegressor::builder()
///     .learning_rate(0.01)
///     .epochs(200)
///     .build();
/// ```
pub struct SGDRegressor {
    /// Learned feature weights.
    ///
    /// Initialized during `fit`.
    weights: Vec<f64>,

    /// Learned bias (intercept).
    ///
    /// Initialized during `fit`.
    bias: f64,

    /// Number of training epochs.
    epochs: usize,

    /// Learning rate used during optimization.
    learning_rate: f64,
}

/// Builder for [`SGDRegressor`].
///
/// Provides a configurable way to construct an
/// `SGDRegressor` with custom hyperparameters.
///
/// Defaults:
/// - `epochs = 100`
/// - `learning_rate = 0.1`
pub struct Builder {
    epochs: usize,
    learning_rate: f64,
}

impl Default for Builder {
    /// Creates a builder with default hyperparameters.
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.1,
        }
    }
}

impl SGDRegressor {
    /// Creates an `SGDRegressor` with default hyperparameters.
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
    ///     .learning_rate(0.01)
    ///     .epochs(500)
    ///     .build();
    /// ```
    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl Builder {
    /// Sets the learning rate.
    ///
    /// # Panics
    ///
    /// You should later validate that learning rate is positive
    /// inside `build()` in production code.
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the number of training epochs.
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Builds the `SGDRegressor`.
    ///
    /// The model is returned in an untrained state.
    /// Weights and bias are initialized during `fit`.
    pub fn build(self) -> SGDRegressor {
        SGDRegressor {
            weights: Vec::new(),
            bias: 0.0,
            epochs: self.epochs,
            learning_rate: self.learning_rate,
        }
    }
}
