#[derive(Debug, thiserror::Error)]
pub enum AnvilError {
    #[error("model not fitted — call fit() before predict()")]
    NotFitted,

    #[error("shape mismatch on {axis}: expected {expected}, got {got}")]
    ShapeMismatch {
        expected: usize,
        got: usize,
        axis: &'static str,
    },

    #[error("X and y have inconsistent number of samples: X={x_samples}, y={y_samples}")]
    DimensionMismatch {
        x_samples: usize,
        y_samples: usize,
    },

    #[error("invalid parameter '{param}': {reason}")]
    InvalidParam {
        param: &'static str,
        reason: String,
    },

    #[error("convergence failure: {algorithm} did not converge after {iters} iterations")]
    ConvergenceFailure {
        algorithm: &'static str,
        iters: usize,
    },

    #[error("empty dataset: {target}")]
    EmptyDataset {
        target: &'static str,
    },

    #[error("operation '{operation}' is not supported by this model")]
    NotSupported {
        operation: &'static str,
    },
}

