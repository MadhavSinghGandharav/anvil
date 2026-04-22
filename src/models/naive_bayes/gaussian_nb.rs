//! Gaussian Naive Bayes module.
//!
//! This module implements the Gaussian Naive Bayes algorithm for classification.
//! It assumes that the continuous values associated with each class are distributed 
//! according to a Gaussian (Normal) distribution.

use crate::{
    preprocessing::encoder::LabelEncoder,
    core::{Estimator, Classifier, AnvilError,Transformer},
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use std::f64::consts::PI;

/// Gaussian Naive Bayes (GaussianNB) classifier.
///
/// This model calculates the likelihood of the features based on a Gaussian 
/// distribution defined by the mean and variance of each feature per class.
///
/// # Examples
///
/// ```
/// use anvil::models::GaussianNB;
///
/// let model = GaussianNB::builder()
///     .var_smoothing(1e-10)
///     .build()
///     .unwrap();
/// ```
pub struct GaussianNB {
    mean: Option<Array2<f64>>,
    log_gauss_const: Option<Array2<f64>>,
    inv_var: Option<Array2<f64>>,
    class_log_prior: Option<Vec<f64>>,
    class_prob: Option<Vec<f64>>,
    classes: Vec<usize>,
    var_smoothing: f64,
}

/// A builder for configuring and creating a [`GaussianNB`] instance.
pub struct Builder {
    class_prob: Option<Vec<f64>>,
    var_smoothing: f64,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            class_prob: None,
            var_smoothing: 1e-9,
        }
    }
}

impl Builder {
    /// Sets user-defined prior probabilities for the classes.
    ///
    /// If not provided, priors are adjusted according to the data.
    pub fn probability(mut self, class_prob: Vec<f64>) -> Self {
        self.class_prob = Some(class_prob);
        self
    }

    /// Sets the portion of the largest variance of all features that is added 
    /// to variances for calculation stability.
    pub fn var_smoothing(mut self, value: f64) -> Self {
        self.var_smoothing = value;
        self
    }

    /// Consumes the builder and returns a [`GaussianNB`] instance.
    ///
    /// # Errors
    ///
    /// Returns `AnvilError::InvalidParam` if `var_smoothing` is non-positive or 
    /// if `class_prob` sums to zero or contains invalid values.
    pub fn build(self) -> Result<GaussianNB, AnvilError> {

        if self.var_smoothing <= 0.0 {
            return Err(AnvilError::InvalidParam {
                param: "var_smoothing",
                reason: "must be > 0".into(),
            });
        }

        if let Some(ref probs) = self.class_prob {

            if probs.is_empty() {
                return Err(AnvilError::InvalidParam {
                    param: "class_prob",
                    reason: "cannot be empty".into(),
                });
            }

            let mut sum = 0.0;

            for &p in probs {
                if !p.is_finite() || p < 0.0 {
                    return Err(AnvilError::InvalidParam {
                        param: "class_prob",
                        reason: "must contain finite, non-negative values".into(),
                    });
                }
                sum += p;
            }

            if sum <= 0.0 {
                return Err(AnvilError::InvalidParam {
                    param: "class_prob",
                    reason: "sum must be > 0".into(),
                });
            }
        }

        Ok(GaussianNB {
            mean: None,
            log_gauss_const: None,
            inv_var: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
            var_smoothing: self.var_smoothing,
        })
    }
}

impl GaussianNB {
    /// Returns a new [`GaussianNB`] with default parameters.
    pub fn new() -> Result<Self, AnvilError> {
        Builder::default().build()
    }

    /// Returns a [`Builder`] to configure the model.
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Returns the unique class labels found during training.
    pub fn classes(&self) -> &Vec<usize> {
        &self.classes
    }
}

impl Estimator<usize> for GaussianNB {
    /// Fits the Gaussian Naive Bayes model according to the given training data.
    ///
    /// # Errors
    ///
    /// * `AnvilError::DimensionMismatch`: If `x` and `y` have different sample counts.
    /// * `AnvilError::EmptyDataset`: If the input `x` contains no samples.
    /// * `AnvilError::InvalidParam`: If the number of provided priors does not match the class count.
    fn fit(
        &mut self,
        x: ArrayView2<f64>,
        y: ArrayView1<usize>,
    ) -> Result<(), AnvilError> {

        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: y.len(),
            });
        }

        if n_samples == 0 {
            return Err(AnvilError::EmptyDataset {
                target: "X",
            });
        }

        let mut encoder = LabelEncoder::new();
        let target = encoder.fit_transform(y)?;

        self.classes = encoder.classes()?.to_vec();
        let n_classes = self.classes.len();

        if let Some(priors) = &self.class_prob {
            if priors.len() != n_classes {
                return Err(AnvilError::InvalidParam {
                    param: "class_prob",
                    reason: "length must match number of classes".into(),
                });
            }
        }

        let mut sum    = Array2::<f64>::zeros((n_classes, n_features));
        let mut sum_sq = Array2::<f64>::zeros((n_classes, n_features));
        let mut count  = vec![0usize; n_classes];

        // Accumulate sums and squared sums for mean and variance calculation
        for (i, &c) in target.iter().enumerate() {
            count[c] += 1;
            let row = x.row(i);

            Zip::from(sum.row_mut(c))
                .and(sum_sq.row_mut(c))
                .and(&row)
                .for_each(|s, sq, &val| {
                    *s  += val;
                    *sq += val * val;
                });
        }

        let mut max_var = 0.0;

        // Compute mean and variance per feature per class
        for c in 0..n_classes {
            let n = count[c] as f64;

            Zip::from(sum.row_mut(c))
                .and(sum_sq.row_mut(c))
                .for_each(|s, sq| {
                    let mean = *s / n;
                    let var  = *sq / n - mean * mean;

                    *s = mean;
                    *sq = var;

                    if var > max_var {
                        max_var = var;
                    }
                });
        }

        // Apply variance smoothing
        let eps = (self.var_smoothing * max_var).max(self.var_smoothing);

        let mut log_const = Array2::<f64>::zeros((n_classes, n_features));
        let mut inv_var   = Array2::<f64>::zeros((n_classes, n_features));

        Zip::from(&sum_sq)
            .and(&mut log_const)
            .and(&mut inv_var)
            .for_each(|&var, lc, iv| {
                let v = var + eps;
                *lc = -0.5 * (2.0 * PI * v).ln();
                *iv = 1.0 / (2.0 * v);
            });

        let total = target.len() as f64;

        let log_priors = if let Some(ref p) = self.class_prob {
            p.iter().map(|p| p.ln()).collect()
        } else {
            count.iter().map(|&c| (c as f64 / total).ln()).collect()
        };

        self.mean = Some(sum);
        self.log_gauss_const = Some(log_const);
        self.inv_var = Some(inv_var);
        self.class_log_prior = Some(log_priors);

        Ok(())
    }
}

impl Classifier for GaussianNB {
    /// Performs classification on an array of test vectors `x`.
    ///
    /// # Errors
    ///
    /// * `AnvilError::NotFitted`: If the model has not been trained yet.
    /// * `AnvilError::ShapeMismatch`: If the feature count of `x` does not match the fitted data.
    fn predict(
        &self,
        x: ArrayView2<f64>,
    ) -> Result<Array1<usize>, AnvilError> {

        let mean = self.mean.as_ref().ok_or(AnvilError::NotFitted)?;
        let log_const = self.log_gauss_const.as_ref().ok_or(AnvilError::NotFitted)?;
        let inv_var = self.inv_var.as_ref().ok_or(AnvilError::NotFitted)?;
        let log_prior = self.class_log_prior.as_ref().ok_or(AnvilError::NotFitted)?;

        if x.ncols() != mean.ncols() {
            return Err(AnvilError::ShapeMismatch {
                expected: mean.ncols(),
                got: x.ncols(),
                axis: "features",
            });
        }

        let n_classes = mean.nrows();
        let mut preds = Array1::zeros(x.nrows());

        for (i, row) in x.outer_iter().enumerate() {

            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;

            for c in 0..n_classes {
                // Log-posterior ∝ log(prior) + Σ log(likelihood)
                let mut score = log_prior[c];

                Zip::from(&row)
                    .and(mean.row(c))
                    .and(log_const.row(c))
                    .and(inv_var.row(c))
                    .for_each(|&val, &m, &lc, &iv| {
                        score += lc - (val - m) * (val - m) * iv;
                    });

                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            preds[i] = self.classes[best_class];
        }

        Ok(preds)
    }
}
