use crate::{
    preprocessing::encoder::LabelEncoder,
    core::{Estimator, Classifier, AnvilError,Transformer},
};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Zip};
use std::f64::consts::PI;

/// Gaussian Naive Bayes
pub struct GaussianNB {
    mean: Option<Array2<f64>>,
    log_gauss_const: Option<Array2<f64>>,
    inv_var: Option<Array2<f64>>,
    class_log_prior: Option<Vec<f64>>,
    class_prob: Option<Vec<f64>>,
    classes: Vec<usize>,
    var_smoothing: f64,
}

/// Builder
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
    pub fn probability(mut self, class_prob: Vec<f64>) -> Self {
        self.class_prob = Some(class_prob);
        self
    }

    pub fn var_smoothing(mut self, value: f64) -> Self {
        self.var_smoothing = value;
        self
    }

    pub fn build(self) -> GaussianNB {
        GaussianNB {
            mean: None,
            log_gauss_const: None,
            inv_var: None,
            class_log_prior: None,
            class_prob: self.class_prob,
            classes: Vec::new(),
            var_smoothing: self.var_smoothing,
        }
    }
}

impl GaussianNB {
    pub fn new() -> Self {
        Builder::default().build()
    }

    pub fn builder() -> Builder {
        Builder::default()
    }

    pub fn classes(&self) -> &Vec<usize> {
        &self.classes
    }
}

impl Estimator<usize> for GaussianNB {
    /// # Errors
    /// - DimensionMismatch
    /// - EmptyDataset
    /// - InvalidParam
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

        // accumulate
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

        // compute mean + variance
        let mut max_var = 0.0;

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
    /// # Errors
    /// - NotFitted
    /// - ShapeMismatch
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
