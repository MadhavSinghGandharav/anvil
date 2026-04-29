
use rand::{Rng,RngExt, seq::SliceRandom};
use ndarray::{ArrayView1, ArrayView2, Array1};

use crate::{
    core::{Estimator, AnvilError, Classifier, Transformer},
    models::trees::*,
    models::trees::tree_clf::*,
    preprocessing::encoder::LabelEncoder,
};

/// =======================
/// MODEL
/// =======================
pub struct ExtraTreeClassifier {
    criterion: Criterion,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
    root: Option<Box<Node<usize>>>,
    classes: Vec<usize>,
}

/// =======================
/// BUILDER
/// =======================
pub struct Builder {
    criterion: Criterion,
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            criterion: Criterion::Gini,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_depth: None,
            max_features: None,
        }
    }
}

impl Builder {
    pub fn criterion(mut self, c: Criterion) -> Self {
        self.criterion = c;
        self
    }

    pub fn min_samples_split(mut self, v: usize) -> Self {
        self.min_samples_split = v;
        self
    }

    pub fn min_samples_leaf(mut self, v: usize) -> Self {
        self.min_samples_leaf = v;
        self
    }

    pub fn max_depth(mut self, v: usize) -> Self {
        self.max_depth = Some(v);
        self
    }

    pub fn max_features(mut self, v: usize) -> Self {
        self.max_features = Some(v);
        self
    }

    pub fn build(self) -> Result<ExtraTreeClassifier, AnvilError> {
        if self.min_samples_split < 2 {
            return Err(AnvilError::InvalidParam {
                param: "min_samples_split",
                reason: "must be >= 2".into(),
            });
        }

        if self.min_samples_leaf < 1 {
            return Err(AnvilError::InvalidParam {
                param: "min_samples_leaf",
                reason: "must be >= 1".into(),
            });
        }

        Ok(ExtraTreeClassifier {
            criterion: self.criterion,
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_depth: self.max_depth,
            max_features: self.max_features,
            root: None,
            classes: Vec::new(),
        })
    }
}

impl ExtraTreeClassifier {
    pub fn new() -> Result<Self, AnvilError> {
        Builder::default().build()
    }

    pub fn builder() -> Builder {
        Builder::default()
    }

    pub fn classes(&self) -> &[usize] {
        &self.classes
    }
}

/// =======================
/// FIT
/// =======================
impl Estimator<usize> for ExtraTreeClassifier {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<usize>) -> Result<(), AnvilError> {

        let n_samples = x.nrows();
        let n_features = x.ncols();

        let max_features = self.max_features.unwrap_or(n_features);

        if n_samples != y.len() {
            return Err(AnvilError::DimensionMismatch {
                x_samples: n_samples,
                y_samples: y.len(),
            });
        }

        if max_features > n_features {
            return Err(AnvilError::InvalidParam {
                param: "max_features",
                reason: "cannot exceed feature count".into(),
            });
        }

        // label encoding
        let mut encoder = LabelEncoder::new();
        let encoded = encoder.fit_transform(y)?;
        let n_classes = encoder.classes()?.len();
        self.classes = encoder.classes()?.to_vec();

        let impurity_fn: fn(&[usize], usize) -> f64 = match self.criterion {
            Criterion::Entropy => entropy,
            Criterion::Gini => gini,
        };

        let indices: Vec<usize> = (0..n_samples).collect();

        let ctx = SplitContext {
            x,
            y: encoded.view(),
            indices: &indices,
            depth: 0,
        };

        let mut rng = rand::rng();

        let root = build_tree_clf(
            self,
            impurity_fn,
            &ctx,
            n_classes,
            self.min_samples_leaf,
            self.max_depth.unwrap_or(usize::MAX),
            self.min_samples_split,
            &mut rng,
        );

        self.root = Some(Box::new(root));

        Ok(())
    }
}

/// =======================
/// SPLITTER
/// =======================
impl ClfSplitter for ExtraTreeClassifier {
    fn best_split(
        &self,
        ctx: &SplitContext<usize>,
        criterion: fn(&[usize], usize) -> f64,
        n_classes: usize,
        rng: &mut impl Rng,
    ) -> Option<SplitResult> {

        let n_node = ctx.indices.len();

        let mut parent_counts = vec![0usize; n_classes];
        for &i in ctx.indices {
            parent_counts[ctx.y[i]] += 1;
        }

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_pos = 0;
        let mut best_impurity = f64::INFINITY;

        let mut features: Vec<usize> = (0..ctx.x.ncols()).collect();

        if let Some(max_f) = self.max_features {
            features.shuffle(rng);
            features.truncate(max_f);
        }

        let mut left_counts = vec![0usize; n_classes];
        let mut right_counts = vec![0usize; n_classes];

        const EPS: f64 = 1e-12;

        for &feature in &features {

            // -------- 2-sample midpoint threshold --------
            let i1 = ctx.indices[rng.random_range(0..n_node)];
            let i2 = ctx.indices[rng.random_range(0..n_node)];

            let v1 = ctx.x[[i1, feature]];
            let v2 = ctx.x[[i2, feature]];

            if (v1 - v2).abs() < EPS {
                continue;
            }

            let thr = (v1 + v2) * 0.5;

            left_counts.fill(0);
            right_counts.copy_from_slice(&parent_counts);

            let mut left_n = 0;

            for &idx in ctx.indices {
                if ctx.x[[idx, feature]] <= thr {
                    let c = ctx.y[idx];
                    left_counts[c] += 1;
                    right_counts[c] -= 1;
                    left_n += 1;
                }
            }

            let right_n = n_node - left_n;

            if left_n < self.min_samples_leaf || right_n < self.min_samples_leaf {
                continue;
            }

            let left_imp = criterion(&left_counts, left_n);
            let right_imp = criterion(&right_counts, right_n);

            let score = left_n as f64 * left_imp
                + right_n as f64 * right_imp;

            if score < best_impurity {
                best_impurity = score;
                best_feature = feature;
                best_threshold = thr;
                best_pos = left_n;
            }
        }

        if best_impurity == f64::INFINITY {
            return None;
        }

        Some(SplitResult {
            feature: best_feature,
            threshold: best_threshold,
            pos: best_pos,
        })
    }
}

/// =======================
/// PREDICT
/// =======================
impl Classifier for ExtraTreeClassifier {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<usize>, AnvilError> {

        let root = self.root.as_ref().ok_or(AnvilError::NotFitted)?;
        let mut preds = Array1::zeros(x.nrows());

        for (i, row) in x.outer_iter().enumerate() {
            let enc = traverse(root, row);
            preds[i] = self.classes[enc];
        }

        Ok(preds)
    }
}
