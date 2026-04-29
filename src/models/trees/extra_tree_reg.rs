use rand::{Rng,RngExt};
use ndarray::{ArrayView1, ArrayView2, Array1};

use crate::{
    core::{Estimator, AnvilError, Regressor},
    models::trees::{RegSplitter, Node, SplitContext, SplitResult},
    models::trees::tree_reg::*,
};

pub struct ExtraTreeRegressor {
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
    root: Option<Box<Node<f64>>>,
}

pub struct Builder {
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
}

impl Default for Builder {
    fn default() -> Self {
        Self {
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_depth: None,
            max_features: None,
        }
    }
}

impl Builder {
    pub fn build(self) -> Result<ExtraTreeRegressor, AnvilError> {
        Ok(ExtraTreeRegressor {
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_depth: self.max_depth,
            max_features: self.max_features,
            root: None,
        })
    }
}

impl ExtraTreeRegressor {
    pub fn new() -> Result<Self, AnvilError> {
        Builder::default().build()
    }

    pub fn builder() -> Builder {
        Builder::default()
    }
}

impl Estimator<f64> for ExtraTreeRegressor {
    fn fit(&mut self, x: ArrayView2<f64>, y: ArrayView1<f64>) -> Result<(), AnvilError> {

        let n_samples = x.nrows();

        let indices: Vec<usize> = (0..n_samples).collect();

        let ctx = SplitContext {
            x,
            y,
            indices: &indices,
            depth: 0,
        };

        let mut rng = rand::rng();

        let root = build_tree_reg(
            self,
            &ctx,
            self.min_samples_leaf,
            self.max_depth.unwrap_or(usize::MAX),
            self.min_samples_split,
            &mut rng,
        );

        self.root = Some(Box::new(root));
        Ok(())
    }
}

use rand::seq::SliceRandom;

impl RegSplitter for ExtraTreeRegressor {
    fn best_split(
        &self,
        ctx: &SplitContext<f64>,
        rng: &mut impl Rng,
    ) -> Option<SplitResult> {

        let n_node = ctx.indices.len();

        // initial right sums
        let (r_sum_init, r_sumsq_init) =
            ctx.indices.iter().fold((0.0, 0.0), |(s, sq), &i| {
                let v = ctx.y[i];
                (s + v, sq + v * v)
            });

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_pos = 0;
        let mut best_impurity = f64::INFINITY;

        let mut features: Vec<usize> = (0..ctx.x.ncols()).collect();

        if let Some(max_f) = self.max_features {
            features.shuffle(rng);
            features.truncate(max_f);
        }

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

            let mut left_sum = 0.0;
            let mut left_sumsq = 0.0;
            let mut right_sum = r_sum_init;
            let mut right_sumsq = r_sumsq_init;

            let mut left_n = 0usize;

            for &idx in ctx.indices {
                if ctx.x[[idx, feature]] <= thr {
                    let v = ctx.y[idx];
                    let sumsq = v * v;
                    left_sum += v;
                    left_sumsq += sumsq;
                    right_sum -= v;
                    right_sumsq -= sumsq;
                    left_n += 1;
                }
            }

            let right_n = n_node - left_n;

            if left_n < self.min_samples_leaf || right_n < self.min_samples_leaf {
                continue;
            }
            let sse =
                (left_sumsq - (left_sum * left_sum) / left_n as f64) +
                (right_sumsq - (right_sum * right_sum) / right_n as f64);

            if sse < best_impurity {
                best_impurity = sse;
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

impl Regressor for ExtraTreeRegressor {
    fn predict(&self, x: ArrayView2<f64>) -> Result<Array1<f64>, AnvilError> {
        let root = self.root.as_ref().ok_or(AnvilError::NotFitted)?;

        let mut preds = Array1::zeros(x.nrows());

        for (i, row) in x.outer_iter().enumerate() {
            preds[i] = traverse(root, row);
        }

        Ok(preds)
    }
}
