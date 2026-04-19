use std::cmp::Ordering;
use rand::Rng;
use rand::seq::SliceRandom;
use ndarray::{Array1, ArrayView1, ArrayView2};
use crate::models::tree::Node;

/// Decision Tree regressor.
///
/// Builds a binary tree by recursively splitting the feature space
/// using the best split found at each node.
///
/// # Notes
///
/// - Uses MSE as the split criterion
/// - Uses presorted indices — O(n log n) once at `fit`, O(n) per node
pub struct DecisionTreeRegressor {

    /// Minimum number of samples required to split a node
    min_samples_split: usize,

    /// Minimum number of samples required in each leaf
    min_samples_leaf: usize,

    /// Maximum depth of the tree
    max_depth: Option<usize>,

    /// Max number of features to consider per split
    max_features: Option<usize>,

    /// Root node of the fitted tree
    root: Option<Box<Node<f64>>>,
}

/// Builder for configuring [`DecisionTreeRegressor`]
pub struct Builder {
    min_samples_split: usize,
    min_samples_leaf: usize,
    max_depth: Option<usize>,
    max_features: Option<usize>,
}

impl Default for Builder {
    /// Default configuration
    ///
    /// - min_samples_split = 2
    /// - min_samples_leaf = 1
    /// - max_depth = None
    /// - max_features = None (uses all features)
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

    /// Set minimum samples required to split a node
    pub fn min_samples_split(mut self, value: usize) -> Self {
        self.min_samples_split = value;
        self
    }

    /// Set minimum samples required in each leaf
    pub fn min_samples_leaf(mut self, value: usize) -> Self {
        self.min_samples_leaf = value;
        self
    }

    /// Set maximum tree depth
    pub fn max_depth(mut self, value: usize) -> Self {
        self.max_depth = Some(value);
        self
    }

    /// Set maximum number of features considered per split
    pub fn max_features(mut self, value: usize) -> Self {
        self.max_features = Some(value);
        self
    }

    /// Build model
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - min_samples_split < 2
    /// - min_samples_leaf < 1
    /// - max_features < 1
    pub fn build(self) -> DecisionTreeRegressor {

        assert!(
            self.min_samples_split >= 2,
            "min_samples_split must be >= 2"
        );

        assert!(
            self.min_samples_leaf >= 1,
            "min_samples_leaf must be >= 1"
        );

        if let Some(max_features) = self.max_features {
            assert!(
                max_features >= 1,
                "max_features must be >= 1"
            );
        }

        DecisionTreeRegressor {
            min_samples_split: self.min_samples_split,
            min_samples_leaf: self.min_samples_leaf,
            max_depth: self.max_depth,
            max_features: self.max_features,
            root: None,
        }
    }
}

/// Shared context passed to split finding functions
struct SplitContext<'a> {
    features:         ArrayView2<'a, f64>,
    target:           ArrayView1<'a, f64>,
    sorted_idx:       &'a [Vec<usize>],
    n_samples:        usize,
    min_samples_leaf: usize,
}

/// Result of a split search
struct SplitResult {
    least_error:    f64,
    best_feature:   usize,
    best_threshold: f64,
    best_split_pos: usize,
}

/// Computes mean of target values for the given indices
#[inline]
fn mean(indices: &[usize], target: ArrayView1<f64>) -> f64 {
    indices.iter().fold(0.0, |acc, &i| acc + target[i]) / indices.len() as f64
}

/// Evaluates all split candidates for a single feature
#[inline]
fn evaluate_feature(ctx: &SplitContext, f: usize) -> (f64, usize, f64) {

    let col = &ctx.sorted_idx[f];

    // initialize right sums from all target values
    let (mut r_sum, mut r_sumsq) = col.iter().fold((0.0, 0.0), |(s, sq), &i| {
        let v = ctx.target[i];
        (s + v, sq + v * v)
    });

    let (mut l_sum, mut l_sumsq) = (0.0_f64, 0.0_f64);
    let (mut left_size, mut right_size) = (0.0_f64, ctx.n_samples as f64);

    let mut least_error    = f64::INFINITY;
    let mut best_pos       = 0;
    let mut best_threshold = 0.0;

    for j in 0..col.len() - 1 {

        let curr = col[j];
        let next = col[j + 1];

        let v    = ctx.target[curr];
        let v_sq = v * v;

        left_size  += 1.0;
        right_size -= 1.0;

        l_sum   += v;
        l_sumsq += v_sq;
        r_sum   -= v;
        r_sumsq -= v_sq;

        if (left_size  as usize) < ctx.min_samples_leaf
        || (right_size as usize) < ctx.min_samples_leaf {
            continue;
        }

        // skip if same feature value — not a valid split boundary
        if ctx.features[[curr, f]] == ctx.features[[next, f]] {
            continue;
        }

        let l_mean = l_sum / left_size;
        let r_mean = r_sum / right_size;

        let l_mse = l_sumsq / left_size  - l_mean * l_mean;
        let r_mse = r_sumsq / right_size - r_mean * r_mean;

        // weighted MSE
        let error = (l_mse * left_size + r_mse * right_size) / ctx.n_samples as f64;

        if error < least_error {
            least_error    = error;
            best_pos       = j;
            best_threshold = (ctx.features[[curr, f]] + ctx.features[[next, f]]) / 2.0;
        }
    }

    (least_error, best_pos, best_threshold)
}

/// Finds the best split across a randomly shuffled subset of features.
///
/// When `max_features == n_features`, all features are evaluated in random order.
/// When `max_features < n_features`, only a random subset is evaluated.
fn find_best_split(ctx: &SplitContext, max_features: usize, rng: &mut impl Rng) -> SplitResult {

    let mut order: Vec<usize> = (0..ctx.sorted_idx.len()).collect();
    order.shuffle(rng);

    let mut least_error    = f64::INFINITY;
    let mut best_feature   = 0;
    let mut best_threshold = 0.0;
    let mut best_split_pos = 0;

    for &f in order.iter().take(max_features) {
        let (error, pos, threshold) = evaluate_feature(ctx, f);
        if error < least_error {
            least_error    = error;
            best_feature   = f;
            best_split_pos = pos;
            best_threshold = threshold;
        }
    }

    SplitResult { least_error, best_feature, best_threshold, best_split_pos }
}

/// Traverses the tree for a single sample and returns the predicted value
#[inline]
fn traverse(node: &Node<f64>, row: ArrayView1<f64>) -> f64 {
    match node {
        Node::Leaf { value } => *value,
        Node::Internal { feature, threshold, left, right } => {
            if row[*feature] <= *threshold {
                traverse(left, row)
            } else {
                traverse(right, row)
            }
        }
    }
}

impl DecisionTreeRegressor {

    /// Create model with default configuration
    pub fn new() -> Self {
        Builder::default().build()
    }

    /// Returns builder
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Fits the decision tree regressor
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - features and target size mismatch
    /// - dataset contains zero samples
    /// - max_features > total features
    pub fn fit(&mut self, features: ArrayView2<f64>, target: ArrayView1<f64>) {

        let n_samples  = features.nrows();
        let n_features = features.ncols();

        let max_features = self.max_features.unwrap_or(n_features);

        assert!(
            n_samples == target.len(),
            "Number of samples mismatch"
        );

        assert!(
            n_samples > 0,
            "Cannot fit with zero samples"
        );

        assert!(
            max_features <= n_features,
            "max_features cannot be > total features"
        );

        // precompute sorted indices per feature — O(n log n) once
        let sorted_idx: Vec<Vec<usize>> = (0..n_features)
            .map(|f| {
                let mut col: Vec<usize> = (0..n_samples).collect();
                col.sort_unstable_by(|&a, &b| {
                    features[[a, f]]
                        .partial_cmp(&features[[b, f]])
                        .unwrap_or(Ordering::Equal)
                });
                col
            })
            .collect();

        let mut rng = rand::rng();

        self.root = Some(Box::new(self.build_tree(
            sorted_idx,
            0,
            features,
            target,
            max_features,
            &mut rng,
        )));
    }

    /// Predict target values
    ///
    /// # Panics
    ///
    /// Panics if model not fitted
    pub fn predict(&self, features: ArrayView2<f64>) -> Array1<f64> {

        let root = self.root.as_ref().expect("Model not fitted");

        let mut preds = Array1::zeros(features.nrows());

        for (i, row) in features.outer_iter().enumerate() {
            preds[i] = traverse(root, row);
        }

        preds
    }

    fn build_tree(
        &self,
        sorted_idx: Vec<Vec<usize>>,
        depth: usize,
        features: ArrayView2<f64>,
        target: ArrayView1<f64>,
        max_features: usize,
        rng: &mut impl Rng,
    ) -> Node<f64> {

        let n_samples = sorted_idx[0].len();
        let first_col = &sorted_idx[0];

        // 1. small node — return mean of target values
        if n_samples < self.min_samples_split {
            return Node::Leaf {
                value: mean(first_col, target),
            };
        }

        // 2. pure node — all target values identical
        let first = target[first_col[0]];
        if first_col.iter().all(|&i| (target[i] - first).abs() < 1e-10) {
            return Node::Leaf { value: first };
        }

        // 3. max depth
        if let Some(max_d) = self.max_depth {
            if depth >= max_d {
                return Node::Leaf {
                    value: mean(first_col, target),
                };
            }
        }

        let ctx = SplitContext {
            features,
            target,
            sorted_idx:       &sorted_idx,
            n_samples,
            min_samples_leaf: self.min_samples_leaf,
        };

        // 4. find best split
        let result = find_best_split(&ctx, max_features, rng);

        // 5. no valid split found
        if result.least_error == f64::INFINITY {
            return Node::Leaf {
                value: mean(first_col, target),
            };
        }

        // 6. partition — mark left indices using best split
        let mut mark = vec![false; features.nrows()];
        for &idx in &sorted_idx[result.best_feature][..=result.best_split_pos] {
            mark[idx] = true;
        }

        let left_cap   = result.best_split_pos + 1;
        let right_cap  = n_samples - left_cap;
        let n_features = sorted_idx.len();

        let mut left_sorted: Vec<Vec<usize>> = (0..n_features)
            .map(|_| Vec::with_capacity(left_cap))
            .collect();

        let mut right_sorted: Vec<Vec<usize>> = (0..n_features)
            .map(|_| Vec::with_capacity(right_cap))
            .collect();

        for (f, col) in sorted_idx.iter().enumerate() {
            for &idx in col {
                if mark[idx] {
                    left_sorted[f].push(idx);
                } else {
                    right_sorted[f].push(idx);
                }
            }
        }

        // reset mark — avoid reallocation
        for &idx in &sorted_idx[result.best_feature][..=result.best_split_pos] {
            mark[idx] = false;
        }

        // 7. recurse
        let left = self.build_tree(
            left_sorted,
            depth + 1,
            features,
            target,
            max_features,
            rng,
        );

        let right = self.build_tree(
            right_sorted,
            depth + 1,
            features,
            target,
            max_features,
            rng,
        );

        Node::Internal {
            feature:   result.best_feature,
            threshold: result.best_threshold,
            left:      Box::new(left),
            right:     Box::new(right),
        }
    }
} 
