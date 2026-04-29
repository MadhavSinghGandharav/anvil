use ndarray::{ArrayView1, ArrayView2};
use rand::Rng;

mod tree_clf;
mod tree_reg;
mod decision_tree_clf;
mod decision_tree_reg;
mod extra_tree_clf;

pub use decision_tree_clf::DecisionTreeClassifier;
pub use decision_tree_reg::DecisionTreeRegressor;
pub use extra_tree_clf::ExtraTreeClassifier;


pub enum Criterion{
    Entropy,
    Gini
}

/// =======================
/// NODE
/// =======================
pub(crate) enum Node<T> {
    Leaf { value: T },
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
}
/// =======================
/// CONTEXT
/// =======================
pub(crate) struct SplitContext<'a, T> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, T>,
    pub indices: &'a [usize],
    pub depth: usize,
}

pub(crate) struct SplitResult {
    pub feature: usize,
    pub threshold: f64,
    pub pos: usize,
}

/// =======================
/// SPLITTER TRAITS
/// =======================

/// Classification splitter
pub(crate) trait ClfSplitter {
    fn best_split(
        &self,
        ctx: &SplitContext<usize>,
        criterion: fn(&[usize], usize) -> f64,
        n_classes: usize,
        rng: &mut impl Rng
    ) -> Option<SplitResult>;
}

/// Regression splitter (only MSE will be used internally)
pub(crate) trait RegSplitter {
    fn best_split(
        &self,
        ctx: &SplitContext<f64>,
        rng: &mut impl Rng
    ) -> Option<SplitResult>;
}
