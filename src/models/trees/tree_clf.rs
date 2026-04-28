use ndarray::{ArrayView1};
use rand::Rng;
use crate::models::trees::{ClfSplitter,SplitContext,Node};


#[inline]
pub(super) fn entropy(counts: &[usize], total: usize) -> f64{

    if total == 0 {
        return 0.0;
    }

    let mut sum = 0.0;

    for &c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / total as f64;
        sum -= p * p.ln();
    }

    sum
}

#[inline]
pub(super) fn gini(counts: &[usize], total: usize) -> f64 {

    if total == 0 {
        return 0.0;
    }

    let mut sum = 1.0;

    for &c in counts {
        let p = c as f64 / total as f64;
        sum -= p * p;
    }

    sum
}

#[inline]
fn majority(target: &ArrayView1<usize>, indices: &[usize], n_classes: usize) -> usize {
    let mut counts = vec![0usize; n_classes];

    for &i in indices {
        counts[target[i]] += 1;
    }

    counts
        .iter()
        .enumerate()
        .max_by_key(|&(_, c)| c)
        .map(|(i, _)| i)
        .unwrap()
}

#[inline]
pub(crate) fn traverse(node: &Node<usize>, row: ArrayView1<f64>) -> usize {
    match node {
        Node::Leaf { value } => *value,
        Node::Internal {
            feature,
            threshold,
            left,
            right,
        } => {
            if row[*feature] <= *threshold {
                traverse(left, row)
                } else {
                traverse(right, row)
            }
        }
    }
}


pub(crate) fn build_tree_clf<S: ClfSplitter>(
    splitter: &S,
    criterion: fn(&[usize], usize) -> f64,
    ctx: &SplitContext<usize>,
    n_classes: usize,
    min_leaf: usize,
    max_depth: usize,
    min_sample_split: usize,
    rng: &mut impl Rng,
) -> Node<usize> {

    let n = ctx.indices.len();

    if n < 2 * min_leaf || n < min_sample_split {
        return Node::Leaf {
            value: majority(&ctx.y, ctx.indices, n_classes),
        };
    }

    let first = ctx.y[ctx.indices[0]];
    if ctx.indices.iter().all(|&i| ctx.y[i] == first) {
        return Node::Leaf { value: first };
    }

    if ctx.depth >= max_depth {
        return Node::Leaf {
            value: majority(&ctx.y, ctx.indices, n_classes),
        };
    }

    let Some(split) = splitter.best_split(ctx, criterion, n_classes, rng) else {
        return Node::Leaf {
            value: majority(&ctx.y, ctx.indices, n_classes),
        };
    };

    let mut left_idx = Vec::with_capacity(split.pos);
    let mut right_idx = Vec::with_capacity(n - split.pos);

    for &i in ctx.indices {
        if ctx.x[[i, split.feature]] <= split.threshold {
            left_idx.push(i);
        } else {
            right_idx.push(i);
        }
    } 

    let left_ctx = SplitContext {
        x: ctx.x,
        y: ctx.y,
        indices: &left_idx,
        depth: ctx.depth + 1,
    };

    let right_ctx = SplitContext {
        x: ctx.x,
        y: ctx.y,
        indices: &right_idx,
        depth: ctx.depth + 1,
    };

    let left_node = build_tree_clf(
        splitter,
        criterion,
        &left_ctx,
        n_classes,
        min_leaf,
        max_depth,
        min_sample_split,
        rng,
    );

    let right_node = build_tree_clf(
        splitter,
        criterion,
        &right_ctx,
        n_classes,
        min_leaf,
        max_depth,
        min_sample_split,
        rng,
    );

    Node::Internal {
        feature: split.feature,
        threshold: split.threshold,
        left: Box::new(left_node),
        right: Box::new(right_node),
    }
}       
