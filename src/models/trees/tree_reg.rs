use ndarray::{ArrayView1};
use rand::Rng;
use crate::models::trees::{RegSplitter,SplitContext,Node};




#[inline]
pub(crate) fn traverse(node: &Node<f64>, row: ArrayView1<f64>) -> f64 {
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
 
#[inline]
fn mean(target: &ArrayView1<f64>, indices: &[usize]) -> f64 {
    indices.iter().fold(0.0, |acc, &i| acc + target[i]) / indices.len() as f64
}

pub(crate) fn build_tree_reg<S: RegSplitter>(
    splitter: &S,
    ctx: &SplitContext<f64>,
    min_leaf: usize,
    max_depth: usize,
    min_sample_split: usize,
    rng: &mut impl Rng,
) -> Node<f64> {

    let n = ctx.indices.len();

    if n < 2 * min_leaf || n < min_sample_split {
        return Node::Leaf {
            value: mean(&ctx.y, ctx.indices)
        };
    }

    let first = ctx.y[ctx.indices[0]];
    if ctx.indices.iter().all(|&i| ctx.y[i] == first) {
        return Node::Leaf { value: first };
    }

    if ctx.depth >= max_depth {
        return Node::Leaf {
            value: mean(&ctx.y, ctx.indices),
        };
    }

    let Some(split) = splitter.best_split(ctx, rng) else {
        return Node::Leaf {
            value: mean(&ctx.y, ctx.indices),
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

    let left_node = build_tree_reg(
        splitter,
        &left_ctx,
        min_leaf,
        max_depth,
        min_sample_split,
        rng,
    );

    let right_node = build_tree_reg(
        splitter,
        &right_ctx,
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
