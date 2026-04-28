mod tree_reg;
mod extra_tree_clf;
mod impurity; 

pub use tree_reg::DecisionTreeRegressor;
pub use extra_tree_clf::ExtraTreeClassifier;

#[derive(Debug)]
enum Node<T>{
    Leaf {
        value: T,
    },
    Internal {
        feature: usize,
        threshold: f64,
        left: Box<Node<T>>,
        right: Box<Node<T>>,
    },
}

pub enum Criteria{
    Gini,
    Entropy,
    MeanSquared,
    MeanAbs,
}

pub enum Splitter{
    Best,
    Random
}

