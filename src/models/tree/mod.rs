mod tree_clf;
mod tree_reg;
mod impurity; 

pub use tree_clf::DecisionTreeClassifier; 
pub use tree_reg::DecisionTreeRegressor;

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

