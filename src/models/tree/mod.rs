mod tree_clf;
mod impurity; 

pub use tree_clf::DecisionTreeClassifier; 

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
    Entropy
}

pub enum Splitter{
    Best,
    Random
}

