use std::collections::HashMap;
use std::hash::Hash;

pub struct LabelEncoder<T> {
    classes: Vec<T>,
}

impl<T> LabelEncoder<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self { classes: Vec::new() }
    }

    pub fn fit_transform(&mut self, y: &[T]) -> Vec<usize> {
        let mut map: HashMap<T, usize> = HashMap::new();
        let mut encoded = Vec::with_capacity(y.len());

        for label in y {
            let idx = match map.get(label) {
                Some(&i) => i,
                None => {
                    let new_idx = map.len();
                    map.insert(label.clone(), new_idx);
                    self.classes.push(label.clone());
                    new_idx
                }
            };
            encoded.push(idx);
        }

        encoded
    }

    pub fn classes(&self) -> &[T] {
        &self.classes
    }
}
