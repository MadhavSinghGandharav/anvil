
use std::collections::{HashMap, hash_map::Entry};
use std::hash::Hash;
use std::fmt::Debug;

use ndarray::Array1;

/// Encodes categorical labels as integers.
///
/// Example:
///
/// ["cat", "dog", "cat"] → [0, 1, 0]
pub struct LabelEncoder<T> {
    classes: Option<Vec<T>>,
    map: Option<HashMap<T, usize>>,
}

impl<T> LabelEncoder<T>
where
    T: Eq + Hash + Clone + Debug,
{
    /// Creates a new `LabelEncoder`.
    pub fn new() -> Self {
        Self {
            classes: None,
            map: None,
        }
    }

    /// Returns learned classes.
    ///
    /// # Panics
    ///
    /// Panics if the encoder has not been fitted.
    pub fn classes(&self) -> &[T] {
        self.classes
            .as_ref()
            .expect("LabelEncoder not fitted. Call `fit` first.")
    }

    /// Learns unique classes from the input labels.
    pub fn fit(&mut self, y: &[T]) {

        let mut map: HashMap<T, usize> = HashMap::with_capacity(y.len());
        let mut classes: Vec<T> = Vec::with_capacity(y.len());

        for label in y {
            match map.entry(label.clone()) {
                Entry::Occupied(_) => {}
                Entry::Vacant(v) => {
                    let id = classes.len();
                    v.insert(id);
                    classes.push(label.clone());
                }
            }
        }

        self.classes = Some(classes);
        self.map = Some(map);
    }

    /// Transforms labels into encoded integers.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - encoder not fitted
    /// - unseen label encountered
    pub fn transform(&self, y: &[T]) -> Array1<usize> {

        let map = self
            .map
            .as_ref()
            .expect("LabelEncoder not fitted. Call `fit` first.");

        let mut encoded = Vec::with_capacity(y.len());

        for label in y {
            match map.get(label) {
                Some(&id) => encoded.push(id),
                None => panic!("Unknown label encountered: {:?}", label),
            }
        }

        Array1::from_vec(encoded)
    }

    /// Fits the encoder and returns encoded labels.
    pub fn fit_transform(&mut self, y: &[T]) -> Array1<usize> {
        self.fit(y);
        self.transform(y)
    }

    /// Converts encoded integers back to original labels.
    ///
    /// # Panics
    ///
    /// Panics if:
    ///
    /// - encoder not fitted
    /// - invalid encoded index encountered

    pub fn inverse_transform(&self, y: &[usize]) -> Array1<T> {

        let classes = self
            .classes
            .as_ref()
            .expect("LabelEncoder not fitted. Call `fit` first.");

        let mut decoded = Vec::with_capacity(y.len());

        for &idx in y {
            match classes.get(idx) {
                Some(label) => decoded.push(label.clone()),
                None => panic!("Invalid encoded label index: {}", idx),
            }
        }

        Array1::from_vec(decoded)
    }
}

