use std::collections::{HashMap, hash_map::Entry};
use std::hash::Hash;
use std::fmt::Debug;

use ndarray::{Array1, ArrayView1};

use crate::core::{Transformer, AnvilError};

pub struct LabelEncoder<T> {
    classes: Option<Vec<T>>,
    map: Option<HashMap<T, usize>>,
}

impl<T> LabelEncoder<T>
where
    T: Eq + Hash + Clone + Debug,
{
    pub fn new() -> Self {
        Self {
            classes: None,
            map: None,
        }
    }

    /// # Errors
    /// - NotFitted
    pub fn classes(&self) -> Result<&[T], AnvilError> {
        self.classes
            .as_ref()
            .map(|v| v.as_slice())
            .ok_or(AnvilError::NotFitted)
    }

    /// # Errors
    /// - NotFitted
    /// - InvalidParam
    pub fn inverse_transform(
        &self,
        y: ArrayView1<usize>,
    ) -> Result<Array1<T>, AnvilError> {

        let classes = self.classes.as_ref().ok_or(AnvilError::NotFitted)?;

        let mut decoded = Vec::with_capacity(y.len());

        for &idx in y.iter() {
            match classes.get(idx) {
                Some(label) => decoded.push(label.clone()),
                None => {
                    return Err(AnvilError::InvalidParam {
                        param: "y",
                        reason: format!("invalid encoded index {}", idx),
                    });
                }
            }
        }

        Ok(Array1::from_vec(decoded))
    }
}

impl<'a,T> Transformer<ArrayView1<'a,T>, Array1<usize>> for LabelEncoder<T>
where
    T: Eq + Hash + Clone + Debug,
{
    /// # Errors
    /// - EmptyDataset
    fn fit(&mut self, y: ArrayView1<T>) -> Result<(), AnvilError> {

        if y.is_empty() {
            return Err(AnvilError::EmptyDataset {
                target: "y",
            });
        }

        let mut map = HashMap::with_capacity(y.len());
        let mut classes = Vec::with_capacity(y.len());

        for label in y.iter() {
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

        Ok(())
    }

    /// # Errors
    /// - NotFitted
    /// - InvalidParam (unknown label)
    fn transform(
        &self,
        y: ArrayView1<T>,
    ) -> Result<Array1<usize>, AnvilError> {

        let map = self.map.as_ref().ok_or(AnvilError::NotFitted)?;

        let mut encoded = Vec::with_capacity(y.len());

        for label in y.iter() {
            match map.get(label) {
                Some(&id) => encoded.push(id),
                None => {
                    return Err(AnvilError::InvalidParam {
                        param: "y",
                        reason: format!("unknown label: {:?}", label),
                    });
                }
            }
        }

        Ok(Array1::from_vec(encoded))
    }
}
