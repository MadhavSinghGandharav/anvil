use std::collections::{HashMap,hash_map::Entry};
use std::hash::Hash;
use std::usize;
use std::fmt::Debug;

pub struct LabelEncoder<T> {
    pub classes: Vec<T>,
    map: HashMap<T, usize>
}



impl<T> LabelEncoder<T>
where
    T: Eq + Hash + Clone + Debug,
{
    pub fn new() -> Self {
        Self { 
            classes: Vec::new(),
            map: HashMap::new()
        }
    }
    
    pub fn classes(&self) -> &Vec<T>{
        &self.classes
    }
    pub fn fit(&mut self, y: &[T]) {
        for label in y {
            match self.map.entry(label.clone()) {
                Entry::Occupied(_) => {}
                Entry::Vacant(v) => {
                    let id = self.classes.len();
                    v.insert(id);
                    self.classes.push(label.clone());
                }
            }
        }
    }

    pub fn transform(&self, y:&[T]) -> Vec<usize> {
        let mut encoded = Vec::with_capacity(y.len());

        for label in y{
            match self.map.get(label) {
                Some(&id) => encoded.push(id),
                None => panic!("new values encounterd {:?}",label),
            }
        }
        encoded
    }
    pub fn fit_transform(&mut self, y: &[T]) -> Vec<usize> {
        self.fit(y);
        self.transform(y)
    }
}
