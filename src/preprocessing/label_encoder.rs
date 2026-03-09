use std::collections::{HashMap,hash_map::Entry};
use std::hash::Hash;
use std::usize;
use std::fmt::Debug;

pub struct LabelEncoder<T> {
    classes: Option<Vec<T>>,
map: Option<HashMap<T, usize>>
}



impl<T> LabelEncoder<T>
where
    T: Eq + Hash + Clone + Debug,
{
    pub fn new() -> Self {
        Self { 
            classes: None,
            map: None
        }
    }
    
    pub fn classes(&self) -> &[T]{
        self.classes.as_ref().expect("You need to fit beform transform")
    }
    pub fn fit(&mut self, y: &[T]) {

        let mut map: HashMap<T,usize> = HashMap::new();
        let mut classes: Vec<T> = Vec::new();

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

    pub fn transform(&self, y:&[T]) -> Vec<usize> {

        let map = self.map.as_ref().expect("You need to fit before transorm");
        let mut encoded = Vec::with_capacity(y.len());

        for label in y{
            match map.get(label) {
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
