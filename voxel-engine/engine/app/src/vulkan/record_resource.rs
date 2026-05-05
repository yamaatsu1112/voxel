use std::any::{Any, TypeId};
use std::collections::HashMap;

trait CloneAny: Any + Send + Sync {
    fn clone_box(&self) -> Box<dyn CloneAny>;
}

impl<T: Clone + Send + Sync + 'static> CloneAny for T {
    fn clone_box(&self) -> Box<dyn CloneAny> {
        Box::new(self.clone())
    }
}

pub struct RecordResource {
    data: HashMap<TypeId, Box<dyn CloneAny>>,
}

impl Clone for RecordResource {
    fn clone(&self) -> Self {
        Self {
            data: self
                .data
                .iter()
                .map(|(key, value)| (*key, value.as_ref().clone_box()))
                .collect(),
        }
    }
}

impl Default for RecordResource {
    fn default() -> Self {
        Self::new()
    }
}

impl RecordResource {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.data
            .get(&TypeId::of::<T>())
            .and_then(|data| (data.as_ref() as &dyn Any).downcast_ref::<T>())
    }

    pub fn insert<T: Clone + Send + Sync + 'static>(&mut self, data: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(data));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_resource_insert_get() {
        let mut resources = RecordResource::new();
        resources.insert::<u32>(42);

        assert_eq!(resources.get::<u32>(), Some(&42));
        assert!(resources.get::<i32>().is_none());
    }

    #[test]
    fn record_resource_clone_preserves_inserted_values() {
        let mut resources = RecordResource::new();
        resources.insert::<u32>(42);
        resources.insert::<String>("test".to_string());

        let cloned = resources.clone();

        assert_eq!(cloned.get::<u32>(), Some(&42));
        assert_eq!(cloned.get::<String>().map(|s| s.as_str()), Some("test"));
    }
}
