use std::any::Any;
use std::any::TypeId;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

pub struct Resource {
    data: HashMap<TypeId, Box<dyn Any + Send>>,
}

impl Resource {
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.data
            .get(&TypeId::of::<T>())
            .and_then(|data| data.downcast_ref::<T>())
    }

    pub fn get_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.data
            .get_mut(&TypeId::of::<T>())
            .and_then(|data| data.downcast_mut::<T>())
    }

    pub fn insert<T: Send + 'static>(&mut self, data: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(data));
    }

    pub fn remove<T: 'static>(&mut self) {
        self.data.remove(&TypeId::of::<T>());
    }
}

impl Default for Resource {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
pub struct Res<'w, T> {
    value: &'w T,
}

impl<'w, T> Res<'w, T> {
    pub fn new(value: &'w T) -> Self {
        Self { value }
    }

    pub fn into_inner(self) -> &'w T {
        self.value
    }
}

impl<'w, T> Deref for Res<'w, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

pub struct ResMut<'w, T> {
    value: &'w mut T,
}

impl<'w, T> ResMut<'w, T> {
    pub fn new(value: &'w mut T) -> Self {
        Self { value }
    }

    pub fn into_inner(self) -> &'w mut T {
        self.value
    }
}

impl<'w, T> Deref for ResMut<'w, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

impl<'w, T> DerefMut for ResMut<'w, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
    }
}
