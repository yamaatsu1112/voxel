use std::alloc::Layout;

pub trait Component: Clone + Copy {}

pub struct ComponentStorage {
    data_ptr: *mut u8,
    data_len: usize,
    layout: Layout,
    type_size: usize,
    type_align: usize,
}

impl ComponentStorage {
    pub fn new<T: Component + 'static>() -> Self {
        Self {
            data_ptr: std::ptr::null_mut(),
            data_len: 0,
            layout: Layout::new::<u8>(),
            type_size: std::mem::size_of::<T>(),
            type_align: std::mem::align_of::<T>(),
        }
    }

    /// Set a component at a specific index
    pub fn set<T: Component + 'static>(&mut self, index: usize, component: T) {
        if std::mem::size_of::<T>() != self.type_size {
            println!("ComponentStorage::set: Type size mismatch");
            return;
        }

        // For ZST, set dangling pointer and update data_len
        if self.type_size == 0 {
            self.data_ptr = std::ptr::NonNull::dangling().as_ptr();
            let new_capacity = index + 1;
            if new_capacity > self.data_len {
                self.data_len = new_capacity;
            }
            return;
        }

        let new_capacity = index + 1;
        let new_size = new_capacity * self.type_size;

        // Check if we need to reallocate
        if new_size > self.data_len * self.type_size {
            let new_layout = Layout::array::<T>(new_capacity)
                .expect("Failed to create layout for component array");

            unsafe {
                // If we have existing data, reallocate; otherwise allocate new memory
                let new_ptr = if !self.data_ptr.is_null() {
                    std::alloc::realloc(self.data_ptr, self.layout, new_size)
                } else {
                    std::alloc::alloc(new_layout)
                };

                // Check for allocation failure
                if new_ptr.is_null() {
                    std::alloc::handle_alloc_error(new_layout);
                }

                self.data_ptr = new_ptr;
                self.data_len = new_capacity;
            }

            self.layout = new_layout;
        }

        // Write the component at the specified index
        let ptr = self.data_ptr as *mut T;
        unsafe {
            let elem_ptr = ptr.add(index);
            std::ptr::write(elem_ptr, component);
        }
    }

    /// Get a component at a specific index
    pub fn get<T: Component + 'static>(&self, index: usize) -> Option<&T> {
        if std::mem::size_of::<T>() != self.type_size {
            panic!("ComponentStorage::get: Type size mismatch");
        }
        if self.data_ptr.is_null() {
            panic!("ComponentStorage::get: Data pointer is null");
        }
        if index >= self.data_len {
            panic!("ComponentStorage::get: Index out of bounds");
        }

        // Handle ZST
        if self.type_size == 0 {
            if index >= self.data_len {
                panic!("ComponentStorage::get: Index out of bounds");
            }
            return Some(unsafe { &*std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        let ptr = self.data_ptr as *mut T;
        unsafe {
            let elem_ptr = ptr.add(index);
            Some(&*elem_ptr)
        }
    }

    /// Get a mutable component at a specific index
    pub fn get_mut<T: Component + 'static>(&mut self, index: usize) -> Option<&mut T> {
        if std::mem::size_of::<T>() != self.type_size {
            panic!("ComponentStorage::get_mut: Type size mismatch");
        }

        // Handle ZST
        if self.type_size == 0 {
            if index >= self.data_len {
                panic!("ComponentStorage::get_mut: Index out of bounds");
            }
            return Some(unsafe { &mut *std::ptr::NonNull::<T>::dangling().as_ptr() });
        }

        if self.data_ptr.is_null() {
            panic!("ComponentStorage::get_mut: Data pointer is null");
        }
        if index >= self.data_len {
            panic!("ComponentStorage::get_mut: Index out of bounds");
        }

        let ptr = self.data_ptr as *mut T;
        unsafe {
            let elem_ptr = ptr.add(index);
            Some(&mut *elem_ptr)
        }
    }

    pub fn get_copy(&self, index: usize) -> &[u8] {
        if self.data_ptr.is_null() {
            panic!("ComponentStorage::get_copy: Data pointer is null");
        }
        if index >= self.data_len {
            panic!("ComponentStorage::get_copy: Index out of bounds");
        }

        // Return empty slice for ZST
        if self.type_size == 0 {
            if index >= self.data_len {
                panic!("ComponentStorage::get_copy: Index out of bounds");
            }
            return &[];
        }

        unsafe {
            std::slice::from_raw_parts(self.data_ptr.add(index * self.type_size), self.type_size)
        }
    }

    pub fn set_copy(&mut self, index: usize, component: &[u8]) {
        if component.len() != self.type_size {
            println!("ComponentStorage::set_copy: Component size mismatch");
            return;
        }

        // For ZST, set dangling pointer and update data_len
        if self.type_size == 0 {
            self.data_ptr = std::ptr::NonNull::dangling().as_ptr();
            let new_capacity = index + 1;
            if new_capacity > self.data_len {
                self.data_len = new_capacity;
            }
            return;
        }

        let new_capacity = index + 1;
        let new_size = new_capacity * self.type_size;

        // Check if we need to reallocate
        if new_size > self.data_len * self.type_size {
            let new_layout = Layout::from_size_align(new_size, self.type_align) // TODO: check overflow
                .expect("Failed to create layout for component array");

            unsafe {
                // If we have existing data, reallocate; otherwise allocate new memory
                let new_ptr = if !self.data_ptr.is_null() {
                    std::alloc::realloc(self.data_ptr, self.layout, new_size)
                } else {
                    std::alloc::alloc(new_layout)
                };

                // Check for allocation failure
                if new_ptr.is_null() {
                    std::alloc::handle_alloc_error(new_layout);
                }

                self.data_ptr = new_ptr;
                self.data_len = new_capacity;
            }

            self.layout = new_layout;
        }

        let ptr = self.data_ptr;
        unsafe {
            let elem_ptr = ptr.add(index * self.type_size);
            std::ptr::copy_nonoverlapping(component.as_ptr(), elem_ptr, self.type_size);
        }
    }

    pub fn empty_clone(&self) -> ComponentStorage {
        Self {
            data_ptr: std::ptr::null_mut(),
            data_len: 0,
            layout: Layout::new::<u8>(),
            type_size: self.type_size,
            type_align: self.type_align,
        }
    }
}

// SAFETY: ComponentStorage owns its allocated memory exclusively.
// The raw pointer is only used internally and is not shared with other threads.
unsafe impl Send for ComponentStorage {}

impl Drop for ComponentStorage {
    fn drop(&mut self) {
        // Skip deallocation for ZST (no memory was allocated)
        if self.type_size == 0 {
            return;
        }
        // Deallocate memory if it was allocated
        if !self.data_ptr.is_null() {
            unsafe {
                std::alloc::dealloc(self.data_ptr, self.layout);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }
    impl Component for Position {}

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Marker;
    impl Component for Marker {}

    #[test]
    fn test_zst_component_storage_new() {
        let storage = ComponentStorage::new::<Marker>();
        assert_eq!(storage.type_size, 0);
        assert_eq!(storage.data_len, 0);
    }

    #[test]
    fn test_zst_component_set_and_get() {
        let mut storage = ComponentStorage::new::<Marker>();
        storage.set(0, Marker);
        storage.set(2, Marker);

        let marker0 = storage.get::<Marker>(0);
        assert!(marker0.is_some());

        let marker2 = storage.get::<Marker>(2);
        assert!(marker2.is_some());
    }

    #[test]
    fn test_zst_component_get_mut() {
        let mut storage = ComponentStorage::new::<Marker>();
        storage.set(0, Marker);

        let marker = storage.get_mut::<Marker>(0);
        assert!(marker.is_some());
    }

    #[test]
    fn test_zst_component_get_copy() {
        let mut storage = ComponentStorage::new::<Marker>();
        storage.set(0, Marker);

        let copy = storage.get_copy(0);
        assert_eq!(copy.len(), 0);
    }

    #[test]
    fn test_zst_component_set_copy() {
        let mut storage = ComponentStorage::new::<Marker>();
        storage.set_copy(0, &[]);
        storage.set_copy(2, &[]);

        assert!(storage.data_len >= 3);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_zst_component_get_out_of_bounds() {
        let mut storage = ComponentStorage::new::<Marker>();
        storage.set(0, Marker);
        storage.get::<Marker>(5);
    }

    #[test]
    fn test_normal_component_still_works() {
        let mut storage = ComponentStorage::new::<Position>();
        storage.set(0, Position { x: 1.0, y: 2.0 });
        storage.set(1, Position { x: 3.0, y: 4.0 });

        let pos0 = storage.get::<Position>(0).unwrap();
        assert_eq!(pos0.x, 1.0);
        assert_eq!(pos0.y, 2.0);

        let pos1 = storage.get::<Position>(1).unwrap();
        assert_eq!(pos1.x, 3.0);
        assert_eq!(pos1.y, 4.0);
    }

    #[test]
    fn test_zst_empty_clone() {
        let mut storage = ComponentStorage::new::<Marker>();
        storage.set(0, Marker);

        let cloned = storage.empty_clone();
        assert_eq!(cloned.type_size, 0);
        assert_eq!(cloned.data_len, 0);
    }
}
