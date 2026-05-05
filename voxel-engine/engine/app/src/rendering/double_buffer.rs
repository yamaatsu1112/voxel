use std::ops::{Deref, DerefMut};
use std::sync::{Mutex, MutexGuard};

pub struct DoubleBuffer<T> {
    buffers: [Mutex<T>; 2],
    write_index: Mutex<usize>,
}

pub struct DoubleBufferWriteGuard<'a, T> {
    buffer_guard: MutexGuard<'a, T>,
    _write_index_guard: MutexGuard<'a, usize>,
}

pub struct DoubleBufferReadGuard<'a, T> {
    buffer_guard: MutexGuard<'a, T>,
    _write_index_guard: MutexGuard<'a, usize>,
}

impl<T> Deref for DoubleBufferWriteGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.buffer_guard
    }
}

impl<T> DerefMut for DoubleBufferWriteGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer_guard
    }
}

impl<T> Deref for DoubleBufferReadGuard<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.buffer_guard
    }
}

impl<T> DerefMut for DoubleBufferReadGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer_guard
    }
}

impl<T> DoubleBuffer<T> {
    pub fn new(a: T, b: T) -> Self {
        Self {
            buffers: [Mutex::new(a), Mutex::new(b)],
            write_index: Mutex::new(0),
        }
    }

    pub fn write_buffer(&self) -> DoubleBufferWriteGuard<'_, T> {
        let write_index_guard = self
            .write_index
            .lock()
            .expect("Failed to lock double buffer write index");
        let index = *write_index_guard;
        let buffer_guard = self.buffers[index]
            .lock()
            .expect("Failed to lock double buffer write buffer");
        DoubleBufferWriteGuard {
            buffer_guard,
            _write_index_guard: write_index_guard,
        }
    }

    #[allow(dead_code)]
    pub fn read_buffer(&self) -> DoubleBufferReadGuard<'_, T> {
        let write_index_guard = self
            .write_index
            .lock()
            .expect("Failed to lock double buffer write index");
        let read_index = 1 - *write_index_guard;

        let buffer_guard = self.buffers[read_index]
            .lock()
            .expect("Failed to lock double buffer read buffer");
        DoubleBufferReadGuard {
            buffer_guard,
            _write_index_guard: write_index_guard,
        }
    }

    pub fn swap_and_read(&self) -> MutexGuard<'_, T> {
        let read_index = {
            let mut write_index_guard = self
                .write_index
                .lock()
                .expect("Failed to lock double buffer write index");
            let old_write_index = *write_index_guard;
            *write_index_guard = 1 - old_write_index;
            old_write_index
        };

        self.buffers[read_index]
            .lock()
            .expect("Failed to lock double buffer swapped read buffer")
    }

    #[allow(dead_code)]
    pub fn write_and_swap(&self, value: T) {
        let mut write_index_guard = self
            .write_index
            .lock()
            .expect("Failed to lock double buffer write index");
        let write_index = *write_index_guard;
        let mut buffer_guard = self.buffers[write_index]
            .lock()
            .expect("Failed to lock double buffer write buffer");
        *buffer_guard = value;
        *write_index_guard = 1 - *write_index_guard;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::DoubleBuffer;

    #[test]
    fn write_buffer_updates_only_write_side() {
        let double_buffer = DoubleBuffer::new(vec![1], vec![2]);

        {
            let mut write = double_buffer.write_buffer();
            write.push(3);
        }

        let read = double_buffer.read_buffer();
        assert_eq!(&*read, &vec![2]);
    }

    #[test]
    fn read_buffer_reads_opposite_of_write_side() {
        let double_buffer = DoubleBuffer::new(10, 20);

        let read = double_buffer.read_buffer();

        assert_eq!(*read, 20);
    }

    #[test]
    fn swap_and_read_flips_and_reads_previous_write_buffer() {
        let double_buffer = DoubleBuffer::new(String::from("write"), String::from("read"));

        let read = double_buffer.swap_and_read();
        assert_eq!(&*read, "write");
        drop(read);

        {
            let mut write = double_buffer.write_buffer();
            write.push_str("-next");
        }

        let read_after_swap = double_buffer.read_buffer();
        assert_eq!(&*read_after_swap, "write");

        drop(read_after_swap);
        let swapped_again = double_buffer.swap_and_read();
        assert_eq!(&*swapped_again, "read-next");
    }

    #[test]
    fn write_and_swap_writes_then_flips_for_reader() {
        let double_buffer = DoubleBuffer::new(1, 2);

        double_buffer.write_and_swap(42);

        let read = double_buffer.read_buffer();
        assert_eq!(*read, 42);
    }

    #[test]
    fn write_buffer_guard_drop_releases_both_locks() {
        let double_buffer = DoubleBuffer::new(1, 2);

        {
            let mut write = double_buffer.write_buffer();
            *write = 7;
        }

        {
            let mut write_again = double_buffer.write_buffer();
            *write_again = 9;
        }

        double_buffer.write_and_swap(11);
        let read = double_buffer.read_buffer();
        assert_eq!(*read, 11);
    }

    #[test]
    fn concurrent_write_and_swap_and_swap_and_read_complete_without_deadlock() {
        let double_buffer = Arc::new(DoubleBuffer::new(0usize, 0usize));
        let writer_buffer = Arc::clone(&double_buffer);
        let reader_buffer = Arc::clone(&double_buffer);

        let writer = thread::spawn(move || {
            for value in 1..=2000 {
                writer_buffer.write_and_swap(value);
            }
        });

        let reader = thread::spawn(move || {
            let mut last = 0usize;
            for _ in 0..2000 {
                let value = *reader_buffer.swap_and_read();
                if value > last {
                    last = value;
                }
            }
            last
        });

        writer.join().expect("writer thread panicked");
        let max_seen = reader.join().expect("reader thread panicked");
        assert!(max_seen <= 2000);
    }
}
