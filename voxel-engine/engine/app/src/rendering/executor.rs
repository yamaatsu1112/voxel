use std::ops::{Deref, DerefMut};

use crate::vulkan::compute_executor::ComputeExecutor;

pub struct Executor {
    executor: ComputeExecutor,
}

impl Executor {
    pub fn new(executor: ComputeExecutor) -> Self {
        Self { executor }
    }
}

impl Deref for Executor {
    type Target = ComputeExecutor;

    fn deref(&self) -> &Self::Target {
        &self.executor
    }
}

impl DerefMut for Executor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.executor
    }
}
