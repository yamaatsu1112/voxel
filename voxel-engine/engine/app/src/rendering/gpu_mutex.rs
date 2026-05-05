use std::any::TypeId;

pub trait GpuMutex: 'static {
    const IS_NOOP: bool = false;
}

pub struct NoMutex;

impl GpuMutex for NoMutex {
    const IS_NOOP: bool = true;
}

pub trait GpuMutexList: 'static {
    fn mutex_ids() -> Vec<TypeId>;
}

impl<M: GpuMutex> GpuMutexList for M {
    fn mutex_ids() -> Vec<TypeId> {
        if M::IS_NOOP {
            Vec::new()
        } else {
            vec![TypeId::of::<M>()]
        }
    }
}

macro_rules! impl_gpu_mutex_list_for_tuple {
    ($($T:ident),+) => {
        impl<$($T: GpuMutex),+> GpuMutexList for ($($T,)+) {
            fn mutex_ids() -> Vec<TypeId> {
                let mut ids = Vec::new();
                $(
                    if !<$T as GpuMutex>::IS_NOOP {
                        ids.push(TypeId::of::<$T>());
                    }
                )+
                ids
            }
        }
    };
}

impl_gpu_mutex_list_for_tuple!(A, B);
impl_gpu_mutex_list_for_tuple!(A, B, C);
impl_gpu_mutex_list_for_tuple!(A, B, C, D);
impl_gpu_mutex_list_for_tuple!(A, B, C, D, E);

#[cfg(test)]
mod tests {
    use super::{GpuMutex, GpuMutexList, NoMutex};
    use std::any::TypeId;

    struct VoxelPipelineMutex;
    impl GpuMutex for VoxelPipelineMutex {}

    struct DynamicVoxelPipelineMutex;
    impl GpuMutex for DynamicVoxelPipelineMutex {}

    #[test]
    fn no_mutex_list_is_empty() {
        assert!(<NoMutex as GpuMutexList>::mutex_ids().is_empty());
    }

    #[test]
    fn single_mutex_list_contains_single_type_id() {
        let ids = <VoxelPipelineMutex as GpuMutexList>::mutex_ids();

        assert_eq!(ids, vec![TypeId::of::<VoxelPipelineMutex>()]);
    }

    #[test]
    fn tuple_mutex_list_contains_both_type_ids_in_order() {
        let ids = <(VoxelPipelineMutex, DynamicVoxelPipelineMutex) as GpuMutexList>::mutex_ids();

        assert_eq!(
            ids,
            vec![
                TypeId::of::<VoxelPipelineMutex>(),
                TypeId::of::<DynamicVoxelPipelineMutex>(),
            ]
        );
    }

    #[test]
    fn tuple_with_noop_filters_no_mutex() {
        let ids = <(NoMutex, VoxelPipelineMutex) as GpuMutexList>::mutex_ids();

        assert_eq!(ids, vec![TypeId::of::<VoxelPipelineMutex>()]);
    }
}
