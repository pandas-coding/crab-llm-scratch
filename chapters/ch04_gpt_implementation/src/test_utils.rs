use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

pub fn get_vb() -> VarBuilder<'static> {
    let device = Device::cuda_if_available(0).unwrap();
    let varmap = VarMap::new();
    VarBuilder::from_varmap(&varmap, DType::F32, &device)
}

pub fn get_batch_token_ids() -> Tensor {
    let device = Device::cuda_if_available(0).unwrap();
    Tensor::new(&[[101_u32, 366, 100, 345], [101, 100, 322, 57]], &device).unwrap()
}
