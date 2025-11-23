use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};

/// VarMap helper builder.
pub(crate) fn get_vb() -> candle_nn::VarBuilder<'static> {
    let dev = Device::cuda_if_available(0).unwrap();
    let varmap = VarMap::new();
    VarBuilder::from_varmap(&varmap, DType::F32, &dev)
}