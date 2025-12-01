use candle_core::{Device, Tensor};

pub mod list_01_self_attention_v1;
pub mod list_02_self_attention_v2;
pub mod list_03_causal_attention;
pub mod list_04_muti_head_attention_wrapper;
pub mod list_05_multi_head_attention;

pub fn get_mask(size: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u32::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

/// polyfill for Pytorch masked_fill_
/// 用于应用一个已经存在的掩码（mask），将张量中对应位置的元素替换成指定值
pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}