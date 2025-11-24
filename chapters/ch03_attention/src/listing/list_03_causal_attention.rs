use candle_core::{Device, Module, Tensor, D};
use candle_nn::{linear_b, Dropout, Linear, VarBuilder};
use candle_nn::ops::softmax;

pub fn get_mask(size: usize, device: &Device) -> candle_core::Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u32::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

/// A compact causal attention class
pub struct CausalAttention {
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    scaling: f64,
    dropout: Dropout,
    drop_p: f32,
}

impl Module for CausalAttention {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // handle batches
        let (b, num_tokens, _d_in) = xs.dims3()?;
        let queries = self.w_query.forward(xs)?;
        let keys = self.w_key.forward(xs)?;
        let values = self.w_value.forward(xs)?;

        // transpose dim1 and dim2, make batch dim remain the same.
        let attention_scores = queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;
        let mask = get_mask(num_tokens, xs.device())?;
        let masked = masked_fill(
            &attention_scores,
            &mask.broadcast_left(b).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // scale
        let mut attention_weights = softmax(&(masked * self.scaling)?, D::Minus1)?;
        // dropout
        attention_weights = self.dropout.forward(&attention_weights, true)?;
        // context vectors
        attention_weights.matmul(&values)
    }
}

impl CausalAttention {

    pub fn new(
        d_in: usize,
        d_out: usize,
        drop_p: f32,
        qkv_bias: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("value"))?;
        let scaling = 1f64 / (w_key.weight().dims()[0] as f64).sqrt();
        let dropout = Dropout::new(drop_p);

        Ok(Self {
            w_query,
            w_key,
            w_value,
            scaling,
            dropout,
            drop_p,
        })
    }

    pub fn w_query(&self) -> &Linear {
        &self.w_query
    }

    pub fn w_key(&self) -> &Linear {
        &self.w_key
    }

    pub fn w_value(&self) -> &Linear {
        &self.w_value
    }

    pub fn drop_p(&self) -> f32 {
        self.drop_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_attention_init() -> anyhow::Result<()> {
        let vb = crate::test_utils::get_vb();
        let (d_in, d_out) = (3usize, 5usize);
        let casual_attention = CausalAttention::new(d_in, d_out, 0.5f32, false, vb.pp("attention"))?;

        assert_eq!(casual_attention.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attention.w_key.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attention.w_value.weight().dims(), &[d_out, d_in]);
        assert_eq!(casual_attention.drop_p, 0.5f32);
        Ok(())
    }

    #[test]
    fn test_causal_attention_forward() -> anyhow::Result<()> {
        let vb = crate::test_utils::get_vb();
        let (d_in, d_out) = (3usize, 5usize);
        let casual_attention = CausalAttention::new(d_in, d_out, 0.5f32, false, vb.pp("attention"))?;

        // create batch
        let input_len = 10usize;
        let xs = Tensor::rand(0f32, 1f32, (input_len, d_in), &vb.device())?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;
        let context_vectors = casual_attention.forward(&batch)?;

        assert_eq!(context_vectors.dims(), &[2usize, input_len, d_out]);
        Ok(())
    }
}

