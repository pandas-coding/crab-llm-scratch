use candle_core::{Module, Tensor};
use candle_nn::{linear_b, Linear, VarBuilder};

/// A self-attention class using candle_nn::Linear
pub struct SelfAttentionV2 {
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    scaling: f64,
}

impl Module for SelfAttentionV2 {
    /// Computes the context vector for `xs`
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let queries = self.w_query.forward(xs)?;
        let keys = self.w_key.forward(xs)?;
        let values = self.w_value.forward(xs)?;
        
        let attention_scores = queries.matmul(&keys.t()?)?;
        let attention_weights = candle_nn::ops::softmax(&(attention_scores * self.scaling)?, 1)?;
        attention_weights.matmul(&values)
    }
}

impl SelfAttentionV2 {
    /// Creates a new `SelfAttentionV2`
    pub fn new(d_in: usize, d_out: usize, qkv_bias: bool, vb: VarBuilder) -> candle_core::Result<Self> {
        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("value"))?;
        let scaling = 1f64 / (w_key.weight().dims()[0] as f64).sqrt();
        
        Ok(Self {
            w_query,
            w_key,
            w_value,
            scaling,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{get_vb};
    
    #[test]
    fn test_self_attention_v2_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out) = (3usize, 5usize);
        let attention_v1_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn"))?;
        
        assert_eq!(attention_v1_layer.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(attention_v1_layer.w_key.weight().dims(),  &[d_out, d_in]);
        assert_eq!(attention_v1_layer.w_value.weight().dims(),  &[d_out, d_in]);
        Ok(())
    }
    
    #[test]
    fn test_self_attention_v2_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out) = (3usize, 5usize);
        let attention_v2_layer = SelfAttentionV2::new(d_in, d_out, false, vb.pp("attn"))?;
        
        let input_len = 10usize;
        let xs = Tensor::rand(0f32, 1f32, (input_len, d_in), vb.device())?;
        let context_vectors = attention_v2_layer.forward(&xs)?;
        
        assert_eq!(context_vectors.dims(), &[input_len, d_out]);
        Ok(())
    }
}
