use candle_core::Tensor;

/// A compact self-attention class
pub struct SelfAttentionV1 {
    pub w_query: Tensor,
    pub w_key: Tensor,
    pub w_value: Tensor,
    pub scaling: f64,
}

impl candle_core::Module for SelfAttentionV1 {
    /// Computes the context vector for `xs`
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let queries = xs.matmul(&self.w_query)?;
        let keys = xs.matmul(&self.w_key)?;
        let values = xs.matmul(&self.w_value)?;

        let attention_scores = queries.matmul(&keys.t()?)?;
        let attention_weights = candle_nn::ops::softmax(&(attention_scores * self.scaling)?, 1)?;
        attention_weights.matmul(&values)
    }
}

impl SelfAttentionV1 {
    /// Creates a new `SelfAttentionV1`
    pub fn new(d_in: usize, d_out: usize, vb: candle_nn::VarBuilder<'_>) -> candle_core::Result<Self> {
        let init = candle_nn::init::DEFAULT_KAIMING_NORMAL;
        let w_query = vb.get_with_hints((d_in, d_out), "query", init)?;
        let w_key = vb.get_with_hints((d_in, d_out), "key", init)?;
        let w_value = vb.get_with_hints((d_in, d_out), "value", init)?;
        let scaling = 1f64 / (w_key.dims()[1] as f64).sqrt();

        Ok(Self {
            w_query,
            w_key,
            w_value,
            scaling,
        })
    }

    pub fn w_query(&self) -> &Tensor {
        &self.w_query
    }

    pub fn w_key(&self) -> &Tensor {
        &self.w_key
    }

    pub fn w_value(&self) -> &Tensor {
        &self.w_value
    }
}

#[cfg(test)]
mod test {
    use candle_core::{DType, Device, Module};
    use candle_nn::{VarBuilder, VarMap};
    use super::*;

    /// VarMap helper builder.
    fn get_vb() -> candle_nn::VarBuilder<'static> {
        let dev = Device::cuda_if_available(0).unwrap();
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, &dev)
    }

    #[test]
    fn test_self_attention_v1_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out) = (3usize, 5usize);
        let attention_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn"))?;

        assert_eq!(attention_v1_layer.w_query.dims(), &[d_in, d_out]);
        assert_eq!(attention_v1_layer.w_key.dims(), &[d_in, d_out]);
        assert_eq!(attention_v1_layer.w_value.dims(), &[d_in, d_out]);
        Ok(())
    }

    #[test]
    fn test_self_attention_v1_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out) = (3_usize, 5_usize);
        let attention_v1_layer = SelfAttentionV1::new(d_in, d_out, vb.pp("attn"))?;

        let input_len = 10usize;
        let xs = Tensor::rand(0f32, 1f32, (input_len, d_in), vb.device())?;
        let context_vectors = attention_v1_layer.forward(&xs)?;

        assert_eq!(context_vectors.dims(), &[input_len, d_out]);
        Ok(())
    }
}
