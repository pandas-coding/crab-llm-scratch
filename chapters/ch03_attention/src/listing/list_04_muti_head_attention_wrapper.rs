use candle_core::{Module, Tensor, D};
use candle_nn::VarBuilder;
use crate::listing::list_03_causal_attention::CausalAttention;

/// A wrapper to implement multi-head attention.
pub struct MultiHeadAttentionWrapper {
    heads: Vec<CausalAttention>,
}

impl Module for MultiHeadAttentionWrapper {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let context_vectors =  self.heads
            .iter()
            .map(|attention| attention.forward(xs).unwrap())
            .collect::<Vec<_>>();
        let reduced = context_vectors.into_iter()
            .reduce(|acc, e| Tensor::cat(&[&acc, &e], D::Minus1).unwrap())
            .unwrap();
        Ok(reduced)
    }
}

impl MultiHeadAttentionWrapper {
    /// Creates a new `MultiHeadAttentionWrapper`
    pub fn new(
        num_heads: usize,
        d_in: usize,
        d_out: usize,
        drop_p: f32,
        qkv_bias: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let heads = (0..num_heads)
            .map(|i| CausalAttention::new(d_in, d_out, drop_p, qkv_bias, vb.pp(format!("head-{i}"))).unwrap())
            .collect::<Vec<_>>();
        Ok(Self { heads })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{get_vb};

    #[test]
    fn test_multihead_attention_wrapper_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out) = (3usize, 5usize);
        let num_heads = 3usize;
        let multihead_attention = MultiHeadAttentionWrapper::new(
            num_heads,
            d_in,
            d_out,
            0.5f32,
            false,
            vb.pp("multihead_attention"),
        )?;

        assert_eq!(multihead_attention.heads.len(), num_heads);

        for i in 0..num_heads {
            let causal_attention = &multihead_attention.heads[i];
            assert_eq!(causal_attention.w_query().weight().dims(), &[d_out, d_in]);
            assert_eq!(causal_attention.w_key().weight().dims(), &[d_out, d_in]);
            assert_eq!(causal_attention.w_value().weight().dims(), &[d_out, d_in]);
            assert_eq!(causal_attention.drop_p(), 0.5f32);
        }

        Ok(())
    }

    #[test]
    fn test_multihead_attention_wrapper_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out) = (3usize, 5usize);
        let num_heads = 3usize;
        let multihead_attention = MultiHeadAttentionWrapper::new(
            num_heads,
            d_in,
            d_out,
            0.5f32,
            false,
            vb.pp("multihead_attention"),
        )?;

        // create batch
        let input_len = 10usize;
        let xs = Tensor::rand(0f32, 1f32, (input_len, d_in), &vb.device())?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;
        let context_vectors = multihead_attention.forward(&batch)?;

        assert_eq!(context_vectors.dims(), &[2usize, input_len, num_heads * d_out]);
        Ok(())
    }
}
