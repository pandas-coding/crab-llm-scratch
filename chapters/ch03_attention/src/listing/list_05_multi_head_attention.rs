use candle_core::{ModuleT, Tensor, D};
use candle_nn::{linear_b, Dropout, Linear, VarBuilder};
use candle_nn::ops::{dropout, softmax};
use crate::listing::{get_mask, masked_fill};
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    num_heads: usize,
    d_out: usize,
    head_dim: usize,
    w_query: Linear,
    w_key: Linear,
    w_value: Linear,
    out_proj: Linear,
    scaling: f64,
    dropout: Dropout,
    drop_p: f32,
}

impl ModuleT for MultiHeadAttention {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (b, num_tokens, d_in) = xs.dims3()?;
        let queries = self.w_query.forward_t(xs, train)?;
        let keys = self.w_key.forward_t(xs, train)?;
        let values = self.w_value.forward_t(xs, train)?;

        // reshapes to facilitate getting attn scores each of the individual heads
        // with one matrix multiplication
        // 从形状(b, num_tokens, num_heads, head_dim)
        // 转换到(b, num_heads, num_tokens, head_dim)
        // split matrix with num_heads dimension
        let queries = queries
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let keys = keys
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let values = values
            .reshape((b, num_tokens, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attention_scores = queries.matmul(&keys.transpose(D::Minus2, D::Minus1)?)?;

        let mask = get_mask(num_tokens, xs.device())?;
        let masked = masked_fill(
            &attention_scores,
            &mask.broadcast_left((b, self.num_heads)).unwrap(),
            f32::NEG_INFINITY,
        )?;

        // scale
        let attention_weights = softmax(&(masked * self.scaling)?, D::Minus1)?;
        let attention_weights = self.dropout.forward(&attention_weights, train)?;

        // context vectors
        let context_vec = attention_weights.matmul(&values)?
            .transpose(1, 2)?;
        let context_vec = context_vec
            .reshape((b, num_tokens, self.d_out))?
            .contiguous()?;

        // an optional linear projection
        self.out_proj.forward_t(&context_vec, train)
    }
}

impl MultiHeadAttention {
    /// Creates a new `MultiHeadAttention`
    pub fn new(
        d_in: usize,
        d_out: usize,
        drop_p: f32,
        num_heads: usize,
        qkv_bias: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        if d_out % num_heads != 0 {
            panic!("`d_out` must be divisible by `num_heads`.")
        }
        let head_dim = d_out / num_heads;

        let w_query = linear_b(d_in, d_out, qkv_bias, vb.pp("w_query"))?;
        let w_key = linear_b(d_in, d_out, qkv_bias, vb.pp("w_key"))?;
        let w_value = linear_b(d_in, d_out, qkv_bias, vb.pp("w_value"))?;
        let out_proj = linear_b(d_out, d_out, true, vb.pp("out_project"))?;
        let scaling = 1. / (head_dim as f64).sqrt();
        let dropout = Dropout::new(drop_p);

        Ok(Self {
            num_heads,
            d_out,
            head_dim,
            w_query,
            w_key,
            w_value,
            out_proj,
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

    pub fn out_proj(&self) -> &Linear {
        &self.out_proj
    }

    pub fn d_out(&self) -> usize {
        self.d_out
    }

    pub fn scaling(&self) -> f64 {
        self.scaling
    }

    pub fn dropout(&self) -> &Dropout {
        &self.dropout
    }

    pub fn drop_p(&self) -> f32 {
        self.drop_p
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Manual implementation of forward
    ///
    /// Note: that blanket implementation of `ModuleT` when a type implements
    /// `Module` prevents having `forward` being overrided. Thus, this type
    /// is `ModuleT` but technicall not `Module`.
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_t(xs, true)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_vb;
    use super::*;

    #[test]
    fn test_mha_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out, num_heads) = (3usize, 6usize, 2usize);
        let mha = MultiHeadAttention::new(
            d_in,
            d_out,
            0.5f32,
            num_heads,
            false,
            vb.pp("attention"),
        )?;

        assert_eq!(mha.w_query.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.w_key.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.w_value.weight().dims(), &[d_out, d_in]);
        assert_eq!(mha.out_proj.weight().dims(), &[d_out, d_out]);
        assert_eq!(mha.head_dim, d_out / num_heads);
        assert_eq!(mha.drop_p, 0.5f32);

        Ok(())
    }

    #[test]
    #[should_panic(expected = "`d_out` must be divisible by `num_heads`.")]
    fn test_mah_init_panics_nondivisible_heads() {
        let vb = get_vb();
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 4_usize);
        let _ =
            MultiHeadAttention::new(d_in, d_out, 0.5_f32, num_heads, false, vb.pp("attn")).unwrap();
    }

    #[test]
    fn test_mah_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let (d_in, d_out, num_heads) = (3_usize, 6_usize, 3_usize);
        let mha = MultiHeadAttention::new(
            d_in,
            d_out,
            0.5f32,
            num_heads,
            false,
            vb.pp("attention"),
        )?;

        // create batch
        let input_len = 10usize;
        let xs = Tensor::rand(0f32, 1f32, (input_len, d_in), &vb.device())?;
        let batch = Tensor::stack(&[&xs, &xs], 0)?;
        let context_vectors = mha.forward(&batch)?;

        assert_eq!(context_vectors.dims(), &[2usize, input_len, d_out]);
        Ok(())
    }
}
