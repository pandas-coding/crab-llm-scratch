use candle_core::{Module, ModuleT, Tensor};
use candle_nn::{Dropout, VarBuilder};
use ch03_attention::listing::list_05_multi_head_attention::MultiHeadAttention;
use crate::listing::list_01_dummy_gpt_model::Config;
use crate::listing::list_02_layer_norm::LayerNorm;
use crate::listing::list_04_feed_forward::FeedForward;

/// The transformer block component of GPT
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    att: MultiHeadAttention,
    ff: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    drop_shortcut: Dropout,
}

impl TransformerBlock {
    /// Creates a new `TransformerBlock`
    pub fn new(cfg: Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let att = MultiHeadAttention::new(
            cfg.emb_dim,
            cfg.emb_dim,
            cfg.drop_rate,
            cfg.n_heads,
            cfg.qkv_bias,
            vb.pp("mha"),
        )?;
        let ff = FeedForward::new(cfg, vb.pp("ff"))?;
        let norm1 = LayerNorm::new(cfg.emb_dim, vb.pp("norm1"))?;
        let norm2 = LayerNorm::new(cfg.emb_dim, vb.pp("norm2"))?;
        let drop_shortcut = Dropout::new(cfg.drop_rate);
        Ok(Self {
            att,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        })
    }

    pub fn from_fields(
        att: MultiHeadAttention,
        ff: FeedForward,
        norm1: LayerNorm,
        norm2: LayerNorm,
        drop_shortcut: Dropout,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            att,
            ff,
            norm1,
            norm2,
            drop_shortcut,
        })
    }

    /// Manual implementation of forward
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_t(xs, true)
    }

    pub fn att(&self) -> &MultiHeadAttention {
        &self.att
    }

    pub fn ff(&self) -> &FeedForward {
        &self.ff
    }

    pub fn norm1(&self) -> &LayerNorm {
        &self.norm1
    }

    pub fn norm2(&self) -> &LayerNorm {
        &self.norm2
    }

    pub fn drop_shortcut(&self) -> &Dropout {
        &self.drop_shortcut
    }
}

impl ModuleT for TransformerBlock {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let shortcut = xs.to_owned();
        let x = xs.to_owned();
        let x = self.norm1.forward(&x)?;
        let x = self.att.forward_t(&x, train)?;
        let x = self.drop_shortcut.forward(&x, train)?;
        let x = (x + shortcut)?;

        let shortcut = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.ff.forward(&x)?;
        let x = self.drop_shortcut.forward(&x, train)?;
        let x = (x + shortcut)?;
        Ok(x)

    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_vb;
    use super::*;

    #[test]
    fn test_transformer_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let cfg = Config::gpt_sm_test();
        let transformer_block = TransformerBlock::new(cfg, vb.pp("transformer"))?;

        assert_eq!(transformer_block.att.num_heads(), cfg.n_heads);
        assert_eq!(transformer_block.att.drop_p(), cfg.drop_rate);
        assert_eq!(
            transformer_block.att.w_key().weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block.att.w_query().weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        assert_eq!(
            transformer_block.att.w_value().weight().dims(),
            &[cfg.emb_dim, cfg.emb_dim]
        );
        assert_eq!(transformer_block.att.head_dim(), cfg.emb_dim / cfg.n_heads);
        assert_eq!(transformer_block.ff.layers().len(), 3_usize);
        assert_eq!(transformer_block.norm1.scale().dims(), &[cfg.emb_dim]);
        assert_eq!(transformer_block.norm1.shift().dims(), &[cfg.emb_dim]);
        Ok(())
    }

    #[test]
    fn test_transformer_block_transform() -> anyhow::Result<()> {
        let vb = get_vb();
        let cfg = Config::gpt_sm_test();
        let transformer_block = TransformerBlock::new(cfg, vb.pp("transformer"))?;

        let batch_size = 2usize;
        let num_tokens = 4usize;
        let batch_example = Tensor::rand(0f32, 1f32, (batch_size, num_tokens, cfg.emb_dim), &vb.device())?;

        let out = transformer_block.forward(&batch_example)?;
        assert_eq!(out.dims(), batch_example.dims());
        Ok(())
    }
}
