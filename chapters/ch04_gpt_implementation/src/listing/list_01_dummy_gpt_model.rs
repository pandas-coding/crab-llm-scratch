use candle_core::{Module, Tensor};
use candle_nn::{embedding, linear_b, seq, Dropout, Embedding, Linear, Sequential, VarBuilder};

/// A placeholder GPT model architecture struct
pub struct DummyGPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    // of transformer blocks
    trf_blocks: Sequential,
    final_norm: DummyLayerNorm,
    out_head: Linear,
}

impl Module for DummyGPTModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (_batch_size, seq_len) = xs.dims2()?;
        let tok_embeds = self.tok_emb.forward(xs)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_embeds = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let x = tok_embeds.broadcast_add(&pos_embeds)?;
        let x = self.drop_emb.forward(&x, true)?;
        let x = self.trf_blocks.forward(&x)?;
        let x = self.final_norm.forward(&x)?;

        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}

impl DummyGPTModel {
    pub fn new(cfg: Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let tok_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.context_len, cfg.emb_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(cfg.drop_rate);
        let mut trf_blocks = seq();

        let trf_blocks = (0..cfg.n_layers).fold(seq(), |seq, _| seq.add(DummyTransformerBlock::new(cfg).unwrap()));
        let final_norm = DummyLayerNorm::new(cfg.emb_dim)?;
        let out_head = linear_b(cfg.emb_dim, cfg.vocab_size, false, vb.pp("out_head"))?;
        Ok(Self {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        })
    }
}

/// A placeholder LayerNorm struct
pub struct DummyLayerNorm {}

impl Module for DummyLayerNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(xs.to_owned())
    }
}

impl DummyLayerNorm {
    #[allow(unused)]
    pub fn new(emb_dim: usize) -> candle_core::Result<Self> {
        Ok(Self {})
    }
}

/// A placeholder TransformerBlock struct (used in Listing 4.1)
pub struct DummyTransformerBlock {}

impl Module for DummyTransformerBlock {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        Ok(xs.to_owned())
    }
}

impl DummyTransformerBlock {
    #[allow(unused)]
    pub fn new(cfg: Config) -> candle_core::Result<Self> {
        Ok(Self {})
    }
}

/// Config for specifying parameters of a GPT-2 model
#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub vocab_size: usize,
    pub context_len: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl Config {
    /// Returns configuration for GPT-2 small
    #[allow(dead_code)]
    pub fn gpt2_124m() -> Self {
        Self {
            vocab_size: 50_257,
            context_len: 1_024,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    /// Returns configuration for GPT-2 medium
    #[allow(dead_code)]
    pub fn gpt2_medium() -> Self {
        Self {
            vocab_size: 50_257,
            context_len: 1_024,
            emb_dim: 1_024,
            n_heads: 16,
            n_layers: 24,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    /// Returns configuration for GPT-2 large
    #[allow(dead_code)]
    pub fn gpt2_large() -> Self {
        Self {
            vocab_size: 50_257,
            context_len: 1_024,
            emb_dim: 1_280,
            n_heads: 20,
            n_layers: 36,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    /// Returns configuration for GPT-2 x-large
    #[allow(dead_code)]
    pub fn gpt2_xlarge() -> Self {
        Self {
            vocab_size: 50_257,
            context_len: 1_024,
            emb_dim: 1_600,
            n_heads: 25,
            n_layers: 48,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }

    /// Returns a custom configuration for GPT-2 to be used in unit tests
    #[allow(dead_code)]
    pub fn gpt_sm_test() -> Self {
        Self {
            vocab_size: 500,
            context_len: 10,
            emb_dim: 12,
            n_heads: 3,
            n_layers: 2,
            drop_rate: 0.1,
            qkv_bias: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{get_batch_token_ids, get_vb};
    use super::*;

    #[test]
    fn test_dummy_gpt_model_init() -> anyhow::Result<()> {
        let cfg = Config::gpt_sm_test();
        let model = DummyGPTModel::new(cfg, get_vb())?;

        assert_eq!(model.tok_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.trf_blocks.len() as usize, cfg.n_layers);
        assert_eq!(
            model.out_head.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
        Ok(())
    }

    #[test]
    fn test_dummy_gpt_model_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let batch_token_ids = get_batch_token_ids();
        let (batch_size, seq_len) = batch_token_ids.dims2()?;

        let cfg = Config::gpt_sm_test();
        let model = DummyGPTModel::new(cfg, get_vb())?;

        let logits = model.forward(&batch_token_ids)?;

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);

        Ok(())
    }
}
