use crate::listing::list_06_transformer_block::TransformerBlock;
use candle_core::{Module, ModuleT, Tensor};
use candle_nn::{embedding, linear_b, Dropout, Embedding, Linear, VarBuilder};
use crate::listing::list_01_dummy_gpt_model::Config;
use crate::listing::list_02_layer_norm::LayerNorm;

pub struct GPTModel {
    tok_emb: Embedding,
    pos_emb: Embedding,
    drop_emb: Dropout,
    /// of transformer blocks
    trf_blocks: SequentialTransformers,
    final_norm: LayerNorm,
    out_head: Linear,
}

impl GPTModel {
    /// Creates a new `GPTModel`
    pub fn new(cfg: Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let tok_emb = embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("tok_emb"))?;
        let pos_emb = embedding(cfg.context_len, cfg.emb_dim, vb.pp("pos_emb"))?;
        let drop_emb = Dropout::new(cfg.drop_rate);
        let trf_blocks = (0..cfg.n_layers)
            .into_iter()
            .try_fold(SequentialTransformers::default(), |trf_seq, ix| -> candle_core::Result<_> {
                let trf_block = TransformerBlock::new(cfg, vb.pp(format!("trf.{ix}")))?;
                let added_trf_seq: SequentialTransformers = trf_seq.add(trf_block);
                Ok(added_trf_seq)
            })?;

        let final_norm = LayerNorm::new(cfg.emb_dim, vb.pp("final_norm"))?;
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

    pub fn from_fields(
        tok_emb: Embedding,
        pos_emb: Embedding,
        drop_emb: Dropout,
        trf_blocks: SequentialTransformers,
        final_norm: LayerNorm,
        out_head: Linear,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            tok_emb,
            pos_emb,
            drop_emb,
            trf_blocks,
            final_norm,
            out_head,
        })
    }

    /// Manual implementation of forward
    ///
    /// Note: that blanket implementation of `ModuleT` when a type implements
    /// `Module` prevents having `forward` being overrided. Thus, this type
    /// is `ModuleT` but technically not `Module`.
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward_t(xs, true)
    }

    pub fn tok_emb(&self) -> &Embedding {
        &self.tok_emb
    }

    pub fn pos_emb(&self) -> &Embedding {
        &self.pos_emb
    }

    pub fn drop_emb(&self) -> &Dropout {
        &self.drop_emb
    }

    pub fn trf_blocks(&self) -> &SequentialTransformers {
        &self.trf_blocks
    }

    pub fn final_norm(&self) -> &LayerNorm {
        &self.final_norm
    }

    pub fn out_head(&self) -> &Linear {
        &self.out_head
    }

    pub fn set_out_head(&mut self, new_out_head: Linear) {
        self.out_head = new_out_head;
    }
}

impl ModuleT for GPTModel {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let (_batch_size, seq_len) = xs.dims2()?;
        let tok_embeds = self.tok_emb.forward(xs)?;
        let pos_ids = Tensor::arange(0u32, seq_len as u32, xs.device())?;
        let pos_embeds = self.pos_emb.embeddings().index_select(&pos_ids, 0)?;

        let x = tok_embeds.broadcast_add(&pos_embeds)?;
        let x = self.drop_emb.forward(&x, train)?;
        let x = self.trf_blocks.forward_t(&x, train)?;
        let x = self.final_norm.forward(&x)?;

        let logits = self.out_head.forward(&x)?;
        Ok(logits)
    }
}


/// Creates a new empty sequential layer.
pub fn seq_transformers() -> SequentialTransformers {
    SequentialTransformers::default()
}

/// Explicit sequential like type for TransformerBlock
///
/// NOTE: preivously used candle_nn::Sequential but this creates trait objects
/// which lose information on the concrete type. Downcasting to the concrete
/// type was proven difficult. Thus, opting for explicit sequence instead.
/// The type information is necessary when wanting to implement LoRA.
#[derive(Debug, Clone)]
pub struct SequentialTransformers {
    layers: Vec<TransformerBlock>,
}

impl SequentialTransformers {
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, layer: TransformerBlock) -> Self {
        self.layers.push(layer);
        self
    }

    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// The number of sub-layers embedded in this layer.
    pub fn len_i64(&self) -> i64 {
        self.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    // Accessor
    pub fn layers(&self) -> &[TransformerBlock] {
        &self.layers
    }
}

impl Default for SequentialTransformers {
    fn default() -> Self {
        SequentialTransformers { layers: vec![] }
    }
}

impl ModuleT for SequentialTransformers {
    fn forward_t(&self, xs: &Tensor, train: bool) -> candle_core::Result<Tensor> {
        let out = self
            .layers
            .iter()
            .try_fold(xs.to_owned(), |x, layer| layer.forward_t(&x, train))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::{get_batch_token_ids, get_vb};
    use super::*;

    #[test]
    fn test_gpt_model_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb)?;

        assert_eq!(model.pos_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.tok_emb.hidden_size(), cfg.emb_dim);
        assert_eq!(model.trf_blocks.len() as usize, cfg.n_layers);
        assert_eq!(
            model.out_head.weight().dims(),
            &[cfg.vocab_size, cfg.emb_dim]
        );
        Ok(())
    }

    #[test]
    fn test_gpt_model_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let batch_token_ids = get_batch_token_ids();
        let (batch_size, seq_len) = batch_token_ids.dims2()?;

        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb)?;

        let logits = model.forward(&batch_token_ids)?;

        assert_eq!(logits.dims(), &[batch_size, seq_len, cfg.vocab_size]);

        Ok(())
    }

}
