use candle_core::{Module, Tensor};
use candle_nn::{linear_b, Linear, VarBuilder};
use crate::listing::list_01_dummy_gpt_model::Config;
use crate::listing::list_03_gelu::GELU;

/// Explicit `FFLayer`` enum for FeedForward
#[derive(Debug, Clone)]
pub enum FFLayer {
    Linear(Linear),
    GELU(GELU),
}

impl Module for FFLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            FFLayer::Linear(l) => l.forward(xs),
            FFLayer::GELU(g) => g.forward(xs),
        }
    }
}

/// A feed forward neural network module
#[derive(Debug, Clone)]
pub struct FeedForward {
    layers: Vec<FFLayer>,
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.layers
            .iter()
            .try_fold(xs.to_owned(), |xs, layer| layer.forward(&xs))
    }
}

impl FeedForward {
    /// Creates a new `FeedForward` via a `Config`
    pub fn new(cfg: Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let layers = vec![
            FFLayer::Linear(linear_b(
                cfg.emb_dim,
                4usize * cfg.emb_dim,
                true,
                vb.pp("first_layer"),
            )?),
            FFLayer::GELU(GELU),
            FFLayer::Linear(linear_b(
                4 * cfg.emb_dim,
                cfg.emb_dim,
                true,
                vb.pp("second_layer"),
            )?)
        ];

        Ok(Self { layers })
    }

    pub fn from_fields(layers: Vec<FFLayer>) -> candle_core::Result<Self> {
        Ok(Self { layers })
    }

    pub fn layers(&self) -> &[FFLayer] {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_vb;
    use super::*;

    #[test]
    fn test_feed_forward_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let ff = FeedForward::new(Config::gpt_sm_test(), vb.pp("feed_forward"))?;

        assert_eq!(ff.layers.len(), 3usize);

        Ok(())
    }

    #[test]
    fn test_feed_forward_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let cfg = Config::gpt_sm_test();
        let ff = FeedForward::new(Config::gpt_sm_test(), vb.pp("feed_forward"))?;

        let (batch_size, seq_len) = (2usize, 3usize);
        let batch_example = Tensor::rand(0f32, 1f32, (batch_size, seq_len, cfg.emb_dim), vb.device())?;
        let out = ff.forward(&batch_example)?;

        assert_eq!(out.dims(), &[batch_size, seq_len, cfg.emb_dim]);

        Ok(())
    }
}
