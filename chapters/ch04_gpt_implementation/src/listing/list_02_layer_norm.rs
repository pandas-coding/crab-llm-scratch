use candle_core::{Module, Tensor, D};
use candle_nn::VarBuilder;

const EPS: f32 = 1e-5;

/// A layer normalization struct
#[derive(Debug, Clone)]
pub struct LayerNorm {
    eps: f32,
    scale: Tensor,
    shift: Tensor,
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let var = xs.var_keepdim(D::Minus1)?;
        let norm_xs = xs.broadcast_sub(&mean)?.broadcast_div(
            &(var.broadcast_add(&Tensor::new(&[self.eps], xs.device())?)?).sqrt()?,
        )?;
        let out_norm = norm_xs
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.shift)?;
        Ok(out_norm)
    }
}

impl LayerNorm {
    /// Creates a new `LayerNorm`
    pub fn new(emb_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let scale = vb.get_with_hints(emb_dim, "scale", candle_nn::Init::Const(1f64))?;
        let shift = vb.get_with_hints(emb_dim, "shift", candle_nn::Init::Const(0f64))?;
        Ok(Self {
            eps: EPS,
            scale,
            shift,
        })
    }
    
    pub fn scale(&self) -> &Tensor {
        &self.scale
    }
    
    pub fn shift(&self) -> &Tensor {
        &self.shift
    }
}

#[cfg(test)]
mod tests {
    use candle_core::IndexOp;
    use crate::listing::list_01_dummy_gpt_model::Config;
    use crate::test_utils::get_vb;
    use super::*;

    #[test]
    fn test_layer_norm_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let cfg = Config::gpt_sm_test();
        let layer_norm = LayerNorm::new(cfg.emb_dim, vb)?;
        assert_eq!(layer_norm.eps, EPS);
        assert_eq!(layer_norm.scale.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.shift.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.scale.i(..=1)?.to_vec1::<f32>()?, &[1., 1.]);
        assert_eq!(layer_norm.shift.i(..=1)?.to_vec1::<f32>()?, &[0., 0.]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let cfg = Config::gpt_sm_test();
        let layer_norm = LayerNorm::new(cfg.emb_dim, vb)?;

        assert_eq!(layer_norm.eps, EPS);
        assert_eq!(layer_norm.scale.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.shift.dims(), &[cfg.emb_dim]);
        assert_eq!(layer_norm.scale.i(..=1)?.to_vec1::<f32>()?, &[1., 1.]);
        assert_eq!(layer_norm.shift.i(..=1)?.to_vec1::<f32>()?, &[0., 0.]);
        Ok(())
    }
}
