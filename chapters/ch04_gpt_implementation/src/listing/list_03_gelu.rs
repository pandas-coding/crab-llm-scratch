use candle_core::{Module, Tensor};

#[derive(Debug, Clone)]
pub struct GELU;

impl Module for GELU {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        (0.5_f64 * xs)?.mul(
            &((2_f64 / std::f64::consts::PI).sqrt() * (xs + (xs.mul(xs)?.mul(xs)? * 0.044715f64)?)?)?
                .tanh()?
                .broadcast_add(&Tensor::ones((1,), candle_core::DType::F32, xs.device())?)?,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use candle_core::Device;
    use candle_nn::Activation;

    #[test]
    fn test_gelu_impl() -> anyhow::Result<()> {
        let device = Device::cuda_if_available(0)?;
        let batch_example = Tensor::rand(0f32, 1f32, (2usize, 3usize), &device)?;

        let gelu = GELU;
        let out = gelu.forward(&batch_example)?;

        let candle_gelu = Activation::Gelu;
        let candle_out = candle_gelu.forward(&batch_example)?;

        let tol: f64 = 1e-3;
        let abs_diff = (out - candle_out)?.abs()?;

        assert_eq!(
            abs_diff.lt(tol)?.sum_all()?.to_scalar::<u8>()?,
            (2usize * 3usize) as u8,
        );

        Ok(())
    }
}
