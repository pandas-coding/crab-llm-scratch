use crate::listing::list_03_gelu::GELU;
use candle_core::{IndexOp, Module, Tensor, TensorId, bail};
use candle_nn::{Sequential, VarBuilder, linear_b, seq, Linear};

/// A neural network to illustrate shortcut connections
pub struct ExampleDeepNeuralNetwork {
    use_shortcut: bool,
    pub layers: Vec<Sequential>,
    pub tensor_ids: Vec<TensorId>,
}

impl Module for ExampleDeepNeuralNetwork {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.layers.iter().try_fold(xs.to_owned(), |x, layer| {
            let layer_forward = layer.forward(&x)?;
            let use_shortcut = self.use_shortcut && xs.dims() == layer_forward.dims();
            let out = if use_shortcut {
                layer_forward.add(&x)?
            } else {
                layer_forward
            };
            Ok(out)
        })
    }
}

impl ExampleDeepNeuralNetwork {
    pub fn new(
        layer_sizes: &[usize],
        use_shortcut: bool,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        if layer_sizes.len() < 2 {
            bail!("inputs has to be at least two elements");
        }

        let (layers, tensor_ids): (Vec<Sequential>, Vec<TensorId>) = layer_sizes
            .windows(2)
            .enumerate()
            .map(|(i, pair)| {
                let &[in_dim, out_dim] = pair else {
                    unreachable!("layer_sizes.windows(2) cannot get pair length shorter than 2");
                };
                let linear_biased = linear_b(
                    in_dim,
                    out_dim,
                    true,
                    vb.pp(format!("layer-{i}"))
                );
                linear_biased.map(|linear: Linear| {
                    let tensor_id = linear.weight().id();
                    let layer = seq().add(linear).add(GELU);
                    (layer, tensor_id)
                })
            })
            .collect::<candle_core::Result<Vec<(Sequential, TensorId)>>>()?
            .into_iter()
            .unzip();

        Ok(Self {
            use_shortcut,
            layers,
            tensor_ids,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::test_utils::get_vb;
    use super::*;

    #[test]
    fn test_example_deep_neural_network_init() -> anyhow::Result<()> {
        let vb = get_vb();
        let layer_sizes = &[3usize, 2, 2, 1];
        let model = ExampleDeepNeuralNetwork::new(layer_sizes, true, vb)?;

        assert_eq!(model.layers.len(), layer_sizes.len() - 1usize);
        assert_eq!(model.use_shortcut, true);
        Ok(())
    }

    #[test]
    fn test_example_deep_neural_network_forward() -> anyhow::Result<()> {
        let vb = get_vb();
        let layer_sizes = &[3usize, 2, 2, 1];
        let model = ExampleDeepNeuralNetwork::new(layer_sizes, true, vb.pp("model"))?;
        let sample_input = Tensor::new(&[[1f32, 0., 1.], [0., 1., 0.]], &vb.device())?;
        let output = model.forward(&sample_input)?;

        assert_eq!(output.dims(), &[2usize, 1usize]);
        Ok(())

    }

}
