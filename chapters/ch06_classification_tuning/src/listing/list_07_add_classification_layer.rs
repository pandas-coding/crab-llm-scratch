use candle_nn::{linear_b, VarBuilder, VarMap};
use ch04_gpt_implementation::listing::list_01_dummy_gpt_model::Config;
use ch04_gpt_implementation::listing::list_07_gpt_model::GPTModel;

/// Adding a classification layer
pub fn modify_out_head_for_classification(
    model: &mut GPTModel,
    cfg: Config,
    num_classes: usize,
    varmap: &VarMap,
    vb: VarBuilder,
) -> candle_core::Result<()> {
    let mut tensor_data = varmap.data().lock().unwrap();
    let out_head_pp = format!("{}.out_head.weight", vb.prefix());
    if tensor_data.remove(&out_head_pp[..]).is_none() { 
        candle_core::bail!("Error when removing old head from VarMap.")
    };
    drop(tensor_data);
    
    let out_head = linear_b(cfg.emb_dim, num_classes, true, vb.pp("out_head"))?;
    model.set_out_head(out_head);
    Ok(())
}

#[cfg(test)]
mod tests {
    use candle_core::{DType, Device};
    use super::*;
    
    #[test]
    fn test_modify_out_head_for_classification() -> anyhow::Result<()> {
        // create typical language task model
        let dev = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let cfg = Config::gpt_sm_test();
        let mut model = GPTModel::new(cfg, vb.pp("model"))?;

        // change to classification head
        let num_classes = 2_usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        assert_eq!(
            model.out_head().weight().dims(),
            &[num_classes, cfg.emb_dim]
        );
        Ok(())
    }
}