use std::cmp;
use candle_core::{Device, ModuleT, Tensor, D};
use ch04_gpt_implementation::listing::list_07_gpt_model::GPT;
use crate::listing::list_05_spam_data_loader::SpamDataLoader;

/// Function to compute the training and validation cross-entropy loss
pub fn calc_loss_loader<M: GPT + ModuleT>(
    data_loader: &SpamDataLoader,
    model: &M,
    device: &Device,
    num_batches: Option<usize>,
    custom_pred_token_index: Option<usize>,
) -> candle_core::Result<f32> {
    let mut data_batcher = data_loader.batcher();
    let n_batches = num_batches
        .map(|n| n.min(data_loader.len()))
        .unwrap_or_else(|| data_loader.len());

    let total_loss = (0..n_batches).try_fold(0f32, |acc, _| {
        // data_batcher.next() : Option<Result<(Input, Target), E>>
        let (input_batch, target_batch) = data_batcher
            .next()
            .transpose()? // Option<Result<T, E>> -> Result<Option<T>, E>
            .ok_or_else(|| candle_core::Error::Msg("unexpected end of dataloader".into()))?;

        let loss = calc_loss_batch(
            &input_batch,
            &target_batch,
            model,
            device,
            custom_pred_token_index,
        )?;

        loss.to_scalar::<f32>().map(|l| acc + l)
    })?;

    Ok(total_loss / n_batches as f32)
}

/// Calculate the cross entropy loss of a given batch
pub fn calc_loss_batch<M: GPT + ModuleT>(
    input_batch: &Tensor,
    target_batch: &Tensor,
    model: &M,
    device: &Device,
    custom_pred_token_index: Option<usize>,
) -> candle_core::Result<Tensor> {
    let input_batch = input_batch.to_device(device)?;
    let target_batch = target_batch.to_device(device)?;
    let outputs = model.forward_t(&input_batch, true)?;
    let (_b, c, _num_classes) = outputs.dims3()?;
    let pred_token_index = match custom_pred_token_index {
        None => c - 1,
        Some(ix) => ix,
    };

    let logits = outputs.index_select(&Tensor::new(&[pred_token_index as u32], device)?, D::Minus2)?;

    // flatten
    let logits_flat = logits.flatten(0, 1)?;
    let targets_flat = target_batch.flatten_all()?;

    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};
    use ch04_gpt_implementation::listing::list_01_dummy_gpt_model::Config;
    use ch04_gpt_implementation::listing::list_07_gpt_model::GPTModel;
    use crate::listing::list_07_add_classification_layer::modify_out_head_for_classification;
    use super::*;

    #[test]
    fn test_calc_loss_batch() -> anyhow::Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt_sm_test();
        let mut model = GPTModel::new(cfg, vb.pp("model"))?;

        let num_classes = 2usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // create sample inputs
        let inputs = Tensor::new(&[[100_u32, 20, 300], [400, 7, 88]], vb.device())?;
        let targets = Tensor::new(&[[1_i64], [0]], vb.device())?;
        let loss = calc_loss_batch(&inputs, &targets, &model, vb.device(), None)?;

        assert_eq!(loss.elem_count(), 1);
        Ok(())
    }
}