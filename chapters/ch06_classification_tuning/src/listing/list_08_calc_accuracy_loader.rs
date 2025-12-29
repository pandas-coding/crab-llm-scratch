use std::cmp;
use candle_core::{DType, Device, IndexOp, ModuleT, Tensor, D};
use ch04_gpt_implementation::listing::list_07_gpt_model::GPT;
use crate::listing::list_05_spam_data_loader::SpamDataLoader;

/// Calculating the classification accuracy
pub fn calc_accuracy_loader<M>(
    data_loader: &SpamDataLoader,
    model: &M,
    device: &Device,
    num_batches: Option<usize>,
    custom_pred_token_index: Option<usize>,
) -> candle_core::Result<f32>
where M: GPT + ModuleT
{
    let n_batches = match num_batches {
        None => data_loader.len(),
        Some(n) => cmp::min(n, data_loader.len()),
    };

    let mut correct_predictions = 0_usize;
    let mut num_examples = 0_usize;
    let mut data_batcher = data_loader.batcher();

    for _ in 0..n_batches {
        let (input_batch, target_batch) = data_batcher.next().unwrap()?;
        let num_correct = calc_num_correct_batch(
            &input_batch,
            &target_batch,
            model,
            device,
            custom_pred_token_index,
        )?;
        correct_predictions += num_correct.to_scalar::<u8>()? as usize;
        num_examples += input_batch.dims()[0];
    }

    Ok(correct_predictions as f32 / num_examples as f32)

}

/// Calculate the number of correct predictions of a given batch
pub fn calc_num_correct_batch<M: GPT + ModuleT>(
    input_batch: &Tensor,
    target_batch: &Tensor,
    model: &M,
    device: &Device,
    custom_pred_token_index: Option<usize>,
) -> candle_core::Result<Tensor> {
    let input_batch = input_batch.to_device(device)?;
    let target_batch = target_batch.to_device(device)?.to_dtype(DType::U32)?;
    let outputs = model.forward_t(&input_batch, false)?;
    let (_b, c, _num_classes) = outputs.dims3()?;
    let pred_token_index = match custom_pred_token_index {
        None => c - 1,
        Some(ix) => ix,
    };
    let logits = outputs.i((.., pred_token_index, ..))?;
    let predicted_labels = logits.argmax_keepdim(D::Minus1)?;
    let num_correct = predicted_labels.eq(&target_batch)?.sum_all()?;
    Ok(num_correct)
}

#[cfg(test)]
mod tests {
    use candle_nn::{VarBuilder, VarMap};
    use ch04_gpt_implementation::listing::list_01_dummy_gpt_model::Config;
    use ch04_gpt_implementation::listing::list_07_gpt_model::GPTModel;
    use crate::listing::list_07_add_classification_layer::modify_out_head_for_classification;
    use super::*;

    #[test]
    fn test_calc_num_correct_batch() -> anyhow::Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt_sm_test();
        let mut model = GPTModel::new(cfg, vb.pp("model"))?;

        // change to classification head
        let num_classes = 2usize;
        modify_out_head_for_classification(&mut model, cfg, num_classes, &varmap, vb.pp("model"))?;

        // create sample inputs
        let inputs = Tensor::new(
            &[
                [100u32, 20, 300],
                [400, 7, 88],
            ],
            vb.device(),
        )?;
        let targets = Tensor::new(
            &[[1i64], [0]],
            vb.device(),
        )?;

        // compute num correct
        let num_correct = calc_num_correct_batch(&inputs, &targets, &model, vb.device(), None)?;

        assert_eq!(num_correct.elem_count(), 1);
        Ok(())
    }
}
