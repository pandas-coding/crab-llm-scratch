use candle_core::{Device, ModuleT, Tensor};
use ch02_text_processing::listing::list_06_dataloader_v1::DataLoader;
use ch04_gpt_implementation::listing::list_07_gpt_model::GPT;

pub const DEFAULT_IGNORE_INDEX: i64 = -100;

/// Function to compute the training and validation loss
pub fn calc_loss_loader<M, L>(
    data_loader: &L,
    model: &M,
    device: &Device,
    num_batches: Option<usize>,
    ignore_index: Option<i64>,
) -> candle_core::Result<f32>
where
    L: DataLoader,
    L::Batcher: Iterator<Item = candle_core::Result<(Tensor, Tensor)>>,
    M: ModuleT + GPT,
{
    let mut total_loss = 0_f32;
    let mut count = 0_usize;
    let max_batches = num_batches.unwrap_or(usize::MAX);

    let mut data_batcher = data_loader.batcher();
    while let Some(Ok((input_batch, target_batch))) = data_batcher.next() {
        if count >= max_batches {
            break;
        }

        let loss = calc_loss_batch(
            &input_batch,
            &target_batch,
            model,
            device,
            false,
            ignore_index,
        )?;
        total_loss += loss.to_scalar::<f32>()?;
        count += 1_usize;
    }

    Ok(total_loss / count as f32)
}

/// Calculate the cross entropy loss of a given batch
pub fn calc_loss_batch<M>(
    input_batch: &Tensor,
    target_batch: &Tensor,
    model: &M,
    device: &Device,
    train: bool,
    ignore_index: Option<i64>,
) -> candle_core::Result<Tensor>
where
    M: ModuleT + GPT,
{
    let input_batch = input_batch.to_device(device)?;
    let target_batch = target_batch.to_device(device)?;
    let logits = model.forward_t(&input_batch, train)?;

    let logits_flat = logits.flatten(0, 1)?;
    let targets_flat = target_batch.flatten_all()?;

    let (logits_flat, targets_flat) = if let Some(ignore_val) = ignore_index {
        let keep = targets_flat
            .to_vec1::<i64>()?
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != ignore_val)
            .map(|(ix, _)| ix as u32)
            .collect::<Vec<_>>();
        let keep = Tensor::new(&keep[..], device)?;
        (
            logits_flat.index_select(&keep, 0)?,
            targets_flat.index_select(&keep, 0)?,
        )
    } else {
        (logits_flat, targets_flat)
    };

    let loss = candle_nn::loss::cross_entropy(&logits_flat, &targets_flat)?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};
    use ch04_gpt_implementation::listing::list_01_dummy_gpt_model::Config;
    use ch04_gpt_implementation::listing::list_07_gpt_model::GPTModel;
    use super::*;

    #[test]
    fn test_calc_loss_batch() -> anyhow::Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        let inputs = Tensor::new(&[[100u32, 20, 300], [400, 7, 88]], &vb.device())?;
        let targets = Tensor::new(&[[1u32, 2, 3], [4, 5, 9]], &vb.device())?;
        let loss = calc_loss_batch(&inputs, &targets, &model, &vb.device(), false, None)?;

        assert_eq!(loss.elem_count(), 1);
        Ok(())
    }

    #[test]
    fn test_calc_loss_batch_with_ignore_index() -> anyhow::Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::cuda_if_available(0)?);
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb.pp("model"))?;

        let inputs = Tensor::new(&[[100u32, 20, 300]], vb.device())?;
        let targets = Tensor::new(&[[1u32, 2, 3]], &vb.device())?;
        let loss = calc_loss_batch(&inputs, &targets, &model, &vb.device(), false, None)?;

        let inputs_2 = Tensor::new(&[[100u32, 20, 300], [400, 7, 88]], &vb.device())?;
        let targets_2 = Tensor::new(&[
            [1i64, 2, 3],
            [
                DEFAULT_IGNORE_INDEX,
                DEFAULT_IGNORE_INDEX,
                DEFAULT_IGNORE_INDEX,
            ],
        ],
        &vb.device())?;

        let loss_2 = calc_loss_batch(
            &inputs_2,
            &targets_2,
            &model,
            &vb.device(),
            false,
            Some(DEFAULT_IGNORE_INDEX),
        )?;

        assert_eq!(loss.to_scalar::<f32>()?, loss_2.to_scalar::<f32>()?);
        Ok(())
    }
}
