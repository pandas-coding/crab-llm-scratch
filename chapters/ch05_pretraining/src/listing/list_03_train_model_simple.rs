use candle_core::{Device, ModuleT, Tensor};
use candle_nn::Optimizer;
use tiktoken_rs::CoreBPE;
use ch02_text_processing::listing::list_06_dataloader_v1::DataLoader;
use ch04_gpt_implementation::listing::list_07_gpt_model::GPT;
use ch04_gpt_implementation::listing::list_08_generate_text::generate_text_simple;
use crate::listing::list_01_text_to_token::{text_to_token_ids, token_ids_to_text};
use crate::listing::list_02_calc_loss::{calc_loss_batch, calc_loss_loader};

/// The main function for pretraining LLMs
#[allow(clippy::too_many_arguments)]
pub fn train_model_simple<T, L, M>(
    model: &M,
    train_loader: &L,
    val_loader: &L,
    mut optimizer: T,
    device: &Device,
    num_epochs: usize,
    eval_freq: usize,
    eval_iter: usize,
    start_context: &str,
    tokenizer: &CoreBPE,
    ignore_index: Option<i64>, // introduced for ch07 instruction finetuning
) -> candle_core::Result<(Vec<f32>, Vec<f32>, Vec<usize>)>
where
    T: Optimizer,
    M: GPT + ModuleT,
    L: DataLoader,
    L::Batcher: Iterator<Item = candle_core::Result<(Tensor, Tensor)>>,
{
    // retvals
    let mut train_losses: Vec<f32> = vec![];
    let mut val_losses: Vec<f32> = vec![];
    let mut track_tokens_seen: Vec<usize> = vec![];

    let (mut tokens_seen, mut global_step) = (0usize, 0_usize);

    for epoch in 0..num_epochs {
        let mut train_batcher = train_loader.batcher();
        while let Some(Ok((input_batch, target_batch))) = train_batcher.next() {
            let loss = calc_loss_batch(
                &input_batch,
                &target_batch,
                model,
                device,
                true,
                ignore_index,
            )?;
            optimizer.backward_step(&loss)?;
            tokens_seen += input_batch.elem_count();

            if global_step % eval_freq == 0 {
                let (train_loss, val_loss) = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                    ignore_index,
                )?;
                train_losses.push(train_loss);
                val_losses.push(val_loss);
                track_tokens_seen.push(tokens_seen);
                println!(
                    "Ep {} (Step {}) \
                    Train loss: {}, \
                    Val loss: {}",
                    epoch + 1,
                    global_step,
                    train_loss,
                    val_loss
                );
            }
            global_step += 1;
        }
        generate_and_print_sample(model, tokenizer, device, start_context)?
    }

    Ok((train_losses, val_losses, track_tokens_seen))
}

/// Returns train and validation loss of a `GPTModel`
pub fn evaluate_model<M, L>(
    model: &M,
    train_loader: &L,
    val_loader: &L,
    device: &Device,
    eval_iter: usize,
    ignore_index: Option<i64>,
) -> candle_core::Result<(f32, f32)>
where
    M: GPT + ModuleT,
    L: DataLoader,
    L::Batcher: Iterator<Item = candle_core::Result<(Tensor, Tensor)>>,
{
    let train_loss = calc_loss_loader(train_loader, model, device, Some(eval_iter), ignore_index)?;
    let val_loss = calc_loss_loader(val_loader, model, device, Some(eval_iter), ignore_index)?;
    Ok((train_loss, val_loss))
}

/// Print a generation sample of model
///
/// This is a convenience function used for qualitative assessment of a model
/// during training.
pub fn generate_and_print_sample<M: GPT + ModuleT>(
    model: &M,
    tokenizer: &CoreBPE,
    device: &Device,
    start_context: &str,
) -> candle_core::Result<()> {
    let context_size = model.context_size();
    let encoded = text_to_token_ids(start_context, tokenizer, device)?;
    let token_ids = generate_text_simple(model, encoded, 50, context_size)?;
    let decoded_text = token_ids_to_text(token_ids, tokenizer).expect("convert token_ids to text error");
    println!("{}", decoded_text.replace("\n", " "));
    Ok(())
}

