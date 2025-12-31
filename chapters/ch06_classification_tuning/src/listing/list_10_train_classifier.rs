use std::fmt::Display;
use crate::listing::list_05_spam_data_loader::SpamDataLoader;
use crate::listing::list_08_calc_accuracy_loader::calc_accuracy_loader;
use crate::listing::list_09_calc_loss_loader::{calc_loss_batch, calc_loss_loader};
use candle_core::{Device, ModuleT};
use candle_nn::Optimizer;
use ch04_gpt_implementation::listing::list_07_gpt_model::GPT;

type ClassifierTrainingResult = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, usize);

/// Fine-tuning the model to classify spam
pub fn train_classifier_simple<T: Optimizer, M: GPT + ModuleT>(
    model: &M,
    train_loader: &SpamDataLoader,
    val_loader: &SpamDataLoader,
    mut optimizer: T,
    device: &Device,
    num_epochs: usize,
    eval_freq: usize,
    eval_iter: usize,
    custom_pred_token_index: Option<usize>,
) -> candle_core::Result<ClassifierTrainingResult> {
    // Return values collected across training.
    let mut train_losses: Vec<f32> = Vec::new();
    let mut val_losses: Vec<f32> = Vec::new();
    let mut train_accs: Vec<f32> = Vec::new();
    let mut val_accs: Vec<f32> = Vec::new();

    let mut examples_seen = 0usize;
    let mut global_step = 0usize;

    let accuracy = |loader: &SpamDataLoader| {
        calc_accuracy_loader(
            loader,
            model,
            device,
            Some(eval_iter),
            custom_pred_token_index,
        )
    };

    for epoch in 1..=num_epochs {
        let mut train_batcher = train_loader.batcher();
        while let Some((input_batch, target_batch)) = train_batcher.next().transpose()? {
            let loss = calc_loss_batch(
                &input_batch,
                &target_batch,
                model,
                device,
                custom_pred_token_index,
            )?;
            optimizer.backward_step(&loss)?;
            let (batch_size, _) = input_batch.dims2()?;
            examples_seen += batch_size;

            if global_step % eval_freq == 0 {
                let (train_loss, val_loss) = evaluate_model(
                    model,
                    train_loader,
                    val_loader,
                    device,
                    eval_iter,
                    custom_pred_token_index,
                )?;
                train_losses.push(train_loss);
                val_losses.push(val_loss);
                println!(
                    "Ep {epoch} (Step {global_step}) Train loss: {train_loss}, Val loss: {val_loss}",
                );
            }
            global_step += 1;
        }

        let train_accuracy = accuracy(train_loader)?;
        let val_accuracy = accuracy(val_loader)?;
        println!("Training accuracy: {}", train_accuracy);
        println!("Validation accuracy: {}", val_accuracy);
        train_accs.push(train_accuracy);
        val_accs.push(val_accuracy);
    }

    Ok((
        train_losses,
        val_losses,
        train_accs,
        val_accs,
        examples_seen,
    ))
}

/// Returns train and validation loss of a `GPTModel` for spam classification
pub fn evaluate_model<M: GPT + ModuleT>(
    model: &M,
    train_loader: &SpamDataLoader,
    val_loader: &SpamDataLoader,
    device: &Device,
    eval_iter: usize,
    custom_pred_token_index: Option<usize>,
) -> candle_core::Result<(f32, f32)> {
    let train_loss = calc_loss_loader(
        train_loader,
        model,
        device,
        Some(eval_iter),
        custom_pred_token_index,
    )?;
    let val_loss = calc_loss_loader(
        val_loader,
        model,
        device,
        Some(eval_iter),
        custom_pred_token_index,
    )?;
    Ok((train_loss, val_loss))
}

#[derive(Debug)]
pub enum TextClassification {
    Spam,
    Ham,
}

impl Display for TextClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TextClassification::Spam => write!(f, "spam"),
            TextClassification::Ham => write!(f, "ham"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_classification_enum() -> anyhow::Result<()> {
        let ham = TextClassification::Ham;
        let spam = TextClassification::Spam;

        assert_eq!(format!("{} is not {}", ham, spam), "ham is not spam");
        Ok(())
    }
}
