
use candle_core::{bail, Device, IndexOp, ModuleT, Tensor, D};
use tiktoken_rs::CoreBPE;
use ch04_gpt_implementation::listing::list_07_gpt_model::GPTModel;
use crate::listing::list_10_train_classifier::TextClassification;

/// Using the model to classify new texts
pub fn classify_review(
    text: &str,
    model: &GPTModel,
    tokenizer: &CoreBPE,
    device: &Device,
    max_len: Option<usize>,
    pad_token_id: u32,
) -> candle_core::Result<TextClassification> {
    let input_ids = tokenizer.encode_with_special_tokens(text);
    let supported_context_length = model.pos_emb().hidden_size();

    let upper = match max_len {
        None => supported_context_length,
        Some(m) => std::cmp::min(m, supported_context_length)
    };
    let mut input_ids = Vec::from_iter(input_ids.into_iter().take(supported_context_length as usize));

    // add padding if necessary
    let num_pad = upper.saturating_sub(input_ids.len());
    let padding = std::iter::repeat(pad_token_id)
        .take(num_pad)
        .collect::<Vec<u32>>();
    input_ids.extend(padding);

    // inference
    let input_tensor = Tensor::new(&input_ids[..], device)?.unsqueeze(0)?;
    let logits = model
        .forward_t(&input_tensor, false)?
        .i((.., input_ids.len() - 1, ..))?;
    let label = logits.argmax(D::Minus1)?.squeeze(0)?.to_scalar::<u32>()?;

    // return type
    match label {
        0 => Ok(TextClassification::Ham),
        1 => Ok(TextClassification::Spam),
        _ => bail!(
            "Unable to classify text as spam/ham. \
        Argmax op resulted in a value different from 0 and 1."
        ),
    }
}
