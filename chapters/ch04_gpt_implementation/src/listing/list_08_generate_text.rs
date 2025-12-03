use crate::listing::list_07_gpt_model::GPT;
use candle_core::{D, IndexOp, ModuleT, Tensor};
use candle_nn::ops::softmax;

/// A function for the GPT model to generate text
pub fn generate_text_simple<M>(
    model: &M,
    idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
) -> candle_core::Result<Tensor>
where
    M: GPT + ModuleT,
{
    let idx = (0..max_new_tokens).into_iter().try_fold(
        idx.to_owned(),
        |x, _| -> candle_core::Result<Tensor> {
            let (_b, seq_len) = x.dims2()?;
            // using saturating_sub to avoid negative result when seq_len < context_size.
            let start_token_index = seq_len.saturating_sub(context_size);
            let idx_cond = x.i((.., start_token_index..seq_len))?;
            let logits = model.forward_t(&idx_cond, false)?;
            let (_b, c, _vocab_size) = logits.dims3()?;
            let logits = logits.i((.., c - 1, ..))?;
            let probas = softmax(&logits, 1)?;
            let idx_next = probas.argmax_keepdim(D::Minus1)?;
            let xs_out = Tensor::cat(&[&x, &idx_next], D::Minus1)?;
            Ok(xs_out)
        },
    )?;
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::listing::list_01_dummy_gpt_model::Config;
    use crate::listing::list_07_gpt_model::GPTModel;
    use crate::test_utils::{get_batch_token_ids, get_vb};

    #[test]
    fn test_generate_text_simple() -> anyhow::Result<()> {
        let vb = get_vb();
        let batch_token_ids = get_batch_token_ids();
        let cfg = Config::gpt_sm_test();
        let model = GPTModel::new(cfg, vb)?;

        // create sample idx
        let (batch_size, seq_len) = batch_token_ids.dims2()?;
        let (context_size, max_new_tokens) = (2usize, 3usize);
        let idx = generate_text_simple(&model, batch_token_ids, max_new_tokens, context_size)?;

        assert_eq!(idx.dims(), &[batch_size, seq_len + max_new_tokens]);
        Ok(())
    }
}
