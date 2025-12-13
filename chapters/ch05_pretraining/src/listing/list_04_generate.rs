
use candle_core::{IndexOp, ModuleT, Tensor, D};
use candle_nn::ops::softmax;
use ch04_gpt_implementation::listing::list_07_gpt_model::GPT;
use candle_addons::top_k::TopK;

/// A modified text generation function with more diversity
pub fn generate<M>(
    model: &M,
    idx: Tensor,
    max_new_tokens: usize,
    context_size: usize,
    temperature: Option<f64>,
    top_k: Option<usize>,
    eos_id: Option<Tensor>,
    rng: &mut rand::rngs::StdRng,
) -> candle_core::Result<Tensor>
where
    M: GPT + ModuleT,
{
    let (idx, _): (Tensor, bool) = (0..max_new_tokens)
        .try_fold((idx.to_owned(), false), |(idx, done), _| -> candle_core::Result<(Tensor, bool)> {
            if done {
                // 已经全部生成到 EOS，只是把状态往后传
                return Ok((idx, true));
            }

            let (b, seq_len) = idx.dims2()?;
            let ctx_len = context_size.min(seq_len);
            let start_token_index = seq_len - ctx_len;

            let idx_cond = idx.i((.., start_token_index..seq_len))?;
            let logits = model.forward_t(&idx_cond, false)?;
            let (_b, c, _vocab_size) = logits.dims3()?;
            let logits = logits.i((.., c - 1, ..))?;

            let logits = if let Some(top_k) = top_k {
                let (top_logits, _top_pos) = logits.contiguous()?.topk_last_dim1(top_k)?;
                let min_top = top_logits.min_keepdim(D::Minus1)?;
                let mask = logits.broadcast_lt(&min_top)?;

                let neg_inf = Tensor::new(f32::NEG_INFINITY, logits.device())?;
                let on_true = logits.ones_like()?.broadcast_mul(&neg_inf)?;
                mask.where_cond(&on_true, &logits)?
            } else {
                logits
            };

            let idx_next = if let Some(temp) = temperature {
                let logits = (logits / temp)?;
                let probas = softmax(&logits, D::Minus1)?;

                let idx_next: Vec<u32> = (0..b)
                    .map(|bx| {
                        let this_probas = probas.i((bx, ..))?;
                        let prs = this_probas.to_vec1::<f32>()?;
                        sample_multinomial(rng, &prs)
                    })
                    .collect::<candle_core::Result<_>>()?;

                Tensor::from_vec(idx_next, (b, 1_usize), logits.device())?
            } else {
                let probas = softmax(&logits, D::Minus1)?;
                probas.argmax_keepdim(D::Minus1)?
            };

            let done = if let Some(ref eos) = eos_id {
                let num_eos = idx_next
                    .broadcast_eq(eos)?
                    .sum_all()?
                    .to_scalar::<u8>()? as usize;
                num_eos == b
            } else {
                false
            };

            // 与原逻辑一致：如果全是 EOS，不再拼接到序列尾部
            let idx = if done {
                idx
            } else {
                Tensor::cat(&[&idx, &idx_next], D::Minus1)?
            };

            Ok((idx, done))
        })?;

    Ok(idx)
}

/// Randomly draws a single observation from a multinomial distribution
///
/// NOTE: Can also use `candle_transformers::LogitProcessor`
pub fn sample_multinomial(rng: &mut rand::rngs::StdRng, prs: &Vec<f32>) -> candle_core::Result<u32> {
    use rand::distr::Distribution;

    let dist = rand::distr::weighted::WeightedIndex::new(prs).map_err(candle_core::Error::wrap)?;
    let sample = dist.sample(rng);
    Ok(sample as u32)
}


#[cfg(test)]
mod test {
    use rand::SeedableRng;
    use super::*;

    #[test]
    fn test_sample_multinomial() -> anyhow::Result<()> {
        let prs = vec![0f32, 1f32];
        let expected = 1u32;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234u64);
        let token = sample_multinomial(&mut rng, &prs)?;
        assert_eq!(token, expected);
        Ok(())
    }
}
