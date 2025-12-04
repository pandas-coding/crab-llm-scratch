use candle_core::{Device, Tensor};
use std::collections::HashSet;

/// Utility function for text to token ID conversion
pub fn text_to_token_ids(text: &str, tokenizer: &tiktoken_rs::CoreBPE, device: &Device) -> candle_core::Result<Tensor> {
    let allowed_special = HashSet::from(["<|endoftext|>"]);
    let (encoded, _last_piece_token_len) = tokenizer.encode(text, &allowed_special);
    let num_tokens = encoded.len();
    Tensor::from_vec(encoded, (1usize, num_tokens), device)
}

/// Utility function for token ID to text ID conversion
pub fn token_ids_to_text(token_ids: Tensor, tokenizer: &tiktoken_rs::CoreBPE) -> anyhow::Result<String> {
    let flat = token_ids.squeeze(0)?;
    tokenizer.decode(flat.to_vec1::<u32>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::get_txt_tokenizer;

    #[test]
    fn test_text_to_token_ids_and_back_to_text() -> anyhow::Result<()> {
        let (text, tokenizer) = get_txt_tokenizer();
        let token_ids = text_to_token_ids(&text[..], &tokenizer, &Device::Cpu)?;
        let decoded_text = token_ids_to_text(token_ids, &tokenizer)?;
        assert_eq!(decoded_text, text);

        Ok(())
    }
}
