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
    use tiktoken_rs::get_bpe_from_model;
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

    #[test]
    #[should_panic(expected = "called `Result::unwrap()` on an `Err` value: Unable to decode into a valid UTF-8 string: incomplete utf-8 byte sequence from index 0")]
    fn test_decode_panics_due_token_ids() {
        let bad_token_id = 49426_u32; // not sure why this results in an error when decoding
        let token_ids = Tensor::new(&[[bad_token_id]], &Device::Cpu).unwrap();
        let tokenizer = get_bpe_from_model("gpt2").unwrap();
        token_ids_to_text(token_ids, &tokenizer).unwrap();
    }
}
