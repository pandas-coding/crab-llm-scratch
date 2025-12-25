use std::rc::Rc;
use anyhow::anyhow;
use polars::datatypes::AnyValue;
use polars::prelude::DataFrame;

pub const PAD_TOKEN_ID: u32 = 50_256_u32;

/// Setting up a `SpamDataset` struct
pub struct SpamDataset {
    data: Rc<DataFrame>,
    encoded_texts: Rc<Vec<Vec<u32>>>,
    max_len: usize,
    pad_token_id: u32,
}

impl AsRef<SpamDataset> for SpamDataset {
    fn as_ref(&self) -> &SpamDataset {
        self
    }
}

impl SpamDataset {
    /// Creates a new `SpamDataset`.
    pub fn new(
        df: DataFrame,
        tokenizer: &tiktoken_rs::CoreBPE,
        max_len: Option<usize>,
        pad_token_id: u32,
    ) -> Self {
        let text_series = df
            .column("sms")
            .expect("column `sms` not found")
            .str()
            .expect("column `sms` must be utf8");

        let encodings: Vec<Vec<u32>> = text_series
            .into_iter()
            .map(|opt_text| {
                let text = opt_text.expect("sms text must not be null");
                tokenizer.encode_with_special_tokens(text)
            })
            .collect();

        let raw_max_len =
            Self::get_raw_max_len(&encodings).expect("failed to compute max length");
        let max_len = max_len.unwrap_or(raw_max_len);

        let encodings: Vec<Vec<u32>> = encodings
            .into_iter()
            .map(|mut v| {
                v.truncate(max_len);
                let num_pad = max_len.saturating_sub(v.len());
                if num_pad != 0 {
                    v.extend(std::iter::repeat(pad_token_id).take(num_pad));
                }
                v
            })
            .collect();

        Self {
            data: Rc::new(df),
            encoded_texts: Rc::new(encodings),
            max_len,
            pad_token_id,
        }
    }

    /// Gets the number of finetuning examples.
    pub fn len(&self) -> usize {
        self.data.shape().0
    }

    /// Checks whether the dataset is empty or has no finetuning examples.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the input tokens for all input sequences.
    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Returns the input-target pair at the specified index.
    pub fn get_item_at_index(&self, index: usize) -> anyhow::Result<(&Vec<u32>, Vec<i64>)> {
        let encoded = &self.encoded_texts[index];
        let binding = self.data.select(["label"])?;
        let label = match &binding.get_row(index)?.0[0] {
            AnyValue::Int64(label_value) => Ok(label_value),
            _ => Err(anyhow!(
                "There was a problem in getting the Label from the dataframe."
            )),
        }?
            .to_owned();

        Ok((encoded, vec![label]))
    }

    fn get_raw_max_len(encodings: &[Vec<u32>]) -> anyhow::Result<usize> {
        encodings
            .iter()
            .map(|v| v.len())
            .max()
            .ok_or_else(|| anyhow!("Error when computing max length encodings"))
    }
}

#[cfg(test)]
mod tests {
    use tiktoken_rs::get_bpe_from_model;
    use crate::test_utils::sms_spam_df;
    use super::*;

    #[test]
    pub fn test_spam_dataset_init() -> anyhow::Result<()> {
        let (df, _num_spam) = sms_spam_df();
        let max_len = Some(10usize);
        let expected_max_len = 10usize;

        let tokenizer = get_bpe_from_model("gpt2")?;
        let spam_dataset = SpamDataset::new(df, &tokenizer, max_len, PAD_TOKEN_ID);

        assert_eq!(spam_dataset.len(), 5);
        assert_eq!(spam_dataset.max_len(), expected_max_len);

        // assert all encoded texts have length == max_length
        for text_enc in spam_dataset.encoded_texts.iter() {
            assert_eq!(text_enc.len(), expected_max_len);
        }

        Ok(())
    }
}
