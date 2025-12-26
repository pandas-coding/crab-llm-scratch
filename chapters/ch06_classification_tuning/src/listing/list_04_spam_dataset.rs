use std::fs::File;
use std::path::Path;
use std::rc::Rc;
use anyhow::anyhow;
use candle_core::{Device, Tensor};
use candle_datasets::Batcher;
use candle_datasets::batcher::IterResult2;
use polars::datatypes::AnyValue;
use polars::prelude::{DataFrame, ParquetReader, SerReader};
use rand::prelude::SliceRandom;
use rand::rng;

pub const PAD_TOKEN_ID: u32 = 50_256_u32;

/// Setting up a `SpamDataset` struct
#[derive(Clone)]
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

/// Builder pattern for `HuggingFaceWeight`
pub struct SpamDatasetBuilder<'a> {
    data: Option<DataFrame>,
    max_len: Option<usize>,
    pad_token_id: u32,
    tokenizer: &'a tiktoken_rs::CoreBPE,
}

impl<'a> SpamDatasetBuilder<'a> {
    /// Creates a new `SpamDatasetBuilder`.
    pub fn new(tokenizer: &'a tiktoken_rs::CoreBPE) -> Self {
        Self {
            data: None,
            max_len: None,
            pad_token_id: PAD_TOKEN_ID,
            tokenizer,
        }
    }

    /// Set data for builder from parquet file.
    pub fn load_data_from_request<P: AsRef<Path>>(mut self, parquet_file: P) -> Self {
        let mut file = File::open(parquet_file).unwrap();
        let df = ParquetReader::new(&mut file).finish().unwrap();
        self.data = Some(df);
        self
    }

    pub fn data(mut self, data: DataFrame) -> Self {
        self.data = Some(data);
        self
    }

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    pub fn pad_token_id(mut self, pad_token_id: u32) -> Self {
        self.pad_token_id = pad_token_id;
        self
    }

    pub fn build(self) -> SpamDataset {
        match self.data {
            None => panic!("DataFrame is not set in SpamDataBuilder."),
            Some(df) => {
                SpamDataset::new(df, self.tokenizer, self.max_len, self.pad_token_id)
            }
        }
    }
}

pub struct SpamDatasetIter {
    dataset: SpamDataset,
    remaining_indices: Vec<usize>,
}

impl Iterator for SpamDatasetIter {
    type Item = candle_core::Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.remaining_indices.pop() {
            None => None,
            Some(idx) => {
                let (encoded, label) = self.dataset.get_item_at_index(idx).unwrap();

                // turn into tensor
                let device = Device::cuda_if_available(0).unwrap();
                let encoded_tensor = Tensor::new(&encoded[..], &device);
                let label_tensor = Tensor::new(&label[..], &device);
                Some(candle_core::error::zip(encoded_tensor, label_tensor))
            },
        }
    }
}

impl SpamDatasetIter {
    /// Creates a new `SpamDatasetIter`.
    ///
    /// ```rust
    /// use llms_from_scratch_rs::listings::ch06::{SpamDataset, SpamDatasetIter, PAD_TOKEN_ID};
    /// use polars::prelude::*;
    /// use tiktoken_rs::get_bpe_from_model;
    ///
    /// let df = df!(
    ///     "sms"=> &[
    ///         "Mock example 1",
    ///         "Mock example 2"
    ///     ],
    ///     "label"=> &[0_i64, 1],
    /// )
    /// .unwrap();
    /// let tokenizer = get_bpe_from_model("gpt2").unwrap();
    /// let max_length = 24_usize;
    /// let dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
    /// let iter = SpamDatasetIter::new(dataset.clone(), false);
    /// ```
    pub fn new(dataset: SpamDataset, shuffle: bool) -> Self {
        let mut remaining_indices = (0..dataset.len()).rev().collect::<Vec<_>>();
        if shuffle {
            remaining_indices.shuffle(&mut rng());
        }
        Self {
            dataset,
            remaining_indices,
        }
    }
}

/// A type alias for candle_datasets::Batcher
///
/// This struct is responsible for getting batches from a type that implements
/// the `Iterator` Trait.
pub type SpamDataBatcher = Batcher<IterResult2<SpamDatasetIter>>;



#[cfg(test)]
mod tests {
    use tiktoken_rs::get_bpe_from_model;
    use crate::test_utils::{sms_spam_df, test_parquet_path};
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

    #[test]
    pub fn test_spam_dataset_builder_parquet_file() -> anyhow::Result<()> {
        let test_parquet_path = test_parquet_path();
        let max_len = 10usize;
        let tokenizer = get_bpe_from_model("gpt2")?;
        let spam_dataset = SpamDatasetBuilder::new(&tokenizer)
            .load_data_from_request(test_parquet_path)
            .max_len(max_len)
            .build();

        assert_eq!(spam_dataset.len(), 5);
        assert_eq!(spam_dataset.max_len, max_len);
        // assert all encoded texts have length == max_length
        for text_enc in spam_dataset.encoded_texts.iter() {
            assert_eq!(text_enc.len(), max_len);
        }

        Ok(())
    }

    #[test]
    pub fn test_spam_dataset_iter() -> anyhow::Result<()> {
        let (df, _num_spam) = sms_spam_df();
        let tokenizer = get_bpe_from_model("gpt2")?;
        let max_length = 10_usize;
        let spam_dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
        let mut iter = SpamDatasetIter::new(spam_dataset.clone(), false);
        let mut count = 0_usize;

        // user iter to sequentially get next pair checking equality with dataset
        while let Some(Ok((this_encodings, this_label))) = iter.next() {
            assert!(this_encodings.shape().dims()[0] == max_length);
            assert!(this_label.shape().dims()[0] == 1_usize);
            count += 1;
        }
        assert_eq!(count, spam_dataset.len());
        Ok(())
    }
}
