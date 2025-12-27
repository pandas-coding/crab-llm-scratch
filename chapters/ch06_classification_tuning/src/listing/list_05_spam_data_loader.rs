use candle_datasets::Batcher;
use crate::listing::list_04_spam_dataset::{SpamDataBatcher, SpamDataset, SpamDatasetIter};

pub struct SpamDataLoader {
    dataset: SpamDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl SpamDataLoader {
    /// Creates a new SpamLoader.
    pub fn new(dataset: SpamDataset, batch_size: usize, shuffle: bool, drop_last: bool) -> SpamDataLoader {
        Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
        }
    }

    /// Returns a `SpamDataBatcher` that itself provides batches over the associated dataset.
    pub fn batcher(&self) -> SpamDataBatcher {
        let iter = SpamDatasetIter::new(self.dataset.to_owned(), self.shuffle);
        Batcher::new_r2(iter)
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(!self.drop_last)
    }
    
    pub fn len(&self) -> usize {
        self.batcher().count()
        // // There is a bug in candle_datasets::Batcher, such that if
        // // return_last_incomplete_batch is set to true, then the iterator
        // // will never return None. This breaks `Iterator.count()` which consumes
        // // the iterator until a None is encountered.
        // match self.drop_last {
        //     true => self.batcher().count(),
        //     false => {
        //         let mut batcher = self.batcher();
        //         let mut count = 0_usize;
        //         while let Some(Ok(_el)) = batcher.next() {
        //             count += 1;
        //         }
        //         count
        //     }
        // }
    }
    
    pub fn is_empty(&self) -> bool {
        (self.dataset.len() < self.batch_size) && self.drop_last
    }
}

#[cfg(test)]
mod tests {
    use tiktoken_rs::get_bpe_from_model;
    use crate::listing::list_04_spam_dataset::PAD_TOKEN_ID;
    use crate::test_utils::sms_spam_df;
    use super::*;
    
    #[test]
    fn test_spam_data_loader() -> anyhow::Result<()> {
        let (df, _num_spam) = sms_spam_df();
        let tokenizer = get_bpe_from_model("gpt2")?;
        let max_length = 10_usize;
        let spam_dataset = SpamDataset::new(df, &tokenizer, Some(max_length), PAD_TOKEN_ID);
        let batch_size = 2_usize;
        let shuffle = false;
        let drop_last = false;
        let data_loader = SpamDataLoader::new(spam_dataset, batch_size, shuffle, drop_last);

        let mut batcher = data_loader.batcher();
        let mut count = 0_usize;
        while let Some(Ok((inputs, targets))) = batcher.next() {
            assert!(inputs.dims()[0] <= batch_size);
            assert!(targets.dims()[0] <= batch_size);
            assert_eq!(inputs.dims()[1], max_length);
            assert_eq!(targets.dims()[1], 1_usize);
            count += 1;
        }
        assert_eq!(data_loader.len(), count);
        assert!(!data_loader.is_empty());
        Ok(())
    }
}
