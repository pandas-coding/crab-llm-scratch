use crate::listing::list_05_gpt_dataset_v1::{GPTDatasetIter, GPTDatasetV1};
use candle_datasets::{Batcher, batcher::IterResult2};

/// A type alias for candle_datasets::Batcher
///
/// This struct is responsible for getting batches from a type that implements
/// the `Iterator` Trait.
pub type GPTDataBatcher = Batcher<IterResult2<GPTDatasetIter>>;

/// A DataLoader trait
pub trait DataLoader {
    type Batcher;

    fn batcher(&self) -> Self::Batcher;
}

pub struct GPTDataLoader {
    dataset: GPTDatasetV1,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
}

impl DataLoader for GPTDataLoader {
    type Batcher = GPTDataBatcher;

    /// Returns a `GPTDataBatcher` that itself provides batches over the
    /// associated dataset.
    fn batcher(&self) -> Self::Batcher {
        let iter = GPTDatasetIter::new(&self.dataset, self.shuffle);
        Batcher::new_r2(iter)
            .batch_size(self.batch_size)
            .return_last_incomplete_batch(!self.drop_last)
    }
}

impl GPTDataLoader {
    /// Creates a new GPTDataLoader.
    pub fn new(dataset: &GPTDatasetV1, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        Self {
            dataset: dataset.clone(),
            batch_size,
            shuffle,
            drop_last,
        }
    }
    
    pub fn len(&self) -> usize {
        self.batcher().count()
        // // 注意：candle_datasets::Batcher 存在 bug，当 return_last_incomplete_batch
        // // 设置为 true 时，迭代器永远不会返回 None，导致 Iterator.count() 无法使用。
        // // 因此这里使用数学计算直接得出批次数量，完全避免依赖迭代器。
        //
        // // 函数式风格：纯表达式，无中间变量，使用元组解构和条件表达式组合
        // let (n, b) = (self.dataset.len(), self.batch_size);
        // if self.drop_last { n / b } else { n.div_ceil(b) }
    }
    
    pub fn is_empty(&self) -> bool {
        (self.dataset.len() < self.batch_size) && self.drop_last
    }
}

/// A data loader to generate batches with input-output pairs
pub fn create_dataloader_v1(
    txt: &str,
    batch_size: usize,
    max_len: usize,
    stride: usize,
    shuffle: bool,
    drop_last: bool,
) -> GPTDataLoader {
    let tokenizer = tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
    let dataset = GPTDatasetV1::new(txt, tokenizer, max_len, stride);
    GPTDataLoader::new(&dataset, batch_size, shuffle, drop_last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dataloader_v1() -> anyhow::Result<()> {
        let txt = "In the heart of the city";
        let batch_size = 2_usize;
        let stride = 1_usize;
        let max_length = 3_usize;
        let shuffle = false;
        let drop_last = false;
        let data_loader =
            create_dataloader_v1(txt, batch_size, max_length, stride, shuffle, drop_last);

        let count = data_loader.batcher()
            .filter_map(|result| result.ok())
            .inspect(|(inputs, targets)| {
                assert_eq!(inputs.dims(), targets.dims());
                assert!(inputs.dims()[0] <= batch_size);
            })
            .count();
        assert!(!data_loader.is_empty());
        assert_eq!(data_loader.len(), count);
        Ok(())
    }
}