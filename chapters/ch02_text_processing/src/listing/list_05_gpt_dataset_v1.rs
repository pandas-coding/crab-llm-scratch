use std::rc::Rc;

/// A dataset for batched inputs and targets
/// GPTDatasetV1 is a wrapper for `GPTDatasetV1Inner` which is refcounted.
/// This makes cloning datasets cheap.
#[derive(Clone)]
pub struct GPTDatasetV1 {
    input_ids: Rc<Vec<Vec<u32>>>,
    target_ids: Rc<Vec<Vec<u32>>>,
}

impl GPTDatasetV1 {

    /// Creates a new `GPTDatasetV1`.
    pub fn new(txt: &str, tokenizer: tiktoken_rs::CoreBPE, max_length: usize, stride: usize) -> Self {
        let token_ids = tokenizer.encode_with_special_tokens(txt);

        let (input_ids, target_ids): (Vec<Vec<u32>>, Vec<Vec<u32>>) = (0..token_ids.len() - max_length)
            .step_by(stride)
            .map(|i| {
                let input_chunk = token_ids[i..i + max_length].to_vec();
                let target_chunk = token_ids[i + 1..i + max_length + 1].to_vec();
                (input_chunk, target_chunk)
            })
            .unzip();

        Self {
            input_ids: Rc::new(input_ids),
            target_ids: Rc::new(target_ids),
        }
    }

    /// Gets the number of input-target sequences in the dataset.
    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// Checks whether the dataset is empty or has no input-target sequences.
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    /// Returns the input tokens for all input sequences.
    pub fn input_ids(&self) -> &Vec<Vec<u32>> {
        &self.input_ids
    }

    /// Returns the target token ides for all input sequences.
    pub fn target_ids(&self) -> &Vec<Vec<u32>> {
        &self.target_ids
    }

    /// Returns the input-target pair at the specified index.
    pub fn get_pair_at_index(&self, index: usize) -> (&Vec<u32>, &Vec<u32>) {
        (&self.input_ids[index], &self.target_ids[index])
    }

}

/// `GPTDatasetIter` analagous to PyTorch's `DataLoader`
/// A data loader to generate batches with input-target pairs
/// We can use `GPTDatasetIter` with `candle_datasets::Batcher` to get desired
/// batches of examples.
pub struct GPTDatasetIter {
    dataset: GPTDatasetV1,
    remaining_indices: Vec<usize>,
}

impl Iterator for GPTDatasetIter {
    type Item = candle_core::Result<(candle_core::Tensor, candle_core::Tensor)>;
    fn next(&mut self) -> Option<Self::Item> {
        use candle_core::{Device, Tensor};

        match self.remaining_indices.pop() {
            None => None,
            Some(idx) => {
                let (input_idx,  target_ids) = self.dataset.get_pair_at_index(idx);

                // turn into Tensors and return
                let device = Device::cuda_if_available(0).unwrap();
                let input_tensor = Tensor::new(&input_idx[..], &device);
                let target_tensor = Tensor::new(&target_ids[..], &device);
                Some(candle_core::error::zip(input_tensor, target_tensor))
            }
        }
    }
}

impl GPTDatasetIter {
    /// Creates a new `GPTDatasetIter`.
    pub fn new(dataset: &GPTDatasetV1, shuffle: bool) -> Self {
        use rand::{rng, seq::SliceRandom};

        let mut indices: Vec<usize> = (0..dataset.len()).collect();

        if shuffle {
            indices.shuffle(&mut rng());
        } else {
            indices.reverse();
        }

        Self {
            dataset: dataset.clone(),
            remaining_indices: indices,
        }
    }
}

#[cfg(test)]
mod tests {
    use candle_datasets::Batcher;
    use super::*;

    /// generate txt tokenizer.
    fn txt_tokenizer() -> (String, tiktoken_rs::CoreBPE) {
        let txt = "In the heart of the city";
        let tokenizer =  tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
        (txt.to_string(), tokenizer)
    }

    fn gpt_dataset() -> GPTDatasetV1 {
        let stride = 1usize;
        let max_len  = 3usize;
        let (txt, tokenizer) = txt_tokenizer();
        GPTDatasetV1::new(&txt, tokenizer, max_len, stride)
    }

    #[test]
    fn test_gpt_dataset_v1_init() {
        let (txt, tokenizer) = txt_tokenizer();
        let token_ids = tokenizer.encode_with_special_tokens(&txt);
        let stride  = 1usize;
        let max_length = 3usize;
        let dataset = GPTDatasetV1::new(&txt, tokenizer, max_length, stride);

        // test target alignments
        for mx in 1..max_length {
            assert_eq!(
                dataset.input_ids[0][mx],
                dataset.target_ids[0][mx - 1],
            )
        }

        for ix in 1..dataset.input_ids.len() {
            // test max length per input
            assert_eq!(dataset.input_ids[ix].len(), max_length);
            // test stride alignments
            assert_eq!(dataset.input_ids[ix][0], token_ids[ix * stride]);
        }
    }

    #[test]
    fn test_gpt_dataset_v1_iter() -> anyhow::Result<()> {
        let (txt, tokenizer) = txt_tokenizer();
        let stride = 1usize;
        let max_length = 3usize;
        let dataset = GPTDatasetV1::new(&txt, tokenizer, max_length, stride);
        let iter = GPTDatasetIter::new(&dataset, false);

        // 使用函数式编程风格：try_for_each 替代 for 循环
        iter.enumerate()
            .try_for_each(|(idx, result)| {
                let (input_tensor, target_tensor) = result?;
                
                // 验证张量形状
                assert_eq!(input_tensor.shape().dims()[0], max_length);
                assert_eq!(target_tensor.shape().dims()[0], max_length);

                // 转换并验证数据一致性
                let verify_slice = |tensor: candle_core::Tensor, expected: &[u32], label: &str| -> anyhow::Result<()> {
                    let actual = tensor.to_vec1::<u32>()?;
                    assert_eq!(
                        actual.as_slice(),
                        expected,
                        "{} 在索引 {} 处不匹配",
                        label,
                        idx
                    );
                    Ok(())
                };

                verify_slice(input_tensor, &dataset.input_ids[idx], "输入序列")?;
                verify_slice(target_tensor, &dataset.target_ids[idx], "目标序列")?;

                Ok(())
            })
    }

    #[test]
    fn test_gpt_dataset_with_batch() -> anyhow::Result<()> {
        let batch_size = 2usize;
        let dataset =  gpt_dataset();
        let iter = GPTDatasetIter::new(&dataset, false);
        let mut batch_iter = Batcher::new_r2(iter).batch_size(batch_size);
        match batch_iter.next() {
            None => panic!("None iter"),
            Some(Err(err)) => panic!("{}", err),
            Some(Ok((inputs, targets))) => {
                assert_eq!(inputs.dims(), targets.dims());
                assert_eq!(inputs.dims()[0], batch_size);
            }
        }

        Ok(())
    }
}
