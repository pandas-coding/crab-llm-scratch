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

#[cfg(test)]
mod tests {
    use super::*;

    /// generate txt tokenizer.
    fn txt_tokenizer() -> (String, tiktoken_rs::CoreBPE) {
        let txt = "In the heart of the city";
        let tokenizer =  tiktoken_rs::get_bpe_from_model("gpt2").unwrap();
        (txt.to_string(), tokenizer)
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
}
