use regex::{Captures, Regex};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

/// A simple tokenizer which can handle unknown words.
pub struct SimpleTokenizerV2 {
    str_to_int: HashMap<Arc<str>, i32>,
    int_to_str: HashMap<i32, Arc<str>>,
}

/// 特殊 token 的定义
const SPECIAL_TOKENS: [&str; 2] = ["<|unk|>", "<|endoftext|>"];

impl SimpleTokenizerV2 {
    /// Create a new `SimpleTokenizerV2` from a vocab.
    /// If `<|unk|>` or `<|endoftext|>` are missing, they will be added automatically.
    pub fn from_vocab<K, V, I>(vocab: I) -> Self
    where
        K: AsRef<str>,
        V: Borrow<i32>,
        I: IntoIterator<Item = (K, V)>,
    {
        // 收集到 HashMap 并找出最大 ID
        let mut search_map: HashMap<String, i32> = vocab
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_string(), *v.borrow()))
            .collect();
        let mut next_token_id = search_map.len();

        search_map.entry("<|unk|>".to_string()).or_insert_with(|| {
            next_token_id += 1;
            next_token_id as i32
        });
        search_map
            .entry("<|endoftext|>".to_string())
            .or_insert_with(|| {
                next_token_id += 1;
                next_token_id as i32
            });

        let (str_to_int, int_to_str): (HashMap<Arc<str>, i32>, HashMap<i32, Arc<str>>) = search_map
            .iter()
            .map(|(k, v)| {
                let str_val: Arc<str> = Arc::from(k.as_ref());
                let int_val = *v.borrow();
                ((str_val.clone(), int_val), (int_val, str_val))
            })
            .unzip();

        Self {
            str_to_int,
            int_to_str,
        }
    }

    /// Encode a text into its token ids.
    pub fn encode(&self, text: &str) -> Vec<i32> {
        let re = Regex::new(r#"([,.?_!"()']|--|\s)"#).unwrap();
        let preprocessed: Vec<&str> = re.split(text).collect();
        preprocessed
            .into_iter()
            .map(|s| {
                self.str_to_int
                    .get(s)
                    .unwrap_or(self.str_to_int.get("<|unk|>").unwrap())
            })
            .cloned()
            .collect()
    }

    /// Decode token ids into its text.
    pub fn decode(&self, ids: Vec<i32>) -> String {
        let text_vec: Vec<Arc<str>> = ids
            .iter()
            .map(|id| self.int_to_str.get(id).unwrap())
            .cloned()
            .collect();
        let text = &text_vec.join(" ")[..];

        let re = Regex::new(r#"\s+([,.?!"()'])"#).unwrap();
        String::from(re.replace_all(text, |caps: &Captures| caps[1].to_string()))
    }
}

#[cfg(test)]
mod tests {
    use crate::listing::list_04_text_tokenizer_v2::SimpleTokenizerV2;

    /// Generate vocab for testing.
    fn gen_vocab() -> Vec<(&'static str, i32)> {
        vec![("this", 1), ("is", 2), ("a", 3), ("test", 4)]
    }

    #[test]
    fn test_simple_tokenizer_v2_encode() {
        let vocab = gen_vocab();
        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);
        let token_ids = tokenizer.encode("this is a test! <|endoftext|>");

        assert_eq!(token_ids[0], 1);
        assert_eq!(token_ids[1], 2);
        assert_eq!(token_ids[2], 3);
        assert_eq!(token_ids[3], 4);
        assert_eq!(token_ids[4], 5);
        assert_eq!(token_ids[5], 6);
    }

    #[test]
    fn test_simple_tokenizer_v2_decode() {
        let vocab = gen_vocab();

        let tokenizer = SimpleTokenizerV2::from_vocab(vocab);
        let token_ids = vec![1, 2, 3, 4, 5, 6];
        let text = tokenizer.decode(token_ids);

        assert_eq!(text, "this is a test <|unk|> <|endoftext|>");
    }
}
