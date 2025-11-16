use regex::{Captures, Regex};
use std::borrow::Borrow;
use std::collections::HashMap;
use std::sync::Arc;

/// A simple text tokenizer implementation.
pub struct SimpleTokenizerV1 {
    str_to_int: HashMap<Arc<str>, i32>,
    int_to_str: HashMap<i32, Arc<str>>,
}

impl SimpleTokenizerV1 {
    /// Create a new `SimpleTokenizerV1` from a vocab.
    pub fn from_vocab<K, V, I>(vocab: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: Borrow<i32>,
    {
        let (str_to_int, int_to_str): (HashMap<Arc<str>, i32>, HashMap<i32, Arc<str>>) = vocab
            .into_iter()
            .map(|(k, v)| {
                let str: Arc<str> = Arc::from(k.as_ref());
                let int = *v.borrow();
                ((str.clone(), int), (int, str))
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
            .map(|s| self.str_to_int.get(s).unwrap())
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
    use crate::listing::list_03_text_tokenizer::SimpleTokenizerV1;

    /// Generate vocab for testing.
    fn gen_vocab() -> Vec<(&'static str, i32)> {
        vec![
            ("this", 1),
            ("is", 2),
            ("a", 3),
            ("test", 4),
        ]
    }

    #[test]
    fn test_simple_tokenizer_v1_encode() {
        let vocab = gen_vocab();
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);
        let token_ids = tokenizer.encode("this is a test");

        assert_eq!(token_ids[0], 1);
        assert_eq!(token_ids[1], 2);
        assert_eq!(token_ids[2], 3);
        assert_eq!(token_ids[3], 4);
    }

    #[test]
    fn test_simple_tokenizer_v1_decode() {
        let vocab = gen_vocab();
        let tokenizer = SimpleTokenizerV1::from_vocab(vocab);

        let token_ids = vec![1, 2, 3, 4];
        let text = tokenizer.decode(token_ids);

        assert_eq!(text, "this is a test");
    }
}
