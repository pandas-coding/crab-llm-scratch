use std::collections::HashMap;
use regex::Regex;
use crate::listing::read_sample_text::{read_sample_text, SAMPLE_TEXT_PATH};

pub fn create_sample_vocab() -> std::io::Result<HashMap<String, i32>> {
    let text = read_sample_text(SAMPLE_TEXT_PATH)?;
    let re = Regex::new(r#"([,.?_!"()']|--|\s)"#).unwrap();
    let mut  preprocessed = re.split(&text[..]).collect::<Vec<_>>();
    preprocessed.sort();

    let vocab: HashMap<String, i32> = HashMap::from_iter(
        preprocessed
            .iter()
            .enumerate()
            .map(|(idx, el)| (el.to_string(), idx as i32))
    );

    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_sample_vocab() {
        let vocab = create_sample_vocab().unwrap();

        // print the most 50 vocabularies
        for (i, item) in vocab.iter().enumerate() {
            println!("{:?}", item);
            if i >= 50 {
                break;
            }
        }
    }
}
