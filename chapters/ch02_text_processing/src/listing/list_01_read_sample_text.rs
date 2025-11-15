use std::fs;
use std::path::Path;

/// Sample text file path.
pub const SAMPLE_TEXT_PATH: &str = "../../assets/data/the-verdict.txt";

/// Reading in a short story as text sample into Rust.
pub fn read_sample_text<P: AsRef<Path>>(path: P) -> Result<String, std::io::Error> {
    let text = fs::read_to_string(&path).expect("unable to read file");
    // println!("Total number of character: {:?}", text.len());
    // println!("First 100 characters:\n{:?}", &text[..100]);
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_text() {
        let text = read_sample_text(SAMPLE_TEXT_PATH).unwrap();
        println!("Total number of character: {:?}", text.len());
        println!("First 100 characters:\n{:?}", &text[..100]);
    }
}
