use std::fs;
use std::path::Path;

/// Reading in a short story as text sample into Rust.
pub fn read_sample_text<P: AsRef<Path>>(path: P) -> Result<String, ()> {
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
        let text = read_sample_text("data/the-verdict.txt").unwrap();
        println!("Total number of character: {:?}", text.len());
        println!("First 100 characters:\n{:?}", &text[..100]);
    }
}
