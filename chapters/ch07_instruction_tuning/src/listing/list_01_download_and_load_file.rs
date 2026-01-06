use std::fmt::Display;
use std::fs::{read_to_string, File};
use std::io;
use std::path::Path;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, NoneAsEmptyString};

/// Downloading the dataset
pub fn download_and_load_file<P: AsRef<Path>>(
    file_path: P,
    url: &str,
    overwrite: bool,
) -> anyhow::Result<Vec<InstructionResponseExample>> {
    if !file_path.as_ref().exists() || overwrite {
        // download json file
        let resp = reqwest::blocking::get(url)?;
        let content = resp.bytes()?;
        let mut out = File::create(&file_path)?;
        io::copy(&mut content.as_ref(), &mut out)?;
    }
    let data = load_instruction_data_from_json(file_path)?;
    Ok(data)
}

// Marker trait for instruction example
pub trait InstructionExample {
    fn instruction(&self) -> &String;
    fn input(&self) -> &Option<String>;
    fn output(&self) -> &String;
}

/// A type for containing an instruction-response pair
#[serde_as]
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct InstructionResponseExample {
    instruction: String,
    #[serde_as(as = "NoneAsEmptyString")]
    input: Option<String>,
    output: String,
    model_response: Option<String>,
}

impl InstructionResponseExample {
    pub fn new(instruction: &str, input: Option<&str>, output: &str) -> Self {
        Self {
            instruction: instruction.to_string(),
            input: input.map(|inp| inp.to_string()),
            output: output.to_string(),
            model_response: None,
        }
    }

    pub fn model_response(&self) -> &Option<String> {
        &self.model_response
    }

    pub fn set_model_response(&mut self, model_response: &str) {
        self.model_response = Some(model_response.to_string());
    }
}

impl Display for InstructionResponseExample {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Instruction: {}\nInput: {:?}\nOutput: {}\nModel Response: {:?}",
            self.instruction, self.input, self.output, self.model_response
        )
    }
}

impl InstructionExample for InstructionResponseExample {
    fn instruction(&self) -> &String {
        &self.instruction
    }

    fn input(&self) -> &Option<String> {
        &self.input
    }

    fn output(&self) -> &String {
        &self.output
    }
}

/// Helper function to write instruction data to a json
pub fn load_instruction_data_from_json<P: AsRef<Path>, S: Serialize + for<'a> Deserialize<'a>>(
    file_path: P,
) -> anyhow::Result<Vec<S>> {
    let json_str = read_to_string(file_path.as_ref())
        .with_context(|| format!("Unable to read {}", file_path.as_ref().display()))?;
    let data = serde_json::from_str(&json_str)?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use tempfile::NamedTempFile;
    use crate::listing::INSTRUCTION_DATA_URL;
    use super::*;

    #[test]
    fn test_download_and_load_file() -> anyhow::Result<()> {
        let test_file = NamedTempFile::new()?;
        let file_path = test_file.into_temp_path().keep()?;
        let data = download_and_load_file(file_path, INSTRUCTION_DATA_URL, true)?;
        assert_eq!(data.len(), 1100);
        Ok(())
    }
}
