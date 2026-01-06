use crate::listing::list_01_download_and_load_file::InstructionResponseExample;

pub fn instruction_example() -> InstructionResponseExample {
    let instruction = "Here is a fake instruction.";
    let input = Some("Here is a fake input.");
    let output = "here is a fake output.";
    
    InstructionResponseExample::new(
        &instruction,
        input,
        &output,
    )
}

pub fn another_instruction_example() -> InstructionResponseExample {
    let instruction = "Here is yet another fake instruction.".to_string();
    let output = "here is yet another fake output.".to_string();
    
    InstructionResponseExample::new(
         &instruction,
         None,
         &output,
    )
}

pub fn instruction_data(
    instruction_example: InstructionResponseExample,
    another_instruction_example: InstructionResponseExample,
) -> Vec<InstructionResponseExample> {
    let data = vec![
        instruction_example.clone(),
        another_instruction_example.clone(),
        instruction_example.clone(),
        another_instruction_example.clone(),
        instruction_example,
    ];
    data
}
