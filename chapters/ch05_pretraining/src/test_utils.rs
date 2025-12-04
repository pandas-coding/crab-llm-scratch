use tiktoken_rs::get_bpe_from_model;

pub fn get_txt_tokenizer() -> (&'static str, tiktoken_rs::CoreBPE) {
    let text = "In the heart of the city";
    let tokenizer = get_bpe_from_model("gpt2").expect("failed to load bpe");
    (text, tokenizer)
}

