use candle_core::Device;
use candle_nn::{VarBuilder, VarMap};
use hf_hub::api::sync::Api;
use ch04_gpt_implementation::listing::list_01_dummy_gpt_model::Config;
use ch04_gpt_implementation::listing::list_07_gpt_model::GPTModel;
use ch05_pretraining::listing::list_05_load_weights_into_gpt::load_weights_into_gpt;

pub const HF_GPT2_MODEL_ID: &str = "openai-community/gpt2";

/// Loading a pretrained GPT model
pub fn download_and_load_gpt2(
    varmap: &VarMap,
    vb: VarBuilder,
    cfg: Config,
    model_id: &str,
) -> candle_core::Result<GPTModel> {
    let device = Device::cuda_if_available(0)?;
    let model = GPTModel::new(cfg, vb)?;

    // get weights from HuggingFace Hub
    let api = Api::new().map_err(candle_core::Error::wrap)?;
    let repo = api.model(model_id.to_string());
    let weights = repo
        .get("model.safetensors")
        .map_err(candle_core::Error::wrap)?;
    let weights = candle_core::safetensors::load(weights, &device)?;

    // load weights
    load_weights_into_gpt(varmap, weights, Some("model"), cfg.n_layers)?;
    
    Ok(model)
}
