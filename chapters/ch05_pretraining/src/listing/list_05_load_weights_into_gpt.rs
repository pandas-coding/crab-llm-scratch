use std::collections::HashMap;
use std::fmt::Display;
use std::rc::Rc;
use std::sync::LazyLock;
use candle_core::{IndexOp, Tensor};
use candle_nn::VarMap;

const HF_TRANSFORMER_PREFIX: &str = "h";

/// Loading OpenAI weights into our GPT model code
pub fn load_weights_into_gpt(
    gpt_varmap: &VarMap,
    // load from HuggingFace
    mut weights: HashMap<String, Tensor>,
    model_prefix: Option<&str>,
    num_layers: usize,
) -> candle_core::Result<()> {
    let weights_mapping = &*WEIGHTS_MAPPING;

    // set weights for everything but transformer blocks
    load_from_weights_mapping(
        gpt_varmap,
        &mut weights,
        model_prefix,
        None,
        weights_mapping.get("not_transformer_wts").unwrap(),
    )?;

    // set transformer block weights
    for b in 0..num_layers {
        let var_prefix = if let Some(prefix) = model_prefix {
            format!("{prefix}.trf.{b}")
        } else {
            format!("trf.{b}")
        };
        let weights_prefix = format!("{HF_TRANSFORMER_PREFIX}.{b}");

        // set weights for everything in this transformer block but its q,k,v
        load_from_weights_mapping(
            gpt_varmap,
            &mut weights,
            Some(var_prefix.as_str()),
            Some(weights_prefix.as_str()),
            weights_mapping.get("transformer_wts_except_qkv").unwrap(),
        )?;

        // split attn.c_attn.bias
        let data_name = format!("{weights_prefix}.attn.c_attn.bias");
        let hf_attn_bias = weights
            .get(data_name.as_str())
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: data_name }.bt())?;
        let dim = hf_attn_bias.dims()[0] / 3_usize;
        let q_b = hf_attn_bias.i(..dim)?;
        let k_b = hf_attn_bias.i(dim..2 * dim)?;
        let v_b = hf_attn_bias.i(2 * dim..)?;
        weights.remove(format!("{weights_prefix}.attn.c_attn.bias").as_str()); // drop after splitting

        // split attn.c_attn.weight
        let data_name = format!("{weights_prefix}.attn.c_attn.weight");
        let hf_attn_weight = weights
            .get(data_name.as_str())
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: data_name }.bt())?;
        let q_w = hf_attn_weight.i((.., ..dim))?;
        let k_w = hf_attn_weight.i((.., dim..2 * dim))?;
        let v_w = hf_attn_weight.i((.., 2 * dim..))?;
        weights.remove(format!("{weights_prefix}.attn.c_attn.weight").as_str()); // drop after splitting

        // add split bias and weights tensors into weights following name convention
        weights.insert(format!("{weights_prefix}.attn.c_attn.query.bias"), q_b);
        weights.insert(format!("{weights_prefix}.attn.c_attn.key.bias"), k_b);
        weights.insert(format!("{weights_prefix}.attn.c_attn.value.bias"), v_b);
        weights.insert(format!("{weights_prefix}.attn.c_attn.query.weight"), q_w);
        weights.insert(format!("{weights_prefix}.attn.c_attn.key.weight"), k_w);
        weights.insert(format!("{weights_prefix}.attn.c_attn.value.weight"), v_w);

        // load q,k,v weights and biases
        load_from_weights_mapping(
            gpt_varmap,
            &mut weights,
            Some(var_prefix.as_str()),
            Some(weights_prefix.as_str()),
            weights_mapping.get("transformer_wts_qkv").unwrap(),
        )?;
    }
    Ok(())
}

/// A helper fn for loading weights from a `HashMap` into a `VarMap`
fn load_from_weights_mapping(
    gpt_varmap: &VarMap,
    weights: &mut HashMap<String, Tensor>,
    var_prefix: Option<&str>,
    weights_prefix: Option<&str>,
    weights_mapping: &HashMap<&str, HuggingFaceWeight>,
) -> candle_core::Result<()> {
    let gpt_data: std::sync::MutexGuard<'_, HashMap<String, candle_core::Var>> =
        gpt_varmap.data().lock().unwrap();

    for (gpt_name, hf_weight) in weights_mapping.iter() {
        let var_name = if let Some(prefix) = var_prefix {
            format!("{prefix}.{gpt_name}")
        } else {
            gpt_name.to_string()
        };

        let data_name = Rc::new(if let Some(w_prefix) = weights_prefix {
            format!("{w_prefix}.{}", hf_weight.name)
        } else {
            hf_weight.name.to_string()
        });

        let var = gpt_data
            .get(var_name.as_str())
            .ok_or_else(|| candle_core::Error::CannotFindTensor { path: var_name }.bt())?;

        let data = weights
            .get(data_name.as_str())
            .ok_or_else(|| {
                candle_core::Error::CannotFindTensor {
                    path: data_name.to_string(),
                }
                    .bt()
            })?
            .to_device(var.device())?; // move to same device as var
        if hf_weight.transpose {
            var.set(&data.t()?)?;
        } else {
            var.set(&data)?;
        }

        // drop weight after loaded into model
        if hf_weight.drop_after_loading {
            weights.remove(data_name.as_str());
        }
    }
    Ok(())
}

/// A convenience type for loading weights from HuggingFace Hub
struct HuggingFaceWeight {
    name: String,
    transpose: bool,
    drop_after_loading: bool,
}

impl Display for HuggingFaceWeight {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

/// Builder pattern for `HuggingFaceWeight`
struct HuggingFaceWeightBuilder {
    name: String,
    transpose: bool,
    drop_after_loading: bool,
}

impl HuggingFaceWeightBuilder {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            transpose: false,
            drop_after_loading: true,
        }
    }

    fn set_transpose(mut self) -> Self {
        self.transpose = true;
        self
    }

    fn unset_transpose(mut self) -> Self {
        self.transpose = false;
        self
    }

    fn unset_drop_after_loading(mut self) -> Self {
        self.drop_after_loading = false;
        self
    }

    fn set_drop_after_loading(mut self) -> Self {
        self.drop_after_loading = true;
        self
    }

    fn build(self) -> HuggingFaceWeight {
        HuggingFaceWeight {
            name: self.name,
            transpose: self.transpose,
            drop_after_loading: self.drop_after_loading,
        }
    }
}

/// A lazily loaded constant `HashMap` specifying mapping between our `GPTModel` and GPT-2 on HuggingFace.
static WEIGHTS_MAPPING: LazyLock<HashMap<&'static str, HashMap<&'static str, HuggingFaceWeight>>> =
    LazyLock::new(|| {
        HashMap::from([
            (
                "not_transformer_wts",
                HashMap::from([
                    (
                        "pos_emb.weight",
                        HuggingFaceWeightBuilder::new("wpe.weight").build(),
                    ),
                    (
                        "tok_emb.weight",
                        HuggingFaceWeightBuilder::new("wte.weight")
                            .unset_drop_after_loading()
                            .build(),
                    ),
                    (
                        "final_norm.scale",
                        HuggingFaceWeightBuilder::new("ln_f.weight").build(),
                    ),
                    (
                        "final_norm.shift",
                        HuggingFaceWeightBuilder::new("ln_f.bias").build(),
                    ),
                    (
                        "out_head.weight",
                        HuggingFaceWeightBuilder::new("wte.weight")
                            .unset_drop_after_loading()
                            .build(),
                    ),
                ]),
            ),
            (
                "transformer_wts_except_qkv",
                HashMap::from([
                    (
                        "ff.first_layer.bias",
                        HuggingFaceWeightBuilder::new("mlp.c_fc.bias").build(),
                    ),
                    (
                        "ff.first_layer.weight",
                        HuggingFaceWeightBuilder::new("mlp.c_fc.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "ff.second_layer.bias",
                        HuggingFaceWeightBuilder::new("mlp.c_proj.bias").build(),
                    ),
                    (
                        "ff.second_layer.weight",
                        HuggingFaceWeightBuilder::new("mlp.c_proj.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "norm1.scale",
                        HuggingFaceWeightBuilder::new("ln_1.weight").build(),
                    ),
                    (
                        "norm1.shift",
                        HuggingFaceWeightBuilder::new("ln_1.bias").build(),
                    ),
                    (
                        "norm2.scale",
                        HuggingFaceWeightBuilder::new("ln_2.weight").build(),
                    ),
                    (
                        "norm2.shift",
                        HuggingFaceWeightBuilder::new("ln_2.bias").build(),
                    ),
                    (
                        "mha.out_proj.bias",
                        HuggingFaceWeightBuilder::new("attn.c_proj.bias").build(),
                    ),
                    (
                        "mha.out_proj.weight",
                        HuggingFaceWeightBuilder::new("attn.c_proj.weight")
                            .set_transpose()
                            .build(),
                    ),
                ]),
            ),
            (
                "transformer_wts_qkv",
                HashMap::from([
                    // NOTE: these weights need to be derived from attn.c_attn.bias and attn.c_attn.weight
                    // and this is done within the loop.
                    (
                        "mha.key.bias",
                        HuggingFaceWeightBuilder::new("attn.c_attn.key.bias").build(),
                    ),
                    (
                        "mha.key.weight",
                        HuggingFaceWeightBuilder::new("attn.c_attn.key.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "mha.query.bias",
                        HuggingFaceWeightBuilder::new("attn.c_attn.query.bias").build(),
                    ),
                    (
                        "mha.query.weight",
                        HuggingFaceWeightBuilder::new("attn.c_attn.query.weight")
                            .set_transpose()
                            .build(),
                    ),
                    (
                        "mha.value.bias",
                        HuggingFaceWeightBuilder::new("attn.c_attn.value.bias").build(),
                    ),
                    (
                        "mha.value.weight",
                        HuggingFaceWeightBuilder::new("attn.c_attn.value.weight")
                            .set_transpose()
                            .build(),
                    ),
                ]),
            ),
        ])
    });