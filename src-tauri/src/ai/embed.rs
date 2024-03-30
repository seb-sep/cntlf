use ort::{GraphOptimizationLevel, Session, TensorElementType, ValueType, inputs};
use tokenizers::Tokenizer;
use ndarray::{Array1, ArrayBase, Axis, Dim, IxDynImpl, OwnedRepr};


#[tauri::command]
pub fn embed_file(file_path: String) -> Result<String, String>{
    println!("Running embedding");
    let embedding = run_embedding(file_path).map_err(|err| {eprintln!("{err}"); err.to_string()})?;
    Ok(embedding.to_string())
}

fn run_embedding(file_path: String) -> Result<ArrayBase<OwnedRepr<f32>, Dim<IxDynImpl>>, ort::Error> {
    let embed_path = "assets/nomic-embed-quantized.onnx";

    // load the model
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(embed_path)?;
    
    // tokenize and reshape input
    let (input_ids, type_ids, masks) = tokenize(&file_path).unwrap();
    let input_ids = input_ids.view().insert_axis(Axis(0));
    let type_ids = type_ids.view().insert_axis(Axis(0));
    let masks = masks.view().insert_axis(Axis(0));

    println!("{:?}", input_ids.dim());

    // run model
    let outputs = model.run(inputs![input_ids, type_ids, masks]?)?;
    let embedding = (&outputs[0]).try_extract_tensor::<f32>()?.mean_axis(Axis(1)).unwrap();
    println!("{:?}", embedding);
    Ok(embedding)
}

type TokenArray = ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>;
fn tokenize(text: &str) -> Result<(TokenArray, TokenArray, TokenArray), tokenizers::Error> {
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json")?;
    let tokens = tokenizer.encode(text, false)?;
    let token_ids = tokens.get_ids().iter().map(|i| *i as i64).collect::<Vec<_>>(); 
    let type_ids = tokens.get_type_ids().iter().map(|i| *i as i64).collect::<Vec<_>>(); 
    let masks = tokens.get_attention_mask().iter().map(|i| *i as i64).collect::<Vec<_>>(); 
    Ok((Array1::from_vec(token_ids), Array1::from_vec(type_ids), Array1::from_vec(masks)))
    
}

fn display_element_type(t: TensorElementType) -> &'static str {
	match t {
		TensorElementType::Bfloat16 => "bf16",
		TensorElementType::Bool => "bool",
		TensorElementType::Float16 => "f16",
		TensorElementType::Float32 => "f32",
		TensorElementType::Float64 => "f64",
		TensorElementType::Int16 => "i16",
		TensorElementType::Int32 => "i32",
		TensorElementType::Int64 => "i64",
		TensorElementType::Int8 => "i8",
		TensorElementType::String => "str",
		TensorElementType::Uint16 => "u16",
		TensorElementType::Uint32 => "u32",
		TensorElementType::Uint64 => "u64",
		TensorElementType::Uint8 => "u8"
	}
}

fn display_value_type(value: &ValueType) -> String {
	match value {
		ValueType::Tensor { ty, dimensions } => {
			format!(
				"Tensor<{}>({})",
				display_element_type(*ty),
				dimensions
					.iter()
					.map(|c| if *c == -1 { "dyn".to_string() } else { c.to_string() })
					.collect::<Vec<_>>()
					.join(", ")
			)
		}
		ValueType::Map { key, value } => format!("Map<{}, {}>", display_element_type(*key), display_element_type(*value)),
		ValueType::Sequence(inner) => format!("Sequence<{}>", display_value_type(inner))
	}
}