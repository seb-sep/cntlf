use ort::{GraphOptimizationLevel, Session, TensorElementType, ValueType, inputs};
use tauri::api::file;
use tokenizers::Tokenizer;
use ndarray::{Array1, ArrayBase, Axis, Dim, OwnedRepr, linalg::Dot};
use std::fs;


const SEQ_LEN: usize = 8192;


#[tauri::command]
pub fn embed_file(file_path: String) -> Result<String, String> {
    println!("Running embedding");
    let text = fs::read_to_string(file_path).map_err(|err| err.to_string())?;
    let embedding = run_embedding(&text).map_err(|err| {eprintln!("{err}"); err.to_string()})?;
    Ok(embedding.to_string())
    // test_embedding();
    // Ok("".into())
}


// fn documents_from_file(file_path: &str, tokenizer: Tokenizer) -> Result<Vec<String>, core::error::Error> {
//     tokenizer.
//     let contents = fs::read_to_string(file_path).
// }

fn test_embedding() {
    let e1 = run_embedding("Hello world!").unwrap();
    let e2 = run_embedding("The mitochondria is the powerhouse of the cell").unwrap();

    let similarity = e1.dot(&e2) / (e1.dot(&e1).sqrt() * e2.dot(&e2).sqrt());

    println!("{similarity}");


    
}

fn run_embedding(document: &str) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>, ort::Error> {
    let embed_path = "assets/nomic-embed-quantized.onnx";

    // load the model
    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(embed_path)?;
    
    // tokenize and reshape input
    let (input_ids, type_ids, masks) = tokenize(&document).unwrap();
    let input_ids = input_ids.view().insert_axis(Axis(0));
    let type_ids = type_ids.view().insert_axis(Axis(0));
    let masks = masks.view().insert_axis(Axis(0));

    // run model
    let outputs = model.run(inputs![input_ids, type_ids, masks]?)?;
    let embedding = (&outputs[0]).try_extract_tensor::<f32>()?.mean_axis(Axis(1)).unwrap().index_axis_move(Axis(0), 0);
    println!("{:?}", embedding.dim());
    Ok(embedding.into_dimensionality().unwrap())
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