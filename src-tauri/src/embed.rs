use crate::sqlite;
use ndarray::{Array1, ArrayBase, Axis, Dim, OwnedRepr};
use ort::{inputs, GraphOptimizationLevel, Session};
use std::{fs, path::PathBuf};
use tauri::State;
use tokenizers::Tokenizer;

use crate::AppResources;

#[tauri::command]
pub fn embed_file(file_path: String, state: State<AppResources>) -> Result<String, String> {
    let text = fs::read_to_string(&file_path).map_err(|err| err.to_string())?;
    let embedder = state.embedder.lock().unwrap();
    let tokenizer = state.tokenizer.lock().unwrap();
    let embedding = run_embedding(
        &text,
        &embedder,
        &tokenizer,
        NomicTaskPrefix::SearchDocument,
    )
    .map_err(|err| err.to_string())?;

    let conn = state.db.lock().unwrap();
    sqlite::add_embedding(&conn, &file_path, &embedding).map_err(|err| err.to_string())?;
    Ok(embedding.to_string())
}

pub type Embedding = ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>;

pub enum NomicTaskPrefix {
    SearchQuery,
    SearchDocument,
    Classification,
    Clustering,
}

impl NomicTaskPrefix {
    fn to_str(&self) -> String {
        match self {
            NomicTaskPrefix::SearchQuery => "search_query".to_owned(),
            NomicTaskPrefix::SearchDocument => "search_document".to_owned(),
            NomicTaskPrefix::Classification => "classification".to_owned(),
            NomicTaskPrefix::Clustering => "clustering".to_owned(),
        }
    }

    fn append_task(&self, document: &str) -> String {
        self.to_str() + document
    }
}

fn cosine_similarity(e1: &Embedding, e2: &Embedding) -> f32 {
    e1.dot(e2) / (e1.dot(e1).sqrt() * e2.dot(e2).sqrt())
}

pub fn init_model(weights_file: PathBuf) -> Result<Session, ort::Error> {
    // load the model
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(weights_file)
}

pub fn run_embedding(
    document: &str,
    model: &Session,
    tokenizer: &Tokenizer,
    task: NomicTaskPrefix,
) -> Result<ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>, ort::Error> {
    // tokenize and reshape input
    let (input_ids, type_ids, masks) = tokenize(&task.append_task(document), tokenizer).unwrap();
    let input_ids = input_ids.view().insert_axis(Axis(0));
    let type_ids = type_ids.view().insert_axis(Axis(0));
    let masks = masks.view().insert_axis(Axis(0));

    // run model
    let outputs = model.run(inputs![input_ids, type_ids, masks]?)?;
    let embedding = (&outputs[0])
        .try_extract_tensor::<f32>()?
        .mean_axis(Axis(1))
        .unwrap()
        .index_axis_move(Axis(0), 0);
    Ok(embedding.into_dimensionality().unwrap())
}

type TokenArray = ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>;
fn tokenize(
    text: &str,
    tokenizer: &Tokenizer,
) -> Result<(TokenArray, TokenArray, TokenArray), tokenizers::Error> {
    let tokens = tokenizer.encode(text, false)?;
    let token_ids = tokens
        .get_ids()
        .iter()
        .map(|i| *i as i64)
        .collect::<Vec<_>>();
    let type_ids = tokens
        .get_type_ids()
        .iter()
        .map(|i| *i as i64)
        .collect::<Vec<_>>();
    let masks = tokens
        .get_attention_mask()
        .iter()
        .map(|i| *i as i64)
        .collect::<Vec<_>>();
    Ok((
        Array1::from_vec(token_ids),
        Array1::from_vec(type_ids),
        Array1::from_vec(masks),
    ))
}
#[cfg(test)]
mod test {
    use crate::embed::{cosine_similarity, NomicTaskPrefix};

    use super::run_embedding;

    #[test]
    fn vector_similarity_test() {
        let tokenizer = tokenizers::Tokenizer::from_file("assets/tokenizer.json").unwrap();
        let model = crate::embed::init_model("assets/nomic-embed-quantized.onnx".into()).unwrap();

        let root = "/Users/sebastiansepulveda/";
        let story = std::fs::read_to_string(root.to_owned() + "story.txt").unwrap();
        let e_story =
            run_embedding(&story, &model, &tokenizer, NomicTaskPrefix::SearchDocument).unwrap();

        let todo = std::fs::read_to_string(root.to_owned() + "todo.md").unwrap();
        let e_todo =
            run_embedding(&todo, &model, &tokenizer, NomicTaskPrefix::SearchDocument).unwrap();

        let code = std::fs::read_to_string(root.to_owned() + "list.rs").unwrap();
        let e_code =
            run_embedding(&code, &model, &tokenizer, NomicTaskPrefix::SearchDocument).unwrap();

        let query = run_embedding(
            "A story about a girl",
            &model,
            &tokenizer,
            NomicTaskPrefix::SearchQuery,
        )
        .unwrap();

        let story_sim = cosine_similarity(&e_story, &query);
        let todo_sim = cosine_similarity(&e_todo, &query);
        let code_sim = cosine_similarity(&e_code, &query);

        println!("Similarity with story: {story_sim}\nSimilarity with todo: {todo_sim}\n, similarity with code: {code_sim}");

        assert!(story_sim > todo_sim && story_sim > code_sim);
    }
}
