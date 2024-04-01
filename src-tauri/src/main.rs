// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use ort::Session;
use rusqlite::Connection;
use std::{path::PathBuf, sync::Mutex};
use tauri::{utils::resources, App, Manager};
use tokenizers::{processors::template::Tokens, tokenizer, Tokenizer};

mod embed;
mod sqlite;

// Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

// TODO: add public struct for holding the tokenizer, embedding model, and sqlite session, an
// use tauri manage to get it https://tauri.app/v1/guides/features/command/#accessing-managed-state

pub struct AppResources {
    pub embedder: Mutex<Session>,
    pub tokenizer: Mutex<Tokenizer>,
    pub db: Mutex<Connection>,
}

#[cfg(debug_assertions)]
fn get_resource_root(app: &tauri::App) -> PathBuf {
    "assets".into()
}

#[cfg(not(debug_assertions))]
fn get_resource_root(app: &tauri::App) -> PathBuf {
    app.path_resolver().resolve_resource("assets").unwrap()
}

fn init_resources(app: &tauri::App) -> AppResources {
    
    let root = get_resource_root(app);
    let embedder = embed::init_model(root.join("nomic-embed-quantized.onnx")).unwrap();
    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    let db = sqlite::init_db(root.join("sqlite-vss-m1")).unwrap();

    AppResources {
        embedder: Mutex::new(embedder),
        tokenizer: Mutex::new(tokenizer),
        db: Mutex::new(db),
    }
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let resources = init_resources(app);
            app.manage(resources);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            embed::embed_file,
            sqlite::find_file
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
