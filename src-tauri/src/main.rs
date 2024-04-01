// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use ort::Session;
use tauri::{utils::resources, App};
use tokenizers::{tokenizer, Tokenizer};
use rusqlite::Connection;
use std::sync::Mutex;

mod ai;
mod db;

use ai::embed::init_model;
use db::sqlite::init_db;


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
    pub db: Mutex<Connection>
}

fn init_resources() -> AppResources {
    let embedder = init_model().unwrap();
    let db = init_db().unwrap();
    let tokenizer = Tokenizer::from_file("assets/tokenizer.json").unwrap();

    AppResources {
        embedder: Mutex::new(embedder),
        tokenizer: Mutex::new(tokenizer),
        db: Mutex::new(db)
    }
}

fn main() {
    let resources = init_resources();
    tauri::Builder::default()
        .manage(resources)
        .invoke_handler(tauri::generate_handler![greet, 
            ai::embed::embed_file,
            db::sqlite::find_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
