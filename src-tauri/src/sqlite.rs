use crate::embed::{run_embedding, Embedding, NomicTaskPrefix};
use byteorder::{ByteOrder, LittleEndian};
use ndarray::{Array1, ArrayBase, Dim, OwnedRepr};
use rusqlite::{params, Connection, Error, LoadExtensionGuard, Result};
use std::{mem, path::PathBuf};

use crate::AppResources;

#[tauri::command]
pub fn find_file(query: String, state: tauri::State<AppResources>) -> Result<String, String> {
    let embedder = state.embedder.lock().unwrap();
    let tokenizer = state.tokenizer.lock().unwrap();
    let db = state.db.lock().unwrap();
    let embedding = run_embedding(&query, &embedder, &tokenizer, NomicTaskPrefix::SearchQuery)
        .map_err(|err| {
            eprintln!("{err}");
            err.to_string()
        })?;
    semantic_file_search(&db, &embedding).map_err(|err| err.to_string())
}

pub fn init_db(obj_root: PathBuf) -> Result<Connection, Error> {
    // let conn = Connection::open("assets/db.sqlite")?;
    let conn = Connection::open_in_memory()?;

    // load extensions
    unsafe {
        let _guard = LoadExtensionGuard::new(&conn)?;
        conn.load_extension(obj_root.join("vector0.dylib"), None)?;
        conn.load_extension(obj_root.join("vss0.dylib"), None)?;
        println!("loading extensions success {:?}", conn);
    }

    conn.execute(
        "create table if not exists files (
            file_path text
        );",
        (),
    )?;

    conn.execute(
        "create virtual table if not exists vss_files using vss0 (
            file_embedding(768)
        );",
        (),
    )?;
    Ok(conn)
}

pub fn add_embedding(conn: &Connection, path: &str, embedding: &Embedding) -> Result<(), Error> {
    // add embedding, get rowid
    conn.execute("insert into files (file_path) values (?1)", params![path])
        .map_err(check_err)?;

    let rowid = conn.last_insert_rowid();
    println!("adding path a success, {rowid}");

    // let vec: Vec<f32> = embedding.iter().cloned().collect();
    // println!("{:?}", vec.to_string());
    // let string_vec = embedding.to_string();
    // println!("{}", string_vec.contains("..."));
    // println!("adding vector {string_vec}");

    conn.execute(
        // "insert into foo (bar, baz) values (?1, ?2)", ("apples", 5)).map_err(|err| {
        "insert into vss_files (rowid, file_embedding) values (?1, ?2)",
        params![rowid, to_byte_array(embedding)],
    )
    .map_err(check_err)?;

    // "insert into vss_files (file_embedding) values (?1)", params![to_byte_array(&embedding)])?;

    println!("adding file path a success");

    Ok(())
}

fn get_file_vector(conn: &Connection, embedding: &Embedding) -> Result<String, Error> {
    let res: String = conn.query_row(
        "select files.file_embedding from vss_files as files
        join (
            select rowid from vss_files where vss_search(
                file_embedding, ?1)
            limit 1
        ) as v on v.rowid = files.rowid;",
        params![to_byte_array(&embedding)],
        |row| row.get(0),
    )?;
    println!("file search retured {:?}", res);
    Ok(res)
}

fn semantic_file_search(conn: &Connection, embedding: &Embedding) -> Result<String, Error> {
    let res: String = conn.query_row(
        "select f.file_path from files as f
        join (
            select rowid from vss_files where vss_search(
                file_embedding, ?1)
            limit 1
        ) as v on v.rowid = f.rowid;",
        params![to_byte_array(&embedding)],
        |row| row.get(0),
    )?;
    println!("file search retured {res}");
    Ok(res)
}

fn to_byte_array(array: &Array1<f32>) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(array.len() * mem::size_of::<f32>());
    for &value in array.iter() {
        let mut buf = [0u8; mem::size_of::<f32>()]; // Create a buffer for each f32
        LittleEndian::write_f32(&mut buf, value); // Write the f32 into the buffer as bytes
        bytes.extend_from_slice(&buf); // Extend the bytes vector with the buffer
    }
    bytes
}

fn check_err(err: Error) -> Error {
    match &err {
        Error::SqlInputError { msg, sql, .. } => println!("input err {msg}, {sql}"),
        Error::SqliteFailure(err, msg) => {
            println!("sqlite failure {:?}, {:?}", err.to_string(), msg)
        }
        Error::ToSqlConversionFailure(err) => println!("conversion {:?}", err.to_string()),
        Error::InvalidQuery => println!("invalid query"),
        _ => println!("some other err"),
    };
    err
}
