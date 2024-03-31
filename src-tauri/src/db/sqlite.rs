use rusqlite::{Connection, Result, LoadExtensionGuard, params, Error};
use ndarray::{ArrayBase, Array1, Dim, OwnedRepr};
use byteorder::{ByteOrder, LittleEndian};
use std::mem;

// #[tauri::command]
// pub fn db_fun() -> Result<String, String> {
    
// }


pub fn init_db() -> Result<Connection, Error> {
    // let conn = Connection::open("assets/db.sqlite")?;
    let conn = Connection::open_in_memory()?;
    
    // load extensions
    unsafe {
        let _guard = LoadExtensionGuard::new(&conn)?;
        conn.load_extension("assets/sqlite-vss-m1/vector0.dylib", None)?;
        conn.load_extension("assets/sqlite-vss-m1/vss0.dylib", None)?;
        println!("loading extensions success {:?}", conn);
    }

    conn.execute(
        "create table if not exists files (
            file_path text
        );", ()
    )?;

    conn.execute(
        "create virtual table if not exists vss_files using vss0 (
            file_embedding(784)
        );", ()
    )?;
    Ok(conn)
}

pub fn add_embedding(conn: &Connection, path: &str, embedding: &ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>>) -> Result<(), Error> {

    // add embedding, get rowid
    conn.execute(
        "insert into files (file_path) values (?1)", params![path]).map_err(check_err)?;
    
    let rowid = conn.last_insert_rowid();
    println!("adding path a success, {rowid}");

    conn.execute(
        // "insert into foo (bar, baz) values (?1, ?2)", ("apples", 5)).map_err(|err| {
        "insert into vss_files (rowid, file_embedding) values (?1, ?2)", 
        params![rowid, to_byte_array(&embedding)]).map_err(check_err)?;
    
        // "insert into vss_files (file_embedding) values (?1)", params![to_byte_array(&embedding)])?;



    println!("adding file path a success");

    Ok(())
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
        Error::SqliteFailure(err, msg) => println!("sqlite failure {:?}, {:?}", err.to_string(), msg),
        Error::ToSqlConversionFailure(err) => println!("conversion {:?}", err.to_string()),
        Error::InvalidQuery => println!("invalid query"),
        _ => println!("some other err")
    };
    err
}

