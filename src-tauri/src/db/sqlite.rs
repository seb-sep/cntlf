use rusqlite::{Connection, Result, LoadExtensionGuard};


#[tauri::command]
pub fn db_fun() -> Result<String, String> {
    test_db().map_err(|err| err.to_string())?;
    Ok("".into())
}


fn test_db() -> Result<(), rusqlite::Error> {
    let conn = Connection::open_in_memory().unwrap();
    println!("{:?}", conn);
    unsafe {
        let _guard = LoadExtensionGuard::new(&conn)?;
        conn.load_extension("assets/sqlite-vss-m1/vector0.dylib", None)?;
        conn.load_extension("assets/sqlite-vss-m1/vss0.dylib", None)?;
    }

    let res = conn.execute(
        "create virtual table vss_articles using vss0(
            headline_embedding(384),
            description_embedding(384),
        );", ()
    )?;
    println!("{res}");
    Ok(())
}