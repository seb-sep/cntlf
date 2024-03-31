use rusqlite::{Connection, Result, LoadExtensionGuard};


#[tauri::command]
pub fn db_fun() -> Result<String, String> {
    test_db().map_err(|err| {eprintln!("{err}"); err.to_string()})?;
    Ok("".into())
}


fn test_db() -> Result<(), rusqlite::Error> {
    let conn = Connection::open("assets/db.sqlite")?;
    println!("{:?}", conn);
    unsafe {
        let _guard = LoadExtensionGuard::new(&conn)?;
        conn.load_extension("assets/sqlite-vss-m1/vector0.dylib", None)?;
        conn.load_extension("assets/sqlite-vss-m1/vss0.dylib", None)?;
        println!("loading extensions success {:?}", conn);
    }

    let res = conn.execute(
        "create virtual table vss_files using vss0(
            file_embedding(784),
        );", ()
    )?;
    println!("{res}");

    let res = conn.execute(
        "create table files (
            embedding_id integer,
            file_path text,
            foreign key (embedding_id) references vss_files(rowid)
        );", ()
    )?;
    println!("{res}");



    Ok(())
}