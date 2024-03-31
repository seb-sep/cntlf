import { useState } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";

function App() {
  const [embedding, setEmbedding] = useState("");
  const [path, setPath] = useState("");

  async function embed() {
    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
    console.log('embedding in js');
    setEmbedding(await invoke("embed_file", { filePath: path }));
  }

  async function testSql() {
    setEmbedding(await invoke("db_fun"));
  }


  return (
    <div className="container">
      <h1>Welcome to Tauri!</h1>

      <p>Enter a filepath to embed</p>

      <form
        className="row"
        onSubmit={(e) => {
          e.preventDefault();
          embed();
        }}
      >
        <input
          id="greet-input"
          onChange={(e) => setPath(e.currentTarget.value)}
          placeholder="Enter a path..."
        />
        <button type="submit">Greet</button>
      </form>

      <p>{embedding}</p>
      <button onClick={testSql}>Test sqlite loading</button>
    </div>
  );
}

export default App;
