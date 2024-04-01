import { useState } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";

import { dialog } from '@tauri-apps/api';

function App() {

  const [embedding, setEmbedding] = useState("");
  const [path, setPath] = useState("");
  const [input, setInput] = useState("");
 
  async function selectFile() {
    try {
      const selected = await dialog.open({
        // You can specify multiple options here, see the documentation for more details
        multiple: false, // Set to true if you want to allow multiple file selection
        directory: false, // Set to true to select directories instead of files
        // You can also specify filters for file types
      });
      console.log(selected); // This will log the path(s) of the selected file(s) or directory
      if (selected) {
        if (Array.isArray(selected)) {
          setPath(selected[0]);
        } else {
          setPath(selected);
        }
      }
    } catch (error) {
      console.error('Error selecting file:', error);
    }
  }

  async function findFile() {
    try {
      setEmbedding(await invoke("find_file", {query: input}));
    } catch (error) {
      console.error("error doing RAG:", error);
      setEmbedding(JSON.stringify(error));
    }
  }


  async function embed() {
    // Learn more about Tauri commands at https://tauri.app/v1/guides/features/command
    console.log('embedding in js');
    try {
      setEmbedding(await invoke("embed_file", { filePath: path }));
    } catch (e) {
      setEmbedding(JSON.stringify(e));
    }

  }

  return (
    <div className="container">
      <h1>Welcome to Tauri!</h1>

      <p>Enter a filepath to embed</p>

      <button onClick={selectFile}>Choose File</button>
      <div>{path}</div>
      <button onClick={embed}>Embed</button>
      <p>{embedding}</p>
      <input
        type="text"
        value={input}
        onChange={(e) => setInput(e.target.value)}
      />
      <button onClick={findFile}>Search</button>
    </div>
  );
}

export default App;
