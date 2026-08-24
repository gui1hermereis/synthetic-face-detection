import { useState } from "react";

const API_BASE_URL = import.meta.env.API_URL || "";
const API_KEY = import.meta.env.API_KEY || "";

const initialResult = null;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(initialResult);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  function selectFile(file) {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (!file) {
      setSelectedFile(null);
      setResult(initialResult);
      setError("");
      setPreviewUrl("");
      return;
    }

    if (!file.type.startsWith("image/")) {
      setError("Envie um arquivo de imagem.");
      return;
    }

    setSelectedFile(file);
    setResult(initialResult);
    setError("");

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
      return;
    }

    setPreviewUrl("");
  }

  function handleFileChange(event) {
    selectFile(event.target.files?.[0] ?? null);
  }

  function handleDragOver(event) {
    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event) {
    event.preventDefault();
    setIsDragging(false);
  }

  function handleDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (!selectedFile) {
      setError("Selecione uma imagem antes de enviar.");
      return;
    }

    if (!API_KEY.trim()) {
      setError("A API key nao foi configurada no arquivo .env do frontend.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    setIsLoading(true);
    setError("");
    setResult(initialResult);

    try {
      const response = await fetch(`${API_BASE_URL}/images/analyze`, {
        method: "POST",
        headers: {
          "X-API-Key": API_KEY.trim(),
        },
        body: formData,
      });

      const payload = await response.json();

      if (!response.ok) {
        const message = payload?.error?.message || "Nao foi possivel analisar a imagem.";
        throw new Error(message);
      }

      setResult(payload);
    } catch (submitError) {
      setError(submitError.message || "Erro inesperado ao enviar a imagem.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="page-shell">
      <section className="hero-panel">
        <header className="hero-copy">
          <p className="eyebrow">Synthetic Face Detection</p>
          <h1>Analise uma imagem facial</h1>
          <p className="description">
            Envie uma foto, confira a predicao do modelo e veja os sinais de
            qualidade usados antes da inferencia.
          </p>
        </header>

        <div className="workspace">
          <form className="upload-card" onSubmit={handleSubmit}>
            

            <label
              className={`dropzone${isDragging ? " dropzone-active" : ""}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <input type="file" accept="image/*" onChange={handleFileChange} />
              <strong>{selectedFile ? selectedFile.name : "Arraste e solte sua imagem aqui"}</strong>
              <small>
                {selectedFile
                  ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
                  : "Ou clique para selecionar JPG, PNG e outros formatos de imagem."}
              </small>
            </label>

            <div className="upload-summary">
              <span>Arquivo</span>
              <strong>{selectedFile ? "Selecionado" : "Aguardando imagem"}</strong>
            </div>

            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? "Analisando..." : "Analisar imagem"}
            </button>

            {error ? <div className="feedback error">{error}</div> : null}
          </form>

          <section className="result-card">
            {previewUrl ? (
              <img className="preview-image" src={previewUrl} alt="Preview da imagem selecionada" />
            ) : null}

            {!result ? (
              <div className="placeholder">
                <h2>{selectedFile ? "Pronto para analisar" : "Resultado"}</h2>
                <p>
                  {selectedFile
                    ? "Clique em analisar imagem para enviar o arquivo para a API."
                    : "Depois do envio, a classificacao do modelo aparecera aqui."}
                </p>
              </div>
            ) : (
              <div className="result-content">
                <div className="result-header">
                  <h2>
                    {result.prediction.label === "synthetic"
                      ? "Imagem artificial"
                      : "Imagem real"}
                  </h2>
                  <strong>{(result.prediction.confidence * 100).toFixed(2)}%</strong>
                </div>

                <div className="stats-grid">
                  <article>
                    <span>Arquivo</span>
                    <strong>{result.filename}</strong>
                  </article>
                  <article>
                    <span>Resolucao</span>
                    <strong>{result.image.width} x {result.image.height}</strong>
                  </article>
                  <article>
                    <span>Rostos detectados</span>
                    <strong>{result.face_detection.faces_detected}</strong>
                  </article>
                  <article>
                    <span>Blur score</span>
                    <strong>{result.quality.blur_score}</strong>
                  </article>
                </div>

                <div className="probabilities">
                  <h3>Probabilidades</h3>
                  {Object.entries(result.prediction.probabilities).map(([label, value]) => (
                    <div className="probability-row" key={label}>
                      <span>{label}</span>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width: `${value * 100}%` }} />
                      </div>
                      <strong>{(value * 100).toFixed(2)}%</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </section>
    </main>
  );
}

export default App;
