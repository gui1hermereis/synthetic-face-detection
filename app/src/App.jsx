import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000/api/v1";
const API_KEY = import.meta.env.VITE_API_KEY || "";

const initialResult = null;

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(initialResult);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  function handleFileChange(event) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setResult(initialResult);
    setError("");

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (file) {
      setPreviewUrl(URL.createObjectURL(file));
      return;
    }

    setPreviewUrl("");
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
        <div className="hero-copy">
          <p className="eyebrow">Synthetic Face Detection</p>
          <h1>Envie uma imagem facial para verificar se ela e real ou artificial</h1>
          <p className="description">
            A pagina envia a imagem para a API e mostra o resultado da
            classificacao com confianca, qualidade e dados da deteccao facial.
          </p>
        </div>

        <div className="workspace">
          <form className="upload-card" onSubmit={handleSubmit}>
            <div className="env-hint">
              <span>Autenticacao configurada por ambiente</span>
              <small>
                O frontend usa <code>VITE_API_KEY</code> e <code>VITE_API_BASE_URL</code>
                do arquivo <code>.env</code>.
              </small>
            </div>

            <label className="dropzone">
              <input type="file" accept="image/*" onChange={handleFileChange} />
              <span>{selectedFile ? selectedFile.name : "Clique para selecionar uma imagem"}</span>
              <small>Formatos comuns como JPG e PNG funcionam melhor.</small>
            </label>

            <button className="primary-button" type="submit" disabled={isLoading}>
              {isLoading ? "Analisando..." : "Analisar imagem"}
            </button>

            {error ? <div className="feedback error">{error}</div> : null}
          </form>

          <section className="result-card">
            {previewUrl ? <img className="preview-image" src={previewUrl} alt="Preview da imagem selecionada" /> : null}

            {!result ? (
              <div className="placeholder">
                <h2>Resultado</h2>
                <p>Depois do envio, a classificacao do modelo aparecera aqui.</p>
              </div>
            ) : (
              <div className="result-content">
                <div className="result-header">
                  <h2>{result.prediction.label === "synthetic" ? "Imagem artificial" : "Imagem real"}</h2>
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
