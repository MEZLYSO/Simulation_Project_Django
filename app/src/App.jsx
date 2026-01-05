import { useMemo, useState } from "react";

const API_BASE_URL = "http://localhost:8000/api";

const DEFAULT_FEATURES_JSON = `{
  "duration": 0,
  "protocol_type": "tcp",
  "service": "http",
  "flag": "SF",
  "src_bytes": 181,
  "dst_bytes": 5450,
  "land": 0,
  "wrong_fragment": 0,
  "urgent": 0,
  "hot": 0,
  "num_failed_logins": 0,
  "logged_in": 1,
  "num_compromised": 0,
  "root_shell": 0,
  "su_attempted": 0,
  "num_root": 0,
  "num_file_creations": 0,
  "num_shells": 0,
  "num_access_files": 0,
  "num_outbound_cmds": 0,
  "is_host_login": 0,
  "is_guest_login": 0,
  "count": 8,
  "srv_count": 8,
  "serror_rate": 0.0,
  "srv_serror_rate": 0.0,
  "rerror_rate": 0.0,
  "srv_rerror_rate": 0.0,
  "same_srv_rate": 1.0,
  "diff_srv_rate": 0.0,
  "srv_diff_host_rate": 0.0,
  "dst_host_count": 9,
  "dst_host_srv_count": 9,
  "dst_host_same_srv_rate": 1.0,
  "dst_host_diff_srv_rate": 0.0,
  "dst_host_same_src_port_rate": 0.11,
  "dst_host_srv_diff_host_rate": 0.0,
  "dst_host_serror_rate": 0.0,
  "dst_host_srv_serror_rate": 0.0,
  "dst_host_rerror_rate": 0.0,
  "dst_host_srv_rerror_rate": 0.0
}`;

const missingOptions = [
  { value: "none", label: "Sin cambios" },
  { value: "drop_rows", label: "Eliminar filas con nulos" },
  { value: "drop_columns", label: "Eliminar columnas con nulos" },
  { value: "mean", label: "Rellenar con media" },
  { value: "median", label: "Rellenar con mediana" },
];

const categoricalOptions = [
  { value: "none", label: "Sin cambios" },
  { value: "factorize", label: "Factorizar" },
  { value: "ordinal", label: "Ordinal encoder" },
  { value: "onehot", label: "One-Hot encoder" },
  { value: "get_dummies", label: "Pandas get_dummies" },
];

const scaleOptions = [
  { value: "none", label: "Sin escalado" },
  { value: "robust", label: "RobustScaler" },
];

const ViewWrapper = ({ title, description, children }) => (
  <section className="mx-auto mt-8 w-full max-w-4xl rounded-xl border border-slate-200 bg-white p-6 shadow-sm">
    <header className="mb-6">
      <h2 className="text-xl font-semibold text-slate-800">{title}</h2>
      {description && <p className="mt-1 text-sm text-slate-500">{description}</p>}
    </header>
    {children}
  </section>
);

const FieldLabel = ({ htmlFor, children }) => (
  <label htmlFor={htmlFor} className="text-sm font-medium text-slate-700">
    {children}
  </label>
);

const Button = ({ children, disabled, variant = "primary", type = "button", ...props }) => {
  const base = "inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-semibold transition";
  const variants = {
    primary:
      "bg-indigo-600 text-white hover:bg-indigo-700 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:bg-indigo-300",
    secondary:
      "bg-white text-slate-700 hover:bg-slate-50 border border-slate-200 disabled:text-slate-400 disabled:bg-slate-100",
  };
  return (
    <button
      disabled={disabled}
      type={type}
      className={`${base} ${variants[variant]}`}
      {...props}
    >
      {children}
    </button>
  );
};

const RecordsTable = ({ records }) => {
  const columns = useMemo(() => {
    if (!records || !records.length) return [];
    return Object.keys(records[0]);
  }, [records]);

  if (!records || !records.length) {
    return <p className="text-sm text-slate-500">Sin registros disponibles.</p>;
  }

  return (
    <div className="max-h-96 overflow-auto rounded-lg border border-slate-200">
      <table className="min-w-full text-left text-sm text-slate-700">
        <thead className="bg-slate-100">
          <tr>
            {columns.map((col) => (
              <th key={col} className="sticky top-0 whitespace-nowrap px-4 py-2 font-semibold text-slate-600">
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {records.map((row, idx) => (
            <tr key={idx} className="odd:bg-white even:bg-slate-50">
              {columns.map((col) => (
                <td key={col} className="px-4 py-2 align-top text-xs text-slate-700">
                  {row[col] === null || row[col] === undefined ? "" : String(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

const Navigation = ({ activeView, onNavigate }) => {
  const buttons = [
    { key: "preview", label: "Vista previa" },
    { key: "split", label: "División" },
    { key: "predict", label: "Predicción" },
  ];

  return (
    <nav className="mx-auto flex max-w-4xl items-center justify-center gap-3 px-4">
      {buttons.map((btn) => (
        <button
          key={btn.key}
          onClick={() => onNavigate(btn.key)}
          className={`w-full rounded-lg border px-4 py-2 text-sm font-semibold transition ${
            activeView === btn.key
              ? "border-indigo-500 bg-indigo-600 text-white"
              : "border-slate-300 bg-white text-slate-600 hover:border-indigo-300 hover:text-indigo-600"
          }`}
        >
          {btn.label}
        </button>
      ))}
    </nav>
  );
};

function App() {
  const [activeView, setActiveView] = useState("preview");
  const [previewFile, setPreviewFile] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewResult, setPreviewResult] = useState(null);
  const [previewError, setPreviewError] = useState("");
  const [previewDisplayLimit, setPreviewDisplayLimit] = useState(25);

  const previewRecords = useMemo(() => {
    if (!previewResult?.records) return [];
    return previewResult.records.slice(0, previewDisplayLimit);
  }, [previewResult, previewDisplayLimit]);

  const handlePreviewSubmit = async (event) => {
    event.preventDefault();
    if (!previewFile) {
      setPreviewError("Selecciona un archivo ARFF antes de enviar.");
      return;
    }

    setPreviewError("");
    setPreviewResult(null);
    setPreviewLoading(true);

    try {
      const formData = new FormData();
      formData.append("file", previewFile);

      const response = await fetch(`${API_BASE_URL}/preview-dataset/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || "Error al obtener la vista previa");
      }

      const data = await response.json();
      setPreviewResult(data);
    } catch (error) {
      setPreviewError(error.message || "Error inesperado al consultar el endpoint.");
    } finally {
      setPreviewLoading(false);
    }
  };

  const initialSplitOptions = {
    targetColumn: "class",
    missingStrategy: "none",
    categoricalStrategy: "none",
    scaleNumeric: "none",
    trainSize: 0.6,
    valSize: 0.2,
    testSize: 0.2,
    stratify: true,
    randomState: 42,
  };

  const [splitFile, setSplitFile] = useState(null);
  const [splitOptions, setSplitOptions] = useState(initialSplitOptions);
  const [splitLoading, setSplitLoading] = useState(false);
  const [splitMessage, setSplitMessage] = useState("");
  const [splitError, setSplitError] = useState("");

  const handleSplitOptionChange = (field, value) => {
    setSplitOptions((prev) => ({ ...prev, [field]: value }));
  };

  const handleSplitSubmit = async (event) => {
    event.preventDefault();
    if (!splitFile) {
      setSplitError("Selecciona un archivo ARFF antes de enviar.");
      return;
    }

    setSplitLoading(true);
    setSplitError("");
    setSplitMessage("");

    const optionsPayload = {
      target_column: splitOptions.targetColumn || undefined,
      missing_strategy: splitOptions.missingStrategy,
      categorical_strategy: splitOptions.categoricalStrategy,
      scale_numeric: splitOptions.scaleNumeric,
      train_size: Number(splitOptions.trainSize),
      val_size: Number(splitOptions.valSize),
      test_size: Number(splitOptions.testSize),
      stratify:
        splitOptions.stratify === "column"
          ? splitOptions.stratifyColumn || splitOptions.targetColumn
          : Boolean(splitOptions.stratify),
      random_state: Number(splitOptions.randomState) || 42,
    };

    try {
      const formData = new FormData();
      formData.append("file", splitFile);
      formData.append("options", JSON.stringify(optionsPayload));

      const response = await fetch(`${API_BASE_URL}/split-dataset/`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        try {
          const errorJson = JSON.parse(text);
          throw new Error(errorJson.error || text);
        } catch (parseError) {
          throw new Error(text || "Error al dividir el dataset");
        }
      }

      const blob = await response.blob();
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const filename = `${splitFile.name.replace(/\.arff$/i, "") || "dataset"}_splits.zip`;
      link.href = downloadUrl;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(downloadUrl);

      setSplitMessage("Descarga iniciada correctamente.");
    } catch (error) {
      setSplitError(error.message || "Error inesperado al dividir el dataset.");
    } finally {
      setSplitLoading(false);
    }
  };

  const [featuresJson, setFeaturesJson] = useState(DEFAULT_FEATURES_JSON);
  const [retrainModel, setRetrainModel] = useState(false);
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState("");
  const [predictResult, setPredictResult] = useState(null);

  const handlePredictSubmit = async (event) => {
    event.preventDefault();
    setPredictError("");
    setPredictResult(null);

    let parsedFeatures;
    try {
      parsedFeatures = JSON.parse(featuresJson);
      if (typeof parsedFeatures !== "object" || Array.isArray(parsedFeatures)) {
        throw new Error("El contenido debe ser un objeto JSON con las características.");
      }
    } catch (error) {
      setPredictError(error.message || "JSON inválido.");
      return;
    }

    setPredictLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/predict-network/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: parsedFeatures, retrain: Boolean(retrainModel) }),
      });

      if (!response.ok) {
        const errorText = await response.text();
        try {
          const data = JSON.parse(errorText);
          throw new Error(data.error || errorText);
        } catch (parseError) {
          throw new Error(errorText || "Error al obtener la predicción");
        }
      }

      const data = await response.json();
      setPredictResult(data);
    } catch (error) {
      setPredictError(error.message || "Error inesperado al predecir.");
    } finally {
      setPredictLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-100 pb-16">
      <header className="bg-slate-900 py-10 text-white">
        <div className="mx-auto max-w-4xl px-4">
          <h1 className="text-3xl font-bold">Herramientas NSL-KDD</h1>
          <p className="mt-2 text-sm text-slate-300">Selecciona una vista para trabajar con los distintos endpoints.</p>
        </div>
      </header>

      <div className="mt-6">
        <Navigation activeView={activeView} onNavigate={setActiveView} />
      </div>

      <main className="px-4">
        {activeView === "preview" && (
          <ViewWrapper
            title="Vista previa de dataset ARFF"
            description="Sube un archivo ARFF para revisar atributos y registros antes de usarlo."
          >
            <form onSubmit={handlePreviewSubmit} className="space-y-4">
              <div className="flex flex-col gap-2 sm:flex-row sm:items-end">
                <div className="flex-1">
                  <FieldLabel htmlFor="previewFile">Archivo ARFF</FieldLabel>
                  <input
                    id="previewFile"
                    type="file"
                    accept=".arff"
                    onChange={(event) => setPreviewFile(event.target.files?.[0] ?? null)}
                    className="mt-1 w-full rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <FieldLabel htmlFor="previewLimit">Filas mostradas</FieldLabel>
                  <input
                    id="previewLimit"
                    type="number"
                    min={1}
                    max={500}
                    value={previewDisplayLimit}
                    onChange={(event) => setPreviewDisplayLimit(Number(event.target.value) || 1)}
                    className="w-24 rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  />
                </div>
                <Button type="submit" disabled={previewLoading}>
                  {previewLoading ? "Procesando..." : "Ver vista previa"}
                </Button>
              </div>
            </form>

            {previewError && <p className="mt-4 text-sm text-red-600">{previewError}</p>}

            {previewResult && (
              <div className="mt-6 space-y-4 text-sm text-slate-700">
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="rounded-lg bg-slate-50 p-4">
                    <p className="text-xs uppercase text-slate-400">Dataset</p>
                    <p className="mt-1 font-medium text-slate-700">{previewResult.dataset_name}</p>
                  </div>
                  <div className="rounded-lg bg-slate-50 p-4">
                    <p className="text-xs uppercase text-slate-400">Total de filas</p>
                    <p className="mt-1 font-medium text-slate-700">{previewResult.total_rows}</p>
                  </div>
                  <div className="rounded-lg bg-slate-50 p-4">
                    <p className="text-xs uppercase text-slate-400">Columnas</p>
                    <p className="mt-1 font-medium text-slate-700">{previewResult.attributes?.length ?? 0}</p>
                  </div>
                </div>

                <div>
                  <h3 className="mb-2 text-sm font-semibold text-slate-800">Atributos</h3>
                  <div className="flex flex-wrap gap-2">
                    {previewResult.attributes?.map((attr) => (
                      <span
                        key={attr.name}
                        className="rounded-full bg-indigo-50 px-3 py-1 text-xs font-medium text-indigo-700"
                      >
                        {attr.name}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="mb-2 text-sm font-semibold text-slate-800">Registros (mostrando {previewRecords.length})</h3>
                  <RecordsTable records={previewRecords} />
                </div>
              </div>
            )}
          </ViewWrapper>
        )}

        {activeView === "split" && (
          <ViewWrapper
            title="División y preparación"
            description="Crea conjuntos train/val/test aplicando las transformaciones disponibles."
          >
            <form onSubmit={handleSplitSubmit} className="space-y-5">
            <div className="flex flex-col gap-2">
              <FieldLabel htmlFor="splitFile">Archivo ARFF</FieldLabel>
              <input
                id="splitFile"
                type="file"
                accept=".arff"
                onChange={(event) => setSplitFile(event.target.files?.[0] ?? null)}
                className="mt-1 w-full rounded-lg border border-dashed border-slate-300 bg-slate-50 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <FieldLabel htmlFor="targetColumn">Columna objetivo</FieldLabel>
                <input
                  id="targetColumn"
                  type="text"
                  value={splitOptions.targetColumn}
                  onChange={(event) => handleSplitOptionChange("targetColumn", event.target.value)}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  placeholder="class"
                />
              </div>

              <div className="space-y-2">
                <FieldLabel htmlFor="missingStrategy">Manejo de nulos</FieldLabel>
                <select
                  id="missingStrategy"
                  value={splitOptions.missingStrategy}
                  onChange={(event) => handleSplitOptionChange("missingStrategy", event.target.value)}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                >
                  {missingOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2">
                <FieldLabel htmlFor="categoricalStrategy">Transformación categórica</FieldLabel>
                <select
                  id="categoricalStrategy"
                  value={splitOptions.categoricalStrategy}
                  onChange={(event) => handleSplitOptionChange("categoricalStrategy", event.target.value)}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                >
                  {categoricalOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="space-y-2">
                <FieldLabel htmlFor="scaleNumeric">Escalado numérico</FieldLabel>
                <select
                  id="scaleNumeric"
                  value={splitOptions.scaleNumeric}
                  onChange={(event) => handleSplitOptionChange("scaleNumeric", event.target.value)}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                >
                  {scaleOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <FieldLabel htmlFor="trainSize">Train</FieldLabel>
                <input
                  id="trainSize"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={splitOptions.trainSize}
                  onChange={(event) => handleSplitOptionChange("trainSize", event.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                />
              </div>
              <div>
                <FieldLabel htmlFor="valSize">Validation</FieldLabel>
                <input
                  id="valSize"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={splitOptions.valSize}
                  onChange={(event) => handleSplitOptionChange("valSize", event.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                />
              </div>
              <div>
                <FieldLabel htmlFor="testSize">Test</FieldLabel>
                <input
                  id="testSize"
                  type="number"
                  min={0}
                  max={1}
                  step={0.05}
                  value={splitOptions.testSize}
                  onChange={(event) => handleSplitOptionChange("testSize", event.target.value)}
                  className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                />
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <FieldLabel htmlFor="stratifyMode">Estrategia de estratificación</FieldLabel>
                <select
                  id="stratifyMode"
                  value={splitOptions.stratify === true ? "true" : splitOptions.stratify === false ? "false" : "column"}
                  onChange={(event) => {
                    const value = event.target.value;
                    if (value === "true") {
                      handleSplitOptionChange("stratify", true);
                    } else if (value === "false") {
                      handleSplitOptionChange("stratify", false);
                    } else {
                      handleSplitOptionChange("stratify", "column");
                    }
                  }}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                >
                  <option value="true">Usar columna objetivo</option>
                  <option value="false">No estratificar</option>
                  <option value="column">Columna personalizada</option>
                </select>
              </div>

              {splitOptions.stratify === "column" && (
                <div className="space-y-2">
                  <FieldLabel htmlFor="stratifyColumn">Nombre de la columna</FieldLabel>
                  <input
                    id="stratifyColumn"
                    type="text"
                    value={splitOptions.stratifyColumn ?? ""}
                    onChange={(event) => handleSplitOptionChange("stratifyColumn", event.target.value)}
                    className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                    placeholder="Ej. protocol_type"
                  />
                </div>
              )}
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <FieldLabel htmlFor="randomState">Random state</FieldLabel>
                <input
                  id="randomState"
                  type="number"
                  value={splitOptions.randomState}
                  onChange={(event) => handleSplitOptionChange("randomState", event.target.value)}
                  className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                />
              </div>
              <div className="flex items-end gap-3">
                <Button type="submit" disabled={splitLoading}>
                  {splitLoading ? "Procesando..." : "Generar ZIP"}
                </Button>
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() => {
                    setSplitOptions(initialSplitOptions);
                    setSplitError("");
                    setSplitMessage("");
                  }}
                >
                  Restablecer opciones
                </Button>
              </div>
            </div>
          </form>

            {splitError && <p className="mt-4 text-sm text-red-600">{splitError}</p>}
            {splitMessage && <p className="mt-4 text-sm text-emerald-600">{splitMessage}</p>}
          </ViewWrapper>
        )}

        {activeView === "predict" && (
          <ViewWrapper
            title="Predicción de tráfico"
            description="Envía un registro con todas las características para obtener la predicción del modelo."
          >
            <form onSubmit={handlePredictSubmit} className="space-y-4">
            <div className="space-y-2">
              <FieldLabel htmlFor="featuresJson">Características (JSON)</FieldLabel>
              <textarea
                id="featuresJson"
                value={featuresJson}
                onChange={(event) => setFeaturesJson(event.target.value)}
                rows={12}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm text-slate-800 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </div>

            <div className="flex items-center gap-2">
              <input
                id="retrain"
                type="checkbox"
                checked={retrainModel}
                onChange={(event) => setRetrainModel(event.target.checked)}
                className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
              <label htmlFor="retrain" className="text-sm text-slate-600">
                Forzar reentrenamiento antes de predecir
              </label>
            </div>

            <div className="flex gap-3">
              <Button type="submit" disabled={predictLoading}>
                {predictLoading ? "Consultando..." : "Predecir"}
              </Button>
              <Button
                type="button"
                variant="secondary"
                onClick={() => {
                  setFeaturesJson(DEFAULT_FEATURES_JSON);
                  setRetrainModel(false);
                  setPredictError("");
                  setPredictResult(null);
                }}
              >
                Restaurar ejemplo
              </Button>
            </div>
          </form>

            {predictError && <p className="mt-4 text-sm text-red-600">{predictError}</p>}

            {predictResult && (
              <div className="mt-6 grid gap-4 text-sm text-slate-700 md:grid-cols-2">
                <div className="rounded-lg bg-slate-50 p-4">
                  <p className="text-xs uppercase text-slate-400">Predicción</p>
                  <p className="mt-2 text-2xl font-semibold text-indigo-600">{predictResult.prediction}</p>
                </div>
                <div className="rounded-lg bg-slate-50 p-4">
                  <p className="text-xs uppercase text-slate-400">Probabilidades</p>
                  <div className="mt-2 space-y-1">
                    {predictResult.probability &&
                      Object.entries(predictResult.probability).map(([label, value]) => (
                        <div key={label} className="flex items-center justify-between text-sm">
                          <span className="font-medium text-slate-600">{label}</span>
                          <span className="text-slate-700">{(value * 100).toFixed(2)}%</span>
                        </div>
                      ))}
                  </div>
                </div>
              </div>
            )}
          </ViewWrapper>
        )}
      </main>
    </div>
  );
}

export default App;
