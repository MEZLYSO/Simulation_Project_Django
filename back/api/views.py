from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
from io import StringIO, BytesIO
import zipfile
import math
import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
import json


# Variables globales para mantener el modelo entrenado en memoria
_nsl_kdd_model = None
_nsl_kdd_data_preparer = None
_nsl_kdd_feature_names = None

NSL_KDD_CATEGORICAL_COLS = [
    "protocol_type",
    "service",
    "flag",
    "land",
    "logged_in",
    "is_host_login",
    "is_guest_login",
]

# ==================== Clases de preparación de datos ====================

# Pipeline para atributos numéricos
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('rbst_scaler', RobustScaler()),
])

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-Hot Encoder personalizado que mantiene los nombres de columnas."""
    
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False)
        self._columns = None
    
    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self
    
    def transform(self, X, y=None):
        X_cat = X.select_dtypes(include=['object'])
        X_cat_oh = self._oh.transform(X_cat)
        return pd.DataFrame(
            X_cat_oh,
            columns=self._columns,
            index=X.index
        )
    
    def get_feature_names_out(self, input_features=None):
        """Retorna los nombres de las características después de la transformación."""
        return np.array(self._columns)


class DataFramePreparer(BaseEstimator, TransformerMixin):
    """Transformador que prepara todo el conjunto de datos."""
    
    def __init__(self):
        self._full_pipeline = None
        self._columns = None
    
    def fit(self, X, y=None):
        num_attribs = list(X.select_dtypes(exclude=['object']))
        cat_attribs = list(X.select_dtypes(include=['object']))
        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoder(), cat_attribs),
        ])
        self._full_pipeline.fit(X)
        
        # Obtener los nombres exactos de columnas del pipeline
        self._columns = list(self._full_pipeline.get_feature_names_out())
        return self
    
    def transform(self, X, y=None):
        X_copy = X.copy()
        X_prep = self._full_pipeline.transform(X_copy)
        
        # Devolver como DataFrame con los nombres correctos
        df_result = pd.DataFrame(
            X_prep,
            columns=self._columns,
            index=X_copy.index
        )
        return df_result


# ==================== Funciones auxiliares ====================

def load_kdd_dataset(data_path):
    """Lectura del Dataset NSL-KDD."""
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)


def cast_categorical_to_str(df):
    """Convierte columnas categóricas a string para mantener consistencia."""
    for col in NSL_KDD_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """Divide el DataFrame en train, validation y test sets."""
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    return (train_set, val_set, test_set)


def _ensure_nsl_kdd_model(force_retrain=False):
    """Entrena el modelo NSL-KDD si no existe o si se fuerza reentrenamiento."""
    global _nsl_kdd_model, _nsl_kdd_data_preparer, _nsl_kdd_feature_names
    
    if _nsl_kdd_model is not None and not force_retrain:
        return
    
    # Cargar dataset
    data_path = Path(__file__).resolve().parent.parent / "datasets" / "NSL-KDD" / "KDDTrain+.arff"
    if not data_path.exists():
        raise RuntimeError(f"Dataset no encontrado en {data_path}")
    
    df = load_kdd_dataset(str(data_path))
    df = cast_categorical_to_str(df)
    
    # Dividir el dataset
    train_set, _, _ = train_val_test_split(df, stratify='protocol_type')
    
    # Separar características y etiquetas
    X_df = df.drop("class", axis=1).copy()
    X_train = train_set.drop("class", axis=1).copy()
    y_train = train_set["class"].copy()
    
    # Guardar nombres de características
    _nsl_kdd_feature_names = list(X_df.columns)
    
    # Preparar el transformador con todo el dataset
    data_preparer = DataFramePreparer()
    data_preparer.fit(X_df)
    
    # Transformar conjunto de entrenamiento
    X_train_prep = data_preparer.transform(X_train)
    
    # Entrenar modelo de Regresión Logística con valores (sin nombres de columnas)
    clf = LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)
    clf.fit(X_train_prep.values, y_train)
    
    # Guardar en variables globales
    _nsl_kdd_model = clf
    _nsl_kdd_data_preparer = data_preparer


# ==================== Endpoints ====================


def _validate_split_sizes(train_size, val_size, test_size):
    total = train_size + val_size + test_size
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError("Las proporciones de train/val/test deben sumar 1.0")


def _split_dataset(df, train_size, val_size, test_size, stratify_col=None, random_state=42):
    stratify_series = df[stratify_col] if stratify_col and stratify_col in df.columns else None

    temp_size = val_size + test_size
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_size,
        shuffle=True,
        random_state=random_state,
        stratify=stratify_series
    )

    if temp_size == 0:
        return train_df, pd.DataFrame(columns=df.columns), pd.DataFrame(columns=df.columns)

    temp_stratify = None
    if stratify_col and stratify_col in temp_df.columns:
        temp_stratify = temp_df[stratify_col]

    relative_test_size = test_size / temp_size if temp_size else 0
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        shuffle=True,
        random_state=random_state,
        stratify=temp_stratify
    )

    return train_df, val_df, test_df


def _apply_missing_strategy(features_df, target_series, strategy):
    if strategy == "drop_rows":
        combined = features_df
        if target_series is not None:
            combined = pd.concat([features_df, target_series], axis=1)
        mask = combined.dropna().index
        features_df = features_df.loc[mask]
        if target_series is not None:
            target_series = target_series.loc[mask]
    elif strategy == "drop_columns":
        cols_with_na = [col for col in features_df.columns if features_df[col].isna().any()]
        features_df = features_df.drop(columns=cols_with_na)
    elif strategy in {"mean", "median"}:
        num_cols = features_df.select_dtypes(include=[np.number]).columns
        if not num_cols.empty:
            if strategy == "mean":
                fill_values = features_df[num_cols].mean()
            else:
                fill_values = features_df[num_cols].median()
            features_df[num_cols] = features_df[num_cols].fillna(fill_values)
    return features_df, target_series


def _apply_categorical_strategy(features_df, strategy):
    if strategy in {None, "", "none"}:
        return features_df

    cat_cols = features_df.select_dtypes(exclude=[np.number]).columns
    if not len(cat_cols):
        return features_df

    if strategy == "factorize":
        for col in cat_cols:
            codes, _ = pd.factorize(features_df[col])
            codes = pd.Series(codes, index=features_df.index)
            codes = codes.replace(-1, np.nan)
            features_df[col] = codes.astype(float)
    elif strategy == "ordinal":
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        original = features_df[cat_cols]
        na_mask = original.isna()
        encoded = encoder.fit_transform(original.fillna("__missing__"))
        encoded_df = pd.DataFrame(encoded, index=features_df.index, columns=cat_cols)
        encoded_df = encoded_df.replace(-1, np.nan)
        for col in cat_cols:
            encoded_df.loc[na_mask[col], col] = np.nan
        for col in cat_cols:
            features_df[col] = encoded_df[col]
    elif strategy in {"onehot", "get_dummies"}:
        features_df = pd.get_dummies(features_df, columns=cat_cols, dummy_na=False)
    else:
        raise ValueError(f"Estrategia de categorización desconocida: {strategy}")
    return features_df


def _scale_numeric(features_df, strategy):
    if strategy not in {"robust"}:
        return features_df
    num_cols = features_df.select_dtypes(include=[np.number]).columns
    if not len(num_cols):
        return features_df
    scaler = RobustScaler()
    features_df[num_cols] = scaler.fit_transform(features_df[num_cols])
    return features_df


def _ensure_arff_ready(df):
    df = df.copy()
    df.replace({'?': np.nan}, inplace=True)
    return df.replace({np.nan: None})


def _dataframe_to_arff(df, relation_name):
    clean_df = _ensure_arff_ready(df)
    attributes = []
    for column in clean_df.columns:
        series = clean_df[column]
        if pd.api.types.is_numeric_dtype(series):
            attr_type = "REAL"
        else:
            unique_vals = sorted({str(val) for val in series if val is not None})
            attr_type = unique_vals if unique_vals else ["__empty__"]
        attributes.append((column, attr_type))

    data_rows = []
    for _, row in clean_df.iterrows():
        converted_row = []
        for value in row:
            if isinstance(value, (np.generic,)):
                converted_row.append(value.item())
            else:
                converted_row.append(value)
        data_rows.append(converted_row)

    arff_obj = {
        "relation": relation_name,
        "attributes": attributes,
        "data": data_rows,
    }
    return arff.dumps(arff_obj)


def _preprocess_dataframe(df, options):
    df = df.copy()
    df.replace({'?': np.nan}, inplace=True)

    target_column = options.get("target_column", "class")
    if target_column not in df.columns:
        target_column = None

    target_series = df[target_column] if target_column else None
    features_df = df.drop(columns=[target_column]) if target_column else df

    missing_strategy = options.get("missing_strategy", "none")
    features_df, target_series = _apply_missing_strategy(features_df, target_series, missing_strategy)

    categorical_strategy = options.get("categorical_strategy", "none")
    features_df = _apply_categorical_strategy(features_df, categorical_strategy)

    scaling_strategy = options.get("scale_numeric", "none")
    features_df = _scale_numeric(features_df, scaling_strategy)

    if target_column:
        processed_df = pd.concat([features_df, target_series.loc[features_df.index]], axis=1)
    else:
        processed_df = features_df

    return processed_df


@csrf_exempt
def split_arff_dataset(request):
    """Divide un dataset ARFF en conjuntos train/val/test con transformaciones opcionales."""
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido. Use POST"}, status=405)

    upload = request.FILES.get("file")
    if upload is None:
        return JsonResponse({"error": "Debe adjuntar un archivo ARFF en el campo 'file'"}, status=400)

    options_raw = request.POST.get("options", "{}")
    try:
        options = json.loads(options_raw) if isinstance(options_raw, str) else options_raw
        if options is None:
            options = {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "El campo 'options' debe ser un JSON válido"}, status=400)

    try:
        file_content = upload.read().decode("utf-8", errors="ignore")
        dataset = arff.load(StringIO(file_content))
    except Exception as exc:
        return JsonResponse({"error": f"No se pudo procesar el archivo ARFF: {exc}"}, status=400)

    attributes = dataset.get("attributes")
    data_rows = dataset.get("data", [])
    if not attributes or not isinstance(attributes, list):
        return JsonResponse({"error": "El archivo ARFF no contiene la sección de atributos válida"}, status=400)

    attribute_names = [attr[0] for attr in attributes]
    df = pd.DataFrame(data_rows, columns=attribute_names)

    processed_df = _preprocess_dataframe(df, options)
    if processed_df.empty:
        return JsonResponse({"error": "El DataFrame resultante está vacío tras el preprocesamiento"}, status=400)

    train_size = float(options.get("train_size", 0.6))
    val_size = float(options.get("val_size", 0.2))
    test_size = float(options.get("test_size", 0.2))

    try:
        _validate_split_sizes(train_size, val_size, test_size)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    random_state = int(options.get("random_state", 42))

    stratify_opt = options.get("stratify", True)
    target_column = options.get("target_column", "class")
    stratify_col = None
    if isinstance(stratify_opt, bool) and stratify_opt:
        if target_column in processed_df.columns:
            stratify_col = target_column
    elif isinstance(stratify_opt, str) and stratify_opt in processed_df.columns:
        stratify_col = stratify_opt

    try:
        train_df, val_df, test_df = _split_dataset(
            processed_df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            stratify_col=stratify_col,
            random_state=random_state,
        )
    except ValueError as exc:
        return JsonResponse({"error": f"Error al dividir el dataset: {exc}"}, status=400)

    relation_name = dataset.get("relation") or Path(upload.name).stem or "dataset"

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{relation_name}_train.arff", _dataframe_to_arff(train_df, f"{relation_name}_train"))
        zf.writestr(f"{relation_name}_val.arff", _dataframe_to_arff(val_df, f"{relation_name}_val"))
        zf.writestr(f"{relation_name}_test.arff", _dataframe_to_arff(test_df, f"{relation_name}_test"))

    zip_buffer.seek(0)
    response = HttpResponse(zip_buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = f"attachment; filename={relation_name}_splits.zip"
    return response


def _normalize_records(records):
    """Convierte valores de numpy/pandas a tipos estándar para JSON."""
    normalized = []
    for row in records:
        clean_row = {}
        for key, value in row.items():
            if pd.isna(value):
                clean_row[key] = None
            elif isinstance(value, np.generic):
                clean_row[key] = value.item()
            else:
                clean_row[key] = value
        normalized.append(clean_row)
    return normalized


@csrf_exempt
def preview_arff_dataset(request):
    """Devuelve una vista previa de un archivo ARFF enviado por el cliente."""
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido. Use POST"}, status=405)

    upload = request.FILES.get("file")
    if upload is None:
        return JsonResponse({
            "error": "Debe adjuntar un archivo ARFF en el campo 'file'"
        }, status=400)

    try:
        file_content = upload.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return JsonResponse({
            "error": f"No se pudo leer el archivo proporcionado: {exc}"
        }, status=400)

    try:
        dataset = arff.load(StringIO(file_content))
    except Exception as exc:
        return JsonResponse({
            "error": f"No se pudo parsear el archivo ARFF: {exc}"
        }, status=400)

    attributes_raw = dataset.get("attributes")
    if not attributes_raw:
        return JsonResponse({"error": "El archivo ARFF no contiene atributos."}, status=400)

    attribute_names = [attr[0] for attr in attributes_raw]
    data_rows = dataset.get("data", [])
    df = pd.DataFrame(data_rows, columns=attribute_names)

    sample_records = df.replace({np.nan: None})
    sample = _normalize_records(sample_records.to_dict(orient="records"))

    attributes = []
    for name, attr_type in attributes_raw:
        if isinstance(attr_type, np.generic):
            attr_type = attr_type.item()
        elif isinstance(attr_type, list):
            attr_type = [item.item() if isinstance(item, np.generic) else item for item in attr_type]
        attributes.append({
            "name": name,
            "type": attr_type
        })

    dataset_name = dataset.get("relation") or Path(upload.name).stem

    response_payload = {
        "dataset_name": dataset_name,
        "total_rows": int(df.shape[0]),
        "attributes": attributes,
        "records": sample
    }

    return JsonResponse(response_payload)

@csrf_exempt
def predict_network_traffic(request):
    """
    Endpoint para predecir si un registro de red es normal o anómalo.
    
    Acepta JSON con las características de una conexión de red según NSL-KDD.
    
    Payload esperado:
    {
        "features": {
            "duration": 0,
            "protocol_type": "tcp",
            "service": "http",
            "flag": "SF",
            "src_bytes": 181,
            ... (todas las características del dataset NSL-KDD excepto 'class')
        },
        "retrain": false  // opcional, fuerza reentrenamiento
    }
    
    Respuesta:
    {
        "prediction": "normal" | "anomaly",
        "probability": {
            "normal": 0.85,
            "anomaly": 0.15
        }
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "Método no permitido. Use POST"}, status=405)
    
    try:
        payload = json.loads(request.body.decode("utf-8")) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({"error": "JSON inválido"}, status=400)
    
    features = payload.get("features")
    if not features or not isinstance(features, dict):
        return JsonResponse({
            "error": "Debe enviar 'features' con las características de la conexión"
        }, status=400)
    
    force_retrain = bool(payload.get("retrain", False))
    
    try:
        # Asegurar que el modelo está entrenado
        _ensure_nsl_kdd_model(force_retrain=force_retrain)
    except RuntimeError as exc:
        return JsonResponse({"error": str(exc)}, status=500)
    except Exception as exc:
        return JsonResponse({"error": f"Error al preparar el modelo: {exc}"}, status=500)
    
    # Verificar que todas las características necesarias están presentes
    missing_features = set(_nsl_kdd_feature_names) - set(features.keys())
    if missing_features:
        return JsonResponse({
            "error": f"Faltan características requeridas: {list(missing_features)}",
            "required_features": _nsl_kdd_feature_names
        }, status=400)
    
    try:
        # Crear DataFrame con los datos de entrada, asegurando el orden correcto
        # Reordenar los features según el orden del dataset original
        ordered_features = {key: features[key] for key in _nsl_kdd_feature_names}
        X_input = pd.DataFrame([ordered_features])
        X_input = cast_categorical_to_str(X_input)
        
        # Preparar los datos usando el mismo transformador
        X_input_prep = _nsl_kdd_data_preparer.transform(X_input)
        
        # Realizar predicción (convertir a valores sin nombres de columnas)
        prediction = _nsl_kdd_model.predict(X_input_prep.values)[0]
        probabilities = _nsl_kdd_model.predict_proba(X_input_prep.values)[0]
        
        classes = list(_nsl_kdd_model.classes_)
        prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        return JsonResponse({
            "prediction": prediction,
            "probability": prob_dict
        })
        
    except Exception as exc:
        return JsonResponse({"error": f"Error al procesar predicción: {exc}"}, status=500)
