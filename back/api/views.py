from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
from io import StringIO
import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
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
