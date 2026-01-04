# Simulation_Project_Django

## API Backend

La aplicación Django expone endpoints para inferencia sobre el dataset NSL-KDD.

### Vista previa de dataset ARFF

- **Ruta:** `/api/preview-dataset/`
- **Método:** `POST`
- **Headers:** `Content-Type: multipart/form-data`
- **Body:**
	- Campo de archivo `file` con el dataset `.arff`.
- **Nota:** se devuelven todos los registros del archivo, por lo que la respuesta puede ser grande.
- **Respuesta:**
	```json
	{
		"dataset_name": "KDDTrain+",
		"total_rows": 125973,
		"attributes": [
			{"name": "duration", "type": "REAL"},
			{"name": "protocol_type", "type": ["tcp", "udp", "icmp"]}
		],
		"records": [
			{
				"duration": 0,
				"protocol_type": "tcp",
				"service": "http",
				"flag": "SF",
				"src_bytes": 181,
				"dst_bytes": 5450,
				"class": "normal"
			}
		]
	}
	```

### Predicción de tráfico de red

- **Ruta:** `/api/predict-network/`
- **Método:** `POST`
- **Headers:** `Content-Type: application/json`
- **Body:**
	```json
	{
			"features": {
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
			},
			"retrain": false
	}
	```
- **Respuesta:**
	```json
	{
			"prediction": "normal",
			"probability": {
					"anomaly": 0.12,
					"normal": 0.88
			}
	}
	```

El primer request entrena el modelo (puede tardar ~20 s); siguientes peticiones reutilizan el modelo en memoria.

### Ejemplo con `curl`

```bash
curl -X POST http://localhost:8000/api/predict-network/ \
	-H "Content-Type: application/json" \
	-d '@payload.json'
```
