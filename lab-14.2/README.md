# Archivos de apoyo · laboratorio 14.2

Versión final de los archivos de la guía 12 (churn con CRISP-ML(Q), parte 2). La guía explica qué hace cada uno, muestra el fragmento que importa y trae el comando para copiarlos a tu proyecto.

Se copian a la raíz de tu fork conservando la ruta: `lab-14.2/src/monitor.py` va a `src/monitor.py`.

Viven en la rama `lab-14.2-archivos`; la rama `main` es el punto de partida del laboratorio y no los incluye.

`.gitignore` y `.dockerignore` no están aquí: la guía indica las líneas que hay que agregarles.

- `.github/workflows/mlops.yml`
- `.github/workflows/model-monitoring.yml`
- `Dockerfile`
- `app/main.py`
- `infra/alb.tf`
- `infra/ecs.tf`
- `infra/envs/prod.tfvars`
- `infra/envs/staging.tfvars`
- `infra/iam.tf`
- `infra/main.tf`
- `infra/outputs.tf`
- `infra/registry/main.tf`
- `infra/variables.tf`
- `infra/vpc.tf`
- `monitoring/docker-compose.yml`
- `monitoring/grafana/dashboards/churn-monitoring.json`
- `monitoring/grafana/provisioning/dashboards/dashboard.yml`
- `monitoring/grafana/provisioning/datasources/datasource.yml`
- `monitoring/prometheus/alerts.yml`
- `monitoring/prometheus/prometheus.yml`
- `requirements-serve.txt`
- `scripts/deploy_ecs.sh`
- `scripts/simulate_traffic.py`
- `src/monitor.py`
- `tests/conftest.py`
- `tests/test_api.py`
- `tests/test_skew.py`

Esta carpeta se genera desde la solución del laboratorio. No la edites a mano.
