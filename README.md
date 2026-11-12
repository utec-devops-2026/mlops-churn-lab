# MLOps Churn Lab

Punto de partida de los laboratorios 14.1 y 14.2: predicción de churn para una fintech siguiendo CRISP-ML(Q).

El proyecto funciona, sus tests pasan y tiene un bug que solo aparece en producción. Encontrarlo y eliminarlo es el primer paso del laboratorio.

## El caso

**Predicción de Churn para una Fintech**

Una fintech detectó un incremento en la pérdida de clientes (churn). El área de analítica ya construyó una primera solución: entrena un modelo, registra experimentos en MLflow, lo publica como API y tiene un pipeline en GitHub Actions. Todo corre y los tests pasan, pero el negocio desconfía de las predicciones y nadie puede explicar por qué se eligió ese modelo ni cuándo habría que reemplazarlo.

El reto del equipo es llevar esa solución a un nivel confiable siguiendo CRISP-ML(Q):

- Validar los datos antes de entrenar.
- Garantizar que la API prediga lo mismo que el modelo evaluado.
- Elegir el modelo con evidencia y decidir con criterios de negocio.
- Promover a producción solo lo que pasa un quality gate.
- Detectar cuándo el modelo deja de funcionar y reentrenarlo.
- Documentar riesgos, controles y decisiones.

## Cómo empezar

1. Haz **fork** de este repositorio a tu cuenta.
2. Clona tu fork (reemplaza `TU-USUARIO` por tu usuario de GitHub) y crea el entorno:

```bash
git clone https://github.com/TU-USUARIO/mlops-churn-lab.git
cd mlops-churn-lab
git checkout -b lab/14.1

conda create -n churn-lab python=3.10 -y
conda activate churn-lab
pip install -r requirements.txt "setuptools<81"
```

3. Comprueba que todo corre:

```bash
pytest tests/ -q                      # 18 passed
python src/data_prep.py
python src/train.py --model all
python src/evaluate_and_register.py
uvicorn app.main:app --port 8000      # http://localhost:8000/docs
```

4. Sigue la guía: [`docs/11. mlops-churn-crisp-mlq-parte-1.md`](docs/11.%20mlops-churn-crisp-mlq-parte-1.md).

## Estructura

```
.
├── app/main.py                     # API FastAPI: /predict, /predict/batch, /health
├── src/
│   ├── data_prep.py                # limpieza, encoding, escalado y split
│   ├── train.py                    # 4 modelos con MLflow tracking
│   └── evaluate_and_register.py    # registro en MLflow Model Registry
├── tests/                          # tests de la API con el modelo mockeado
├── data/
│   └── telco_customer_churn_mlops.csv   # 300 clientes, 7 features, target churn
├── docs/                           # guía del laboratorio
├── .github/workflows/mlops.yml     # test → data-prep → train → register → smoke-test
├── Dockerfile
└── requirements.txt
```

## Dataset

| Columna | Tipo | Descripción |
|---|---|---|
| `customer_id` | texto | Identificador del cliente |
| `gender` | categoría | Male, Female |
| `senior_citizen` | 0/1 | Adulto mayor |
| `tenure_months` | entero | Meses como cliente |
| `contract_type` | categoría | Month-to-Month, One Year, Two Year |
| `internet_service` | categoría | Fiber, DSL, None |
| `monthly_charges` | decimal | Cargo mensual |
| `support_tickets_last_6m` | entero | Tickets de soporte en 6 meses |
| `churn` | Yes/No | El cliente abandonó (target) |

## Qué vas a construir

| Laboratorio | Fases CRISP-ML(Q) | Resultado |
|---|---|---|
| 14.1 | 1 a 4: negocio y datos, preparación, modelado, evaluación | Contrato de datos, Pipeline de scikit-learn, validación cruzada, umbral por costo, modelo `@champion` |
| 14.2 | 5 y 6: despliegue, monitoreo | API sin skew, imagen en Docker Hub, drift, reentrenamiento automático |

## Comparación de candidatos

Validación cruzada estratificada de 5 folds sobre train. `GradientBoostingClassifier`
se ejecuta con `n_estimators=100`, `learning_rate=0.1`, `max_depth=3` y
`random_state=42`; esos hiperparámetros se registran en MLflow.

| Modelo | F1 CV (media ± std) | ROC-AUC CV (media ± std) |
|---|---:|---:|
| Gradient Boosting | 0.6476 ± 0.0761 | 0.6858 ± 0.0604 |

El mejor candidato sigue siendo `random_forest` (F1 CV 0.6900), por lo que
agregar Gradient Boosting no cambia la selección.
