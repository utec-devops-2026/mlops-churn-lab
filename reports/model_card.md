# Model card · churn-predictor v2

## Uso previsto
Priorizar clientes para la campaña de retención. La salida es una probabilidad de
churn; la decisión de contactar se toma con el umbral 0.3.
No usar para negar servicios ni para decisiones sin revisión humana.

## Datos
Dataset `data/telco_customer_churn_mlops.csv`, 240 filas de train y 60 de test.
Features: tenure_months, monthly_charges, support_tickets_last_6m, gender, contract_type, internet_service, senior_citizen.

## Modelo
Tipo: random_forest dentro de un Pipeline de scikit-learn (preprocesamiento incluido).
Run de MLflow: `5f460d9d6c3844a3ac874428286ec580`.

## Métricas en test (umbral 0.3)
| F1 | Precision | Recall | ROC-AUC | Costo por cliente |
|---|---|---|---|---|
| 0.7013 | 0.5745 | 0.9 | 0.7989 | 43.33 |

Costo sin modelo: contactar a todos 50.0, no contactar a nadie 100.0.

## Quality gate
| Check | Valor | Requerido | Resultado |
|---|---|---|---|
| f1 | 0.7013 | >= 0.6 | OK |
| roc_auc | 0.7989 | >= 0.7 | OK |
| recall_gap | 0.1304 | <= 0.25 | OK |
| costo_vs_sin_modelo | 43.33 | < 50.0 | OK |

## Recall por segmento
| Segmento | Grupo | n | Recall |
|---|---|---|---|
| gender | Female | 31 | 0.9286 |
| gender | Male | 29 | 0.875 |
| senior_citizen | 0 | 50 | 0.8696 |
| senior_citizen | 1 | 10 | 1.0 |

## Limitaciones
Dataset pequeño (300 clientes): las métricas tienen alta varianza.
El umbral depende de los costos de negocio definidos en `configs/churn.yaml`; si cambian, hay que recalcularlo.
