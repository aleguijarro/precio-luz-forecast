# ⚡ Precio de la luz: previsión a 1–2 días

**Web en vivo → https://aleguijarro.github.io/precio-luz-forecast/**

Previsión hora a hora del precio regulado de la electricidad en España (PVPC) para mañana y pasado mañana, con intervalos de incertidumbre. Todo el sistema funciona sin servidores: un GitHub Action descarga los datos, reentrena el modelo, evalúa su precisión y publica la web en GitHub Pages dos veces al día.

## Qué problema resuelve

Red Eléctrica publica el precio oficial del día siguiente hacia las 20:15. Antes de esa hora, y para cualquier día posterior, no hay precio oficial. Esta app:

- muestra el precio **oficial** cuando ya está publicado;
- muestra una **previsión** cuando aún no lo está, con un intervalo del 80 %;
- resume lo útil para decidir: la **mejor franja de 3 horas**, la hora más barata y la más cara.

## Cómo funciona

```
apidatos.ree.es ──► data/pvpc.csv ──► matriz día × hora ──► features sin fuga de información
                                                            │
                         ┌──────────────────────────────────┘
                         ▼
     Gradient boosting cuantílico (p10 / p50 / p90), un modelo por horizonte k = 1, 2, 3 días
                         │
          backtest (60 días fuera de muestra) vs. 2 referencias ingenuas
                         │
                         ▼
          site/data/forecast.json ──► web estática en GitHub Pages
```

**Datos.** API pública de Red Eléctrica (`apidatos.ree.es`), sin clave. Histórico desde 2022, actualizado de forma incremental. Los días con cambio de hora (23 o 25 horas) se normalizan a 24.

**Features.** Para predecir el día *D* a *k* días vista solo se usa información de días ≤ *D − k*: precio de la misma hora (y de las horas vecinas), media / mínimo / máximo / dispersión de ese día, misma hora de la semana anterior, nivel medio de los últimos 7 días, y calendario (hora, día de la semana, mes, estacionalidad anual, festivos nacionales calculados incluido el Viernes Santo).

**Modelo.** `HistGradientBoostingRegressor` de scikit-learn con pérdida cuantílica, entrenado de forma directa para cada horizonte (no recursiva), lo que evita acumular errores. Los tres cuantiles se reordenan para garantizar p10 ≤ p50 ≤ p90.

**Evaluación honesta.** Cada ejecución reserva los últimos 60 días como test y compara el modelo con:
1. *persistencia*: repetir el precio de la misma hora de hace *k* días;
2. *persistencia semanal*: repetir el precio de hace 7 días.

Se publican el MAE, el % de mejora sobre la mejor referencia y la cobertura real del intervalo del 80 %. Además, cada previsión publicada se guarda en `data/forecast_history.csv` y se puntúa cuando llega el precio real, generando un historial de precisión **en vivo** (no solo de backtest). Las métricas completas están en `data/metrics.json`.

## Estructura

```
luzcast/
  data.py       descarga y almacenamiento de precios (REE)
  calendar.py   festivos nacionales de España
  features.py   matriz día × hora y features sin leakage
  model.py      modelos cuantílicos, backtest y previsión
  pipeline.py   job diario: datos → evaluación → entrenamiento → JSON
site/           web estática (HTML, CSS y JS sin dependencias)
tests/          tests unitarios (pytest)
.github/workflows/
  update.yml    ejecución programada + despliegue en GitHub Pages
  ci.yml        lint (ruff) + tests en cada push
```

## Ejecutarlo en local (opcional)

```bash
pip install -r requirements.txt
python -m luzcast            # descarga datos, entrena y genera site/data/forecast.json
python -m http.server -d site 8000
pytest
```

## Próximos pasos

- Añadir previsiones de demanda, eólica y solar (API de ESIOS, requiere token gratuito), que son los principales motores del precio.
- Añadir el precio de los futuros de gas (MIBGAS) como variable exógena.
- Un asistente conversacional que use estas previsiones como herramienta ("¿cuándo pongo la lavadora mañana?").

---

Proyecto de **Alejandro Guijarro Santiago** · BSc Cognitive Science & AI, Tilburg University.
Información orientativa; no constituye asesoramiento.
