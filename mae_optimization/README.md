# MAE Optimization

Отдельный модульный пайплайн поверх неизменяемых `avito_raw.csv` и
`cian_raw.csv`.

## Этапы

1. `prepare_data.py`
   - объединяет raw-файлы;
   - запускает существующую стандартизацию признаков;
   - добавляет OSM-признаки из `apartments_ml_osm_1000m_fast.csv`;
   - добавляет нормализованный адрес, адресную зону, ЖК, корпус, метро и
     геоячейки.
2. `tune_models.py`
   - необязательный отдельный Optuna-поиск;
   - настраивает модель полной цены и модель цены за квадратный метр;
   - оптимизирует MAE в рублях на validation.
3. `train_models.py`
   - обучает две независимые CatBoost-модели;
   - выбирает вес ансамбля только на validation;
   - обучает evaluation-модели на train+valid;
   - обучает production-модели на всех очищенных строках.
4. `evaluate_models.py`
   - считает общие и сегментные метрики на отложенном test.
5. `data_diagnostics.py`
   - измеряет разброс цен среди почти одинаковых объектов;
   - проверяет опасную утечку через цену за квадратный метр.
6. `text_experiment.py`
   - отдельно проверяет очищенный текст объявлений;
   - не меняет рабочий ансамбль, если validation не улучшается.

## Запуск

Из `E:\CR`:

```powershell
python -m mae_optimization.prepare_data
python -m mae_optimization.train_models --seeds 42
python -m mae_optimization.evaluate_models
python -m mae_optimization.data_diagnostics
```

Полный запуск одной командой:

```powershell
python -m mae_optimization.run_pipeline --seeds 42
```

Bagging нескольких seed:

```powershell
python -m mae_optimization.run_pipeline --seeds 11,42,73,104,135
```

Новый Optuna-поиск:

```powershell
python -m mae_optimization.tune_models --trials 30
python -m mae_optimization.train_models --seeds 42
python -m mae_optimization.evaluate_models
```

## Защита от утечки

`price_m2_rub` сохраняется в подготовленном CSV только для аудита. В
`modeling.py` есть обязательная проверка, не позволяющая использовать
`price_rub`, `price_log_rub`, `price_m2_rub` или производную целевую цену как
признак.

Текущая честная цель пайплайна: уменьшить исходный MAE около `1.303 млн руб.`
за счёт адресных признаков и ансамбля двух разных целевых представлений.

Полученный результат на отложенном test: **MAE 1 197 757 руб.**, median absolute
error **542 777 руб.**, MAPE **9,71%**.