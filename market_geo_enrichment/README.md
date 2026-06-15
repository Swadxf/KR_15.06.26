# Market + Geo Enrichment

Отдельный экспериментальный контур поверх неизменяемых `avito_raw.csv` и
`cian_raw.csv`. Он не меняет проект `mae_optimization`.

## Что добавлено

### Расширенная география

- плотность объявлений в радиусах 250–10 000 м;
- инфраструктура OSM в радиусах 250, 500 и 750 м;
- расстояние до центра Москвы;
- приблизительное положение относительно МКАД;
- направление от центра Москвы.

Используется уже сохранённая локальная выгрузка
`apartments_ml_osm_1000m_fast_pois.csv`. Это позволяет воспроизводить результат
без новых сетевых запросов.

### Рыночные аналоги

Для каждой квартиры находятся похожие предложения по:

- географическому расстоянию;
- площади и комнатности;
- типу рынка;
- году и этажу дома;
- точному адресу и названию ЖК.

На validation используются только цены train. На test используются только цены
train + validation. Цена оцениваемой строки никогда не попадает в её аналоги.

### Ансамбль

1. Две CatBoost-модели прогнозируют полную цену и цену за квадратный метр.
2. Их веса выбираются на validation.
3. Прогноз по аналогам добавляется с весами по ценовым диапазонам.
4. Test не используется при выборе параметров или весов.

## Запуск

Из `E:\CR`:

```powershell
python -m market_geo_enrichment.run_pipeline --seeds 42
```

Быстрый исследовательский запуск без production-моделей:

```powershell
python -m market_geo_enrichment.run_pipeline --seeds 42 --skip-production
```

Просмотр итоговой метрики:

```powershell
python -m market_geo_enrichment.evaluate
```

Если evaluation уже выполнен, production-модели можно обучить без повторного
test-прогона:

```powershell
python -m market_geo_enrichment.train_production
```

## Артефакты

- `artifacts/prepared_market_geo.csv` — расширенные признаки;
- `artifacts/test_predictions.csv` — честные test-прогнозы;
- `artifacts/metrics.json` — общие и сегментные метрики;
- `artifacts/blend_policies.json` — веса ансамблей;
- `artifacts/comparables_reference.csv` — база аналогов для новых объектов;
- `artifacts/models` — evaluation и production CatBoost-модели.

Текущий результат: **MAE 1 111 152 руб.**, что на **86 606 руб.** лучше
предыдущего модульного пайплайна.
