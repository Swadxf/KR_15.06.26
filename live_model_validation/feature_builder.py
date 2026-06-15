# - бинарные/multi-hot признаки хранятся как 0/1;
# - взаимоисключающие категориальные признаки остаются строками;
# - числовые признаки приведены к float/int.

import re
import os
import math
import warnings
import numpy as np
import pandas as pd
from typing import Any, Iterable, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)


AVITO_PATH = "avito_raw.csv"
CIAN_PATH = "cian_raw.csv"
OUTPUT_PATH = "apartments_ml.csv"


GEO_OSM_PATH = "apartments_ml_osm_1000m_fast.csv"
GEO_RADIUS_M = 1000
MERGE_GEO_FEATURES = True


BASE_YEAR = 2026


def read_csv_safe(path: str, source_name: str) -> pd.DataFrame:
    """
    Читает CSV Avito/CIAN.
    on_bad_lines='skip' нужен для строк, где описание могло сломать структуру через лишний ;
    в исходных выгрузках такое встречается.
    """
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        on_bad_lines="skip",
        low_memory=False,
    )
    if "source" not in df.columns:
        df["source"] = source_name
    df["source"] = df["source"].fillna(source_name)
    return df


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        if pd.isna(x):
            return True
    except Exception:
        pass
    s = str(x).strip().lower()
    return s in {
        "",
        "nan",
        "none",
        "null",
        "нет",
        "нет информации",
        "не указано",
        "не указан",
        "неизвестно",
        "—",
        "-",
    }


def normalize_text(x: Any) -> str:
    """Нормализация текста для регулярных выражений."""
    if is_missing(x):
        return ""
    s = str(x).lower()
    s = s.replace("ё", "е")
    s = s.replace("²", "2")
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def to_float(x: Any) -> Optional[float]:
    """
    Приводит строки вида:
    '37,6 м²', '8 826 621 ₽', '286 579 ₽/м²', '2,85 м'
    к float.
    """
    if is_missing(x):
        return None
    s = str(x).lower().strip()
    s = s.replace("\xa0", " ")
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in {"", ".", "-", "-."}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def to_int(x: Any) -> Optional[int]:
    v = to_float(x)
    if v is None or not np.isfinite(v):
        return None
    return int(round(v))


def coalesce_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    """Берёт первое непустое значение из списка колонок."""
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([np.nan] * len(df), index=df.index)

    tmp = df[existing].copy()
    for c in existing:
        tmp[c] = tmp[c].where(~tmp[c].apply(is_missing), np.nan)
    return tmp.bfill(axis=1).iloc[:, 0]


def first_not_missing_values(*values: Any) -> Any:
    for v in values:
        if not is_missing(v):
            return v
    return np.nan


def parse_number_from_text(
    patterns: list[str],
    text: str,
    min_v: Optional[float] = None,
    max_v: Optional[float] = None,
) -> Optional[float]:
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if not m:
            continue
        val = to_float(m.group(1))
        if val is None:
            continue
        if min_v is not None and val < min_v:
            continue
        if max_v is not None and val > max_v:
            continue
        return val
    return None


def safe_div(a: Any, b: Any) -> Optional[float]:
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return None
        return float(a) / float(b)
    except Exception:
        return None


def extract_ceiling_height(text: Any) -> Optional[float]:
    text = normalize_text(text)
    return parse_number_from_text(
        [
            r"(?:высот[аы]\s+потолк\w*|потолк\w*)[^0-9]{0,35}(\d(?:[\.,]\d{1,2})?)\s*м",
            r"(\d(?:[\.,]\d{1,2})?)\s*м\s*(?:потолк\w*|высот[аы]\s+потолк\w*)",
        ],
        text,
        min_v=2.0,
        max_v=6.0,
    )


def extract_area(text: Any, area_type: str) -> Optional[float]:
    text = normalize_text(text)
    unit = r"(?:кв\.?\s*м|м2|квм|квадратн\w*\s*метр\w*)"

    if area_type == "total":
        patterns = [
            rf"(?:общая\s+площадь|площадь\s+квартиры|площадью)\D{{0,30}}(\d{{1,3}}(?:[\.,]\d{{1,2}})?)\s*[- ]?{unit}",
            rf"(\d{{1,3}}(?:[\.,]\d{{1,2}})?)\s*[- ]?{unit}\s*(?:общая|квартира)",
        ]
        return parse_number_from_text(patterns, text, min_v=10, max_v=500)

    if area_type == "kitchen":
        patterns = [
            rf"(?:кухн[яи]|кухня-гостиная|кухня гостиная)\D{{0,30}}(\d{{1,3}}(?:[\.,]\d{{1,2}})?)\s*[- ]?{unit}",
            rf"(\d{{1,3}}(?:[\.,]\d{{1,2}})?)\s*[- ]?{unit}\s*(?:кухн[яи]|кухня-гостиная|кухня гостиная)",
        ]
        return parse_number_from_text(patterns, text, min_v=3, max_v=100)

    if area_type == "living":
        patterns = [
            rf"(?:жилая\s+площадь|жил[аяые]\s+комнат[аы]?)\D{{0,30}}(\d{{1,3}}(?:[\.,]\d{{1,2}})?)\s*[- ]?{unit}",
            rf"(\d{{1,3}}(?:[\.,]\d{{1,2}})?)\s*[- ]?{unit}\s*(?:жилая|жил[аяые]\s+комнат[аы]?)",
        ]
        return parse_number_from_text(patterns, text, min_v=5, max_v=300)

    return None


def extract_rooms(text: Any) -> Optional[int]:
    text = normalize_text(text)
    if re.search(r"\bстудия\b|\bstudio\b", text):
        return 0
    m = re.search(r"(?:^|\D)([1-6])\s*[- ]?(?:комнатн\w+|комн\.?|к\.)", text)
    if m:
        return int(m.group(1))
    return None


def extract_floor_pair(text: Any) -> Tuple[Optional[int], Optional[int]]:
    text = normalize_text(text)


    m = re.search(
        r"(?:на\s+)?(\d{1,2})\s*[- ]?этаже?[^\.\n]{0,45}?(\d{1,2})\s*[- ]?этажн\w*\s+дом",
        text,
    )
    if m:
        floor, total = int(m.group(1)), int(m.group(2))
        if 1 <= floor <= total <= 100:
            return floor, total


    m = re.search(r"(?:^|\D)(\d{1,2})\s*(?:/|из)\s*(\d{1,2})(?:\D|$)", text)
    if m:
        floor, total = int(m.group(1)), int(m.group(2))
        if 1 <= floor <= total <= 100:
            return floor, total

    floor = None
    total = None

    m = re.search(r"(?:на\s+)?(\d{1,2})\s*[- ]?этаже?", text)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 100:
            floor = v

    m = re.search(r"(\d{1,2})\s*[- ]?этажн\w*\s+дом", text)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 100:
            total = v

    return floor, total


def parse_floor_raw(x: Any) -> Tuple[Optional[int], Optional[int]]:
    if is_missing(x):
        return None, None
    s = normalize_text(x)
    m = re.search(r"(\d{1,2})\s*(?:/|из)\s*(\d{1,2})", s)
    if m:
        floor, total = int(m.group(1)), int(m.group(2))
        if 1 <= floor <= total <= 100:
            return floor, total
    m = re.search(r"^(\d{1,2})$", s)
    if m:
        floor = int(m.group(1))
        if 1 <= floor <= 100:
            return floor, None
    return None, None


def extract_house_year(text: Any) -> Optional[int]:
    text = normalize_text(text)
    patterns = [
        r"(?:дом|построен|постройки|сдан|сдача|год\s+постройки)[^\.\n]{0,45}?((?:18|19|20)\d{2})\s*(?:г\.?|год[ауы]?)?",
        r"((?:18|19|20)\d{2})\s*(?:г\.?|год[ауы]?)\s*(?:постройки|построен|сдачи)",
    ]
    val = parse_number_from_text(patterns, text, min_v=1800, max_v=2035)
    return int(val) if val is not None else None


def extract_min_metro_minutes(text: Any) -> Optional[float]:
    """
    Достаёт минимальное время до метро/МЦД из строк:
    'Планерная (9 мин.); Химки (7 мин.)'
    '13-15 минут пешком до МЦД Сколково'
    """
    s = normalize_text(text)
    if not s:
        return None

    vals = []

    for m in re.finditer(r"(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*мин", s):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        val = (a + b) / 2
        if 1 <= val <= 120:
            vals.append(val)

    return min(vals) if vals else None


def extract_mkad_distance_km(text: Any) -> Optional[float]:
    s = normalize_text(text)
    if not s:
        return None

    vals = []
    for m in re.finditer(r"(\d{1,3}(?:[\.,]\d{1,2})?)\s*км\s+от\s+мкад", s):
        val = to_float(m.group(1))
        if val is not None and 0 <= val <= 150:
            vals.append(val)

    return min(vals) if vals else None


def std_rooms_segment(x: Any) -> Optional[int]:
    s = normalize_text(x)
    if not s:
        return None
    if "студ" in s:
        return 0
    m = re.search(r"([1-6])", s)
    if m:
        return int(m.group(1))
    return None


def std_repair(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"дизайнер", s):
        return "designer"
    if re.search(r"евро|евроремонт", s):
        return "euro"
    if re.search(r"космет", s):
        return "cosmetic"
    if re.search(r"без\s+ремонта|требует\s+ремонта|под\s+ремонт|нужен\s+ремонт|чернов", s):
        return "no_repair"
    if re.search(r"современ\w+\s+ремонт|качествен\w+\s+ремонт|хорош\w+\s+ремонт|ремонт\s+из\s+качествен", s):
        return "modern"
    return None


def std_finish(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"чистовая\s+с\s+мебел", s):
        return "finished_furnished"
    if re.search(r"под\s+ключ", s):
        return "turnkey"
    if re.search(r"предчист", s):
        return "prefinish"
    if re.search(r"чернов", s):
        return "rough"
    if re.search(r"без\s+отделк", s):
        return "no_finish"
    if re.search(r"чистов|с\s+отделк|современн\w+\s+отделк|готов\w+\s+отделк", s):
        return "finished"
    return None


def std_house_type(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"монолитн\w*[- ]кирпич", s):
        return "monolith_brick"
    if re.search(r"монолит", s):
        return "monolith"
    if re.search(r"кирпич", s):
        return "brick"
    if re.search(r"панель", s):
        return "panel"
    if re.search(r"блоч", s):
        return "block"
    if re.search(r"сталин", s):
        return "stalin"
    if re.search(r"дерев", s):
        return "wood"
    return None


def std_bathroom_type(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    has_combined = bool(re.search(r"совмещ", s))
    has_separate = bool(re.search(r"раздельн", s))
    if has_combined and has_separate:
        return "mixed"
    if has_combined:
        return "combined"
    if has_separate:
        return "separate"
    return None


def bathroom_counts(x: Any) -> Tuple[int, int]:
    s = normalize_text(x)
    if not s:
        return 0, 0

    combined = 0
    separate = 0

    for n, word in re.findall(r"(\d+)\s*(совмещ\w+|раздельн\w+)", s):
        if word.startswith("совмещ"):
            combined += int(n)
        elif word.startswith("раздельн"):
            separate += int(n)

    if combined == 0 and re.search(r"совмещ", s):
        combined = 1
    if separate == 0 and re.search(r"раздельн", s):
        separate = 1

    return combined, separate


def std_sale_type(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"свободн", s):
        return "free_sale"
    if re.search(r"альтернатив", s):
        return "alternative"
    if re.search(r"долев|дду|214\s*-?\s*фз|договор\s+долев", s):
        return "ddu"
    if re.search(r"купли-продажи|дкп", s):
        return "dkp"
    if re.search(r"переуступ", s):
        return "assignment"
    return None


def std_housing_market(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"новострой|новый\s+дом|застройщик|дду|214\s*-?\s*фз", s):
        return "new_building"
    if re.search(r"вторич", s):
        return "resale"
    return None


def std_housing_class(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"элит|премиум", s):
        return "premium"
    if re.search(r"бизнес", s):
        return "business"
    if re.search(r"комфорт", s):
        return "comfort"
    if re.search(r"эконом", s):
        return "economy"
    return None


def std_heating(x: Any) -> Optional[str]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"централь", s):
        return "central"
    if re.search(r"индивидуаль|автоном", s):
        return "individual"
    if re.search(r"электр", s):
        return "electric"
    return None


def yes_no_to_binary(x: Any) -> Optional[int]:
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"\bда\b|есть|имеется|предусмотр", s):
        return 1
    if re.search(r"\bнет\b|отсутств", s):
        return 0
    return None


def std_property_format(x: Any) -> Optional[str]:
    """Формат объекта как категория: квартира / апартаменты / пентхаус / доля / комната."""
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"пентхаус", s):
        return "penthouse"
    if re.search(r"апартамент", s):
        return "apartments"
    if re.search(r"дол[яи]|размер\s+доли", s):
        return "share"
    if re.search(r"комнат[аыу]\s+в\s+квартире|продается\s+комната|продаётся\s+комната", s):
        return "room"
    if re.search(r"квартир", s):
        return "flat"
    return None


def std_overlap_type(x: Any) -> Optional[str]:
    """Тип перекрытий оставляем категорией."""
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"железобетон|ж/б|жб|монолит", s):
        return "reinforced_concrete"
    if re.search(r"дерев", s):
        return "wood"
    if re.search(r"смешан", s):
        return "mixed"
    if re.search(r"металл", s):
        return "metal"
    return None


def std_gas_supply_type(x: Any) -> Optional[str]:
    """Газоснабжение: категория, а не число."""
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"централь|магистраль", s):
        return "central"
    if re.search(r"автоном|индивидуаль|газгольдер", s):
        return "individual"
    if re.search(r"нет|отсутств", s):
        return "none"
    if re.search(r"газ", s):
        return "gas_available"
    return None


def std_redevelopment_status(x: Any) -> Optional[str]:
    """
    Перепланировку оставляем категорией.
    Важно: при применении к описанию нельзя считать любое слово "есть" перепланировкой.
    """
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"не\s+узакон|не\s+согласован", s):
        return "not_legalized"
    if re.search(r"узакон|согласован", s):
        return "legalized"
    if re.search(r"нет|отсутств|не\s+было|не\s+проводил", s):
        return "none"
    if re.search(r"\bбыл[ао]?\b|есть\s+переплан|имеется\s+переплан|перепланировк\w+\s+(?:есть|была|имеется)", s):
        return "exists"
    if re.search(r"перепланировк\w+", s):
        return "exists"
    return None


def std_heating_extended(x: Any) -> Optional[str]:
    """Более подробная стандартизация отопления."""
    s = normalize_text(x)
    if not s:
        return None
    if re.search(r"централь", s):
        return "central"
    if re.search(r"индивидуаль|автоном|собственн\w+\s+котельн|итп", s):
        return "individual"
    if re.search(r"электр", s):
        return "electric"
    if re.search(r"газ", s):
        return "gas"
    return None


def parse_share_fraction(x: Any) -> Optional[float]:
    """Размер доли: '1/2', '21/100', '50%' -> 0.5, 0.21, 0.5."""
    s = normalize_text(x)
    if not s:
        return None
    m = re.search(r"(\d{1,3})\s*/\s*(\d{1,3})", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den > 0 and 0 < num <= den:
            return num / den
    m = re.search(r"(\d{1,3}(?:[\.,]\d{1,2})?)\s*%", s)
    if m:
        val = to_float(m.group(1))
        if val is not None and 0 < val <= 100:
            return val / 100
    val = to_float(s)
    if val is not None:
        if 0 < val <= 1:
            return val
        if 1 < val <= 100:
            return val / 100
    return None


def floor_group(floor: Any, floors_total: Any) -> str:
    """Категория этажа: полезна, потому что первый/последний этажи ведут себя нелинейно."""
    try:
        if pd.isna(floor):
            return "unknown"
        f = float(floor)
        t = float(floors_total) if not pd.isna(floors_total) else np.nan
    except Exception:
        return "unknown"

    if f == 1:
        return "first"
    if not pd.isna(t) and t >= 1 and f == t:
        return "last"
    if pd.isna(t) or t <= 1:
        return "middle"
    ratio = f / t
    if ratio <= 0.33:
        return "low"
    if ratio >= 0.67:
        return "high"
    return "middle"


def add_category_ranks(out: pd.DataFrame) -> pd.DataFrame:
    """
    Ранги добавляем только как дополнительные числовые признаки.
    Сами category-поля НЕ удаляем: CatBoost лучше работает со строковыми категориями.
    """
    housing_rank = {
        "economy": 1,
        "comfort": 2,
        "business": 3,
        "premium": 4,
    }
    finish_rank = {
        "no_finish": 0,
        "rough": 1,
        "prefinish": 2,
        "finished": 3,
        "turnkey": 4,
        "finished_furnished": 5,
    }
    out["housing_class_rank"] = out.get("housing_class", pd.Series(index=out.index)).map(housing_rank)
    out["finish_rank"] = out.get("finish_type", pd.Series(index=out.index)).map(finish_rank)
    return out


def binary_from_yes_no_or_text(x: Any, desc: Any = "") -> int:
    """Для полей типа Аварийность/Пандус/Маткапитал."""
    s = normalize_text(f"{x} {desc}")
    if not s:
        return 0
    if re.search(r"\bнет\b|отсутств|не\s+имеется|не\s+предусмотр|не\s+использ", s):
        return 0
    if re.search(r"\bда\b|есть|имеется|предусмотр|использ", s):
        return 1
    return 0


def has_pattern(x: Any, pattern: str) -> int:
    s = normalize_text(x)
    return int(bool(re.search(pattern, s))) if s else 0


def make_balcony_flags(raw: Any, desc: Any) -> dict[str, Any]:
    """
    Балкон/лоджия — это и counts, и категория состава.
    Важно: балкон и лоджия могут быть одновременно.
    """
    raw_s = normalize_text(raw)
    desc_s = normalize_text(desc)
    s = f"{raw_s} {desc_s}"

    negative = bool(re.search(r"без\s+(?:балкон|лоджи)|нет\s+(?:балкон|лоджи)", s))

    balcony_count = 0
    loggia_count = 0

    for n in re.findall(r"(\d+)\s*балкон", s):
        balcony_count += int(n)
    for n in re.findall(r"(\d+)\s*лоджи", s):
        loggia_count += int(n)

    if balcony_count == 0 and re.search(r"\bбалкон\w*", s) and not negative:
        balcony_count = 1
    if loggia_count == 0 and re.search(r"\bлоджи\w*", s) and not negative:
        loggia_count = 1

    has_balcony = int(balcony_count > 0)
    has_loggia = int(loggia_count > 0)

    if has_balcony and has_loggia:
        t = "both"
    elif has_balcony:
        t = "balcony"
    elif has_loggia:
        t = "loggia"
    else:
        t = "none"

    return {
        "balcony_count": balcony_count,
        "loggia_count": loggia_count,
        "has_balcony": has_balcony,
        "has_loggia": has_loggia,
        "balcony_loggia_type": t,
    }


def make_window_flags(raw: Any, desc: Any) -> dict[str, int]:
    """
    Окна/вид — multi-hot. Окна могут быть одновременно во двор, на улицу, на парк и т.д.
    Поэтому НЕ делаем одну категорию windows_view.
    """
    s = normalize_text(f"{raw} {desc}")

    features = {
        "windows_yard": 0,
        "windows_street": 0,
        "windows_park_or_forest": 0,
        "windows_water": 0,
        "windows_sunny": 0,
        "windows_panoramic": 0,
        "windows_two_sides": 0,
        "windows_quiet": 0,
    }
    if not s:
        features["windows_view_count"] = 0
        return features

    if re.search(r"(окн\w*|вид|выходят|смотрят)[^.!?]{0,90}(во\s+двор|на\s+двор|в\s+тихий\s+двор|внутренн\w+\s+двор|закрыт\w+\s+двор)|во\s+двор[^.!?]{0,90}окн", s):
        features["windows_yard"] = 1

    negative_street = bool(re.search(r"не\s+на\s+улиц|не\s+на\s+дорог|окна\s+не\s+выходят\s+на\s+улиц|не\s+на\s+проезж", s))
    positive_street = bool(re.search(r"(окн\w*|вид|выходят|смотрят)[^.!?]{0,90}(на\s+улиц|на\s+дорог|на\s+проспект|на\s+бульвар|на\s+переулок|на\s+проезж)|на\s+улиц[^.!?]{0,90}окн", s))
    if positive_street and not negative_street:
        features["windows_street"] = 1

    if re.search(r"(окн\w*|вид|выходят|смотрят)[^.!?]{0,110}(на\s+парк|на\s+сквер|на\s+лес|на\s+лесопарк|на\s+зелен\w+\s+зон|на\s+рощ)|(?:парк|лес|сквер)[^.!?]{0,110}окн", s):
        features["windows_park_or_forest"] = 1

    if re.search(r"(окн\w*|вид|выходят|смотрят)[^.!?]{0,110}(на\s+рек|на\s+озер|на\s+пруд|на\s+вод|на\s+канал|на\s+набережн|на\s+водоем|на\s+водоём)", s):
        features["windows_water"] = 1

    if re.search(r"солнечн\w+\s+сторон|солнечн\w+\s+квартир|много\s+света|светлая\s+квартир|окна\s+на\s+юг|южн\w+\s+сторон|юго-запад|юго-восток", s):
        features["windows_sunny"] = 1

    if re.search(r"панорамн\w+\s+окн|панорамн\w+\s+вид|видовая\s+квартир|видовые\s+характеристик|красив\w+\s+вид|отличн\w+\s+вид\s+из\s+окон", s):
        features["windows_panoramic"] = 1

    if re.search(r"окна\s+на\s+две\s+сторон|на\s+2\s+сторон|на\s+две\s+стороны\s+света|распашонк|двусторонн\w+\s+планиров", s):
        features["windows_two_sides"] = 1

    if re.search(r"тихий\s+двор|тихая\s+сторон|не\s+шумно|нет\s+шума|окна\s+не\s+на\s+дорог|окна\s+не\s+на\s+улиц|вдали\s+от\s+дорог", s):
        features["windows_quiet"] = 1

    view_cols = [
        "windows_yard",
        "windows_street",
        "windows_park_or_forest",
        "windows_water",
        "windows_sunny",
        "windows_panoramic",
    ]
    features["windows_view_count"] = int(sum(features[c] for c in view_cols))
    return features


def make_parking_flags(raw: Any, desc: Any) -> dict[str, int]:
    """Парковка — multi-hot, потому что типы могут сочетаться."""
    s = normalize_text(f"{raw} {desc}")
    underground = int(bool(re.search(r"подземн\w+\s+паркинг|подземн\w+\s+парковк|\bподземная\b", s)))
    multilevel = int(bool(re.search(r"многоуровнев|наземн\w+\s+многоуровнев", s)))
    surface = int(bool(re.search(r"наземн\w+|открыт\w+\s+парковк|плоскостн\w+\s+парковк", s)))
    yard = int(bool(re.search(r"во\s+двор|дворов\w+\s+парковк|парковк\w+\s+во\s+двор", s)))
    return {
        "parking_underground": underground,
        "parking_surface": surface,
        "parking_multilevel": multilevel,
        "parking_yard": yard,
        "parking_open_yard": int(surface or yard),
        "parking_barrier": int(bool(re.search(r"шлагбаум", s))),
        "parking_guest": int(bool(re.search(r"гостев", s))),
    }


def make_yard_flags(raw: Any, desc: Any) -> dict[str, int]:
    """Двор/территория — multi-hot."""
    s = normalize_text(f"{raw} {desc}")
    return {
        "yard_closed": int(bool(re.search(r"закрыт\w+\s+территор|закрыт\w+\s+двор|огороженн\w+\s+территор", s))),
        "yard_playground": int(bool(re.search(r"детск\w+\s+площад", s))),
        "yard_sportground": int(bool(re.search(r"спортивн\w+\s+площад|воркаут", s))),
        "yard_no_cars": int(bool(re.search(r"двор\s+без\s+машин", s))),
        "yard_barrier": int(bool(re.search(r"шлагбаум", s))),
    }


def make_furniture_flags(raw: Any, desc: Any) -> dict[str, Any]:
    raw_s = normalize_text(raw)
    desc_s = normalize_text(desc)
    both = f"{raw_s} {desc_s}"

    items = []

    has_furniture = bool(re.search(
        r"мебел|меблирован|укомплектован[^\.]{0,80}мебел|остает(?:ся|ься)[^\.]{0,80}мебел",
        both,
    ))
    has_kitchen_furniture = bool(
        re.search(r"\bкухня\b|кухонн\w+", raw_s)
        or re.search(r"кухонн\w+\s+гарнитур|встроенн\w+\s+кухн|кухня\s+остает", desc_s)
    )
    has_wardrobe_storage = bool(re.search(r"шкаф|гардероб|мест[ао]\s+хранени", both))
    has_sleeping_places = bool(re.search(r"спальн\w+\s+мест|кровать|диван", both))

    if has_kitchen_furniture:
        items.append("kitchen")
    if has_wardrobe_storage:
        items.append("storage")
    if has_sleeping_places:
        items.append("sleeping")
    if has_furniture and not items:
        items.append("general")

    return {
        "has_furniture": int(has_furniture or bool(items)),
        "has_kitchen_furniture": int(has_kitchen_furniture),
        "has_wardrobe_storage": int(has_wardrobe_storage),
        "has_sleeping_places": int(has_sleeping_places),
        "furniture_set": "|".join(sorted(set(items))) if items else "none",
    }


def make_appliance_flags(raw: Any, desc: Any) -> dict[str, Any]:
    raw_s = normalize_text(raw)
    desc_s = normalize_text(desc)
    both = f"{raw_s} {desc_s}"

    checks = {
        "fridge": bool(re.search(r"холодильник", both)),
        "washer": bool(re.search(r"стиральн\w+\s+машин", both)),
        "dishwasher": bool(re.search(r"посудомоечн\w+\s+машин", both)),
        "ac": bool(re.search(r"кондиционер|сплит-систем", both)),
        "water_heater": bool(re.search(r"водонагревател|бойлер", both)),
        "oven": bool(re.search(r"духов\w+\s+шкаф|духовк", both)),
        "cooktop": bool(re.search(r"варочн\w+\s+панел", both)),
        "hood": bool(re.search(r"вытяжк", both)),
    }

    flags = {
        "has_fridge": int(checks["fridge"]),
        "has_washer": int(checks["washer"]),
        "has_dishwasher": int(checks["dishwasher"]),
        "has_ac": int(checks["ac"]),
        "has_water_heater": int(checks["water_heater"]),
        "has_oven": int(checks["oven"]),
        "has_cooktop": int(checks["cooktop"]),
        "has_hood": int(checks["hood"]),
    }

    explicit_appliances = bool(re.search(
        r"бытов\w+\s+техник|мебелью\s+и\s+техникой|укомплектован[^\.]{0,80}техник|техника\s+остает",
        both,
    ))
    present_items = [k for k, v in checks.items() if v]
    flags["has_appliances"] = int(explicit_appliances or bool(present_items))
    flags["appliances_set"] = "|".join(sorted(present_items)) if present_items else ("general" if explicit_appliances else "none")
    return flags

def make_deal_flags(raw: Any, desc: Any) -> dict[str, int]:
    s = normalize_text(str(raw) + " " + str(desc))
    return {
        "mortgage_possible": int(bool(re.search(r"ипотек", s))),
        "one_owner": int(bool(re.search(r"один\s+собственник", s))),
        "no_encumbrance": int(bool(re.search(r"без\s+обременен", s))),
        "quick_deal": int(bool(re.search(r"быстр\w+\s+выход\s+на\s+сделк|свободная\s+продажа", s))),
    }


def parse_elevator_counts(*values: Any) -> Tuple[int, int, int]:
    """
    Возвращает: passenger, freight, total.
    Поддерживает строки:
    '10 пассажирских'
    '1 пассажирский, 1 грузовой'
    отдельные поля Авито 'Пассажирский лифт', 'Грузовой лифт'.
    """
    s = normalize_text(" ".join([str(v) for v in values if not is_missing(v)]))
    if not s:
        return 0, 0, 0

    passenger = 0
    freight = 0

    for n in re.findall(r"(\d+)\s*пассажир", s):
        passenger += int(n)
    for n in re.findall(r"(\d+)\s*грузов", s):
        freight += int(n)

    # Если поле просто числовое, например '1', и оно пришло из отдельной колонки пассажирского лифта.

    nums = []
    for v in values:
        iv = to_int(v)
        if iv is not None and 0 <= iv <= 20:
            nums.append(iv)

    if passenger == 0 and freight == 0 and nums:
        passenger = max(nums)

    total = passenger + freight
    return passenger, freight, total


def build_ml_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if "Описание" not in df.columns:
        df["Описание"] = ""

    desc = df["Описание"].fillna("").apply(normalize_text)


    out = pd.DataFrame(index=df.index)
    out["source"] = coalesce_columns(df, ["source"]).fillna("unknown").astype(str)
    out["source_listing_id"] = coalesce_columns(df, ["source_listing_id", "record_id", "ID"])
    out["url"] = coalesce_columns(df, ["canonical_url", "Ссылка"])
    out["parsed_at"] = coalesce_columns(df, ["parsed_at"])
    out["link_collected_at"] = coalesce_columns(df, ["link_collected_at"])


    price_raw = coalesce_columns(df, ["Цена числом", "Цена"])
    price_m2_raw = coalesce_columns(df, ["Цена за м² числом", "Цена за м²"])

    out["price_rub"] = price_raw.apply(to_float)
    out["price_m2_rub"] = price_m2_raw.apply(to_float)


    out["lat"] = coalesce_columns(df, ["Широта"]).apply(to_float)
    out["lon"] = coalesce_columns(df, ["Долгота"]).apply(to_float)
    out["region"] = coalesce_columns(df, ["Регион"])
    out["okrug"] = coalesce_columns(df, ["Округ"])
    out["settlement"] = coalesce_columns(df, ["Населенный пункт"])
    out["street"] = coalesce_columns(df, ["Улица"])
    out["address"] = coalesce_columns(df, ["Полный адрес"])

    metro_raw = coalesce_columns(df, ["Метро"])
    highway_raw = coalesce_columns(df, ["Шоссе"])
    out["metro_min_listed"] = metro_raw.combine_first(desc).apply(extract_min_metro_minutes)
    out["metro_count_listed"] = metro_raw.fillna("").apply(lambda x: 0 if is_missing(x) else len([p for p in str(x).split(";") if p.strip()]))
    out["mkad_distance_km"] = highway_raw.combine_first(desc).apply(extract_mkad_distance_km)


    rooms_existing = coalesce_columns(df, ["Количество комнат", "Об апартаментах: Количество комнат"])
    rooms_segment = coalesce_columns(df, ["Сегмент комнатности", "Комнат", "Название", "Тип жилья"]).apply(std_rooms_segment)
    rooms_desc = desc.apply(extract_rooms)

    out["rooms"] = rooms_existing.apply(to_float).combine_first(rooms_segment).combine_first(rooms_desc)
    out["is_studio"] = (out["rooms"] == 0).astype(int)

    total_area_existing = coalesce_columns(df, ["Общая площадь", "Об апартаментах: Общая площадь", "Площадь м² из названия"])
    kitchen_area_existing = coalesce_columns(df, ["Площадь кухни", "Об апартаментах: Площадь кухни"])
    living_area_existing = coalesce_columns(df, ["Жилая площадь", "Об апартаментах: Жилая площадь"])

    out["total_area_m2"] = total_area_existing.apply(to_float).combine_first(desc.apply(lambda s: extract_area(s, "total")))
    out["kitchen_area_m2"] = kitchen_area_existing.apply(to_float).combine_first(desc.apply(lambda s: extract_area(s, "kitchen")))
    out["living_area_m2"] = living_area_existing.apply(to_float).combine_first(desc.apply(lambda s: extract_area(s, "living")))


    out["price_m2_rub"] = out["price_m2_rub"].combine_first(
        pd.Series([safe_div(a, b) for a, b in zip(out["price_rub"], out["total_area_m2"])], index=out.index)
    )

    out["kitchen_area_share"] = pd.Series(
        [safe_div(a, b) for a, b in zip(out["kitchen_area_m2"], out["total_area_m2"])],
        index=out.index,
    )
    out["living_area_share"] = pd.Series(
        [safe_div(a, b) for a, b in zip(out["living_area_m2"], out["total_area_m2"])],
        index=out.index,
    )


    floor_raw = coalesce_columns(df, ["Этаж", "Об апартаментах: Этаж", "Этаж из названия"])
    parsed_floor = floor_raw.apply(parse_floor_raw)
    desc_floor = desc.apply(extract_floor_pair)

    out["floor"] = pd.Series([p[0] for p in parsed_floor], index=out.index).combine_first(
        pd.Series([p[0] for p in desc_floor], index=out.index)
    )
    out["floors_total"] = pd.Series([p[1] for p in parsed_floor], index=out.index).combine_first(
        pd.Series([p[1] for p in desc_floor], index=out.index)
    )


    floors_total_raw = coalesce_columns(df, ["О доме: Этажей в доме", "О жилом комплексе: Этажей в доме"])
    out["floors_total"] = out["floors_total"].combine_first(floors_total_raw.apply(to_float))

    out["floor_ratio"] = pd.Series(
        [safe_div(a, b) for a, b in zip(out["floor"], out["floors_total"])],
        index=out.index,
    )
    out["is_first_floor"] = (out["floor"] == 1).astype(int)
    out["is_last_floor"] = ((out["floor"].notna()) & (out["floors_total"].notna()) & (out["floor"] == out["floors_total"])).astype(int)
    out["floor_group"] = [floor_group(f, t) for f, t in zip(out["floor"], out["floors_total"])]


    house_year_raw = coalesce_columns(df, ["Год постройки", "О доме: Год постройки", "Год сдачи"])
    out["house_year"] = house_year_raw.apply(to_float).combine_first(desc.apply(extract_house_year))
    out["house_age"] = out["house_year"].apply(lambda y: BASE_YEAR - y if pd.notna(y) and 1800 <= y <= 2035 else np.nan)
    out["is_new_building_year"] = ((out["house_year"].notna()) & (out["house_year"] >= BASE_YEAR)).astype(int)

    ceiling_raw = coalesce_columns(df, ["Высота потолков", "Об апартаментах: Высота потолков"])
    out["ceiling_height_m"] = ceiling_raw.apply(to_float).combine_first(desc.apply(extract_ceiling_height))

    house_type_raw = coalesce_columns(df, ["Тип дома", "О доме: Тип дома", "О жилом комплексе: Тип дома"])
    out["house_type"] = house_type_raw.apply(std_house_type).combine_first(desc.apply(std_house_type)).fillna("unknown")

    property_format_raw = coalesce_columns(df, ["Тип жилья", "Название", "Сегмент комнатности", "Размер доли"])
    out["property_format"] = property_format_raw.apply(std_property_format).combine_first(desc.apply(std_property_format)).fillna("flat")
    out["is_apartment_format"] = (out["property_format"] == "apartments").astype(int)
    out["is_penthouse_format"] = (out["property_format"] == "penthouse").astype(int)

    housing_market_raw = coalesce_columns(df, ["Тип жилья", "Условия сделки", "Способ продажи", "О жилом комплексе: Тип участия", "Название"])
    out["housing_market"] = housing_market_raw.apply(std_housing_market).combine_first(desc.apply(std_housing_market)).fillna("unknown")
    out["housing_class"] = desc.apply(std_housing_class).fillna("unknown")

    heating_raw = coalesce_columns(df, ["Отопление", "О помещении: Отопление"])
    out["heating_type"] = heating_raw.apply(std_heating_extended).combine_first(desc.apply(std_heating_extended)).fillna("unknown")

    overlap_raw = coalesce_columns(df, ["Тип перекрытий"])
    gas_raw = coalesce_columns(df, ["Газоснабжение"])
    redevelopment_raw = coalesce_columns(df, ["Перепланировка"])
    out["overlap_type"] = overlap_raw.apply(std_overlap_type).fillna("unknown")
    out["gas_supply_type"] = gas_raw.apply(std_gas_supply_type).fillna("unknown")
    out["redevelopment_status"] = redevelopment_raw.apply(std_redevelopment_status).combine_first(desc.apply(std_redevelopment_status)).fillna("unknown")

    share_raw = coalesce_columns(df, ["Размер доли"])
    out["ownership_share_fraction"] = share_raw.apply(parse_share_fraction)
    out["is_share_sale"] = ((out["property_format"] == "share") | out["ownership_share_fraction"].notna()).astype(int)


    repair_raw = coalesce_columns(df, ["Ремонт", "Об апартаментах: Ремонт"])
    finish_raw = coalesce_columns(df, ["Отделка", "Об апартаментах: Отделка"])
    sale_raw = coalesce_columns(df, [
        "Способ продажи",
        "Условия сделки",
        "Условия продажи",
        "Об апартаментах: Способ продажи",
        "Об апартаментах: Условия продажи",
        "О жилом комплексе: Тип участия",
    ])

    out["repair_type"] = repair_raw.apply(std_repair).combine_first(desc.apply(std_repair)).fillna("unknown")
    out["finish_type"] = finish_raw.apply(std_finish).combine_first(desc.apply(std_finish)).fillna("unknown")
    out["sale_type"] = sale_raw.apply(std_sale_type).combine_first(desc.apply(std_sale_type)).fillna("unknown")
    out = add_category_ranks(out)


    bathroom_raw = coalesce_columns(df, ["Санузел", "Об апартаментах: Санузел"])
    bathroom_desc_type = desc.apply(std_bathroom_type)
    out["bathroom_type"] = bathroom_raw.apply(std_bathroom_type).combine_first(bathroom_desc_type).fillna("unknown")

    bath_counts_raw = bathroom_raw.apply(bathroom_counts)
    bath_counts_desc = desc.apply(bathroom_counts)
    out["bathroom_combined_count"] = [a[0] for a in bath_counts_raw]
    out["bathroom_separate_count"] = [a[1] for a in bath_counts_raw]
    out["bathroom_combined_count"] = np.where(
        out["bathroom_combined_count"] == 0,
        [a[0] for a in bath_counts_desc],
        out["bathroom_combined_count"],
    )
    out["bathroom_separate_count"] = np.where(
        out["bathroom_separate_count"] == 0,
        [a[1] for a in bath_counts_desc],
        out["bathroom_separate_count"],
    )
    out["bathroom_total_count"] = out["bathroom_combined_count"] + out["bathroom_separate_count"]


    elevator_common = coalesce_columns(df, ["Количество лифтов"])
    passenger_lift_raw = coalesce_columns(df, ["О доме: Пассажирский лифт", "О жилом комплексе: Пассажирский лифт"])
    freight_lift_raw = coalesce_columns(df, ["О доме: Грузовой лифт", "О жилом комплексе: Грузовой лифт"])

    elevator_rows = []
    for i in df.index:
        p, f, total = parse_elevator_counts(elevator_common.at[i], passenger_lift_raw.at[i], freight_lift_raw.at[i])

        # Если отдельные поля Авито числовые, добавляем их точнее.
        p_direct = to_int(passenger_lift_raw.at[i])
        f_direct = to_int(freight_lift_raw.at[i])
        if p_direct is not None:
            p = max(p, p_direct)
        if f_direct is not None:
            f = max(f, f_direct)
        total = p + f

        elevator_rows.append((p, f, total))

    out["passenger_elevator_count"] = [x[0] for x in elevator_rows]
    out["freight_elevator_count"] = [x[1] for x in elevator_rows]
    out["elevator_total_count"] = [x[2] for x in elevator_rows]
    out["has_elevator"] = (out["elevator_total_count"] > 0).astype(int)


    balcony_raw = coalesce_columns(df, ["Балкон или лоджия", "Балкон/лоджия", "Об апартаментах: Балкон или лоджия"])
    windows_raw = coalesce_columns(df, ["Окна", "Вид из окон", "Об апартаментах: Окна"])
    parking_raw = coalesce_columns(df, ["Парковка", "О доме: Парковка", "О жилом комплексе: Парковка", "О здании: Парковка"])
    yard_raw = coalesce_columns(df, ["О доме: Двор", "О жилом комплексе: Двор", "Придомовая территория"])
    furniture_raw = coalesce_columns(df, ["Мебель", "Продаётся с мебелью", "Об апартаментах: Мебель"])
    tech_raw = coalesce_columns(df, ["Техника", "Об апартаментах: Техника"])
    warm_floor_raw = coalesce_columns(df, ["Тёплый пол", "Об апартаментах: Тёплый пол"])
    entrance_raw = coalesce_columns(df, ["О подъезде", "О доме: В доме"])
    emergency_raw = coalesce_columns(df, ["Аварийность", "О доме: Запланирован снос"])
    ramp_raw = coalesce_columns(df, ["Пандус"])
    minor_owners_raw = coalesce_columns(df, ["Несовершеннолетние собственники"])
    maternity_capital_raw = coalesce_columns(df, ["Материнский капитал при покупке"])
    housing_type_for_flags = coalesce_columns(df, ["Тип жилья"])
    title_for_flags = coalesce_columns(df, ["Название"])

    flag_blocks = []
    for i in df.index:
        d = desc.at[i]
        block = {}
        block.update(make_balcony_flags(balcony_raw.at[i], d))
        block.update(make_window_flags(windows_raw.at[i], d))
        block.update(make_parking_flags(parking_raw.at[i], d))
        block.update(make_yard_flags(yard_raw.at[i], d))
        block.update(make_furniture_flags(furniture_raw.at[i], d))
        block.update(make_appliance_flags(tech_raw.at[i], d))
        block.update(make_deal_flags(sale_raw.at[i], d))

        block["has_warm_floor"] = int(
            has_pattern(warm_floor_raw.at[i], r"есть|тепл\w+\s+пол")
            or has_pattern(d, r"тепл\w+\s+пол")
        )

        entrance_s = normalize_text(entrance_raw.at[i])
        block["is_emergency_house"] = binary_from_yes_no_or_text(emergency_raw.at[i])
        block["has_ramp"] = binary_from_yes_no_or_text(ramp_raw.at[i])
        block["has_minor_owners"] = binary_from_yes_no_or_text(minor_owners_raw.at[i])
        block["maternity_capital_used"] = binary_from_yes_no_or_text(maternity_capital_raw.at[i])
        block["has_garbage_chute"] = int(bool(re.search(r"мусоропровод", entrance_s + " " + d)))
        block["has_concierge"] = int(bool(re.search(r"консьерж|консьержка|дежурн\w+\s+по\s+подъезд", entrance_s + " " + d)))
        block["entrance_clean"] = int(bool(re.search(r"чист\w+\s+подъезд|ухоженн\w+\s+подъезд", entrance_s + " " + d)))
        block["entrance_needs_repair"] = int(bool(re.search(r"подъезд[^.!?]{0,80}(требует\s+ремонт|без\s+ремонт|стар\w+)|ремонт\s+подъезд", entrance_s + " " + d)))
        block["has_security"] = int(bool(re.search(r"охрана|консьерж|видеонаблюдени|закрыт\w+\s+территор", d)))
        block["has_storage_room"] = int(bool(re.search(r"кладов|колясочн", d)))
        block["has_mall_nearby_text"] = int(bool(re.search(r"трц|тц|торгов\w+\s+центр", d)))
        block["has_park_nearby_text"] = int(bool(re.search(r"парк|лес|сквер|зелен\w+\s+зон", d)))
        block["has_school_nearby_text"] = int(bool(re.search(r"школ|лицей|гимназ", d)))
        block["has_kindergarten_nearby_text"] = int(bool(re.search(r"детск\w+\s+сад|садик", d)))
        flag_blocks.append(block)

    flags = pd.DataFrame(flag_blocks, index=df.index)
    out = pd.concat([out, flags], axis=1)


    out["description_len_chars"] = df["Описание"].fillna("").astype(str).str.len()
    out["description_len_words"] = desc.apply(lambda s: len(s.split()) if s else 0)


    out["repair_source"] = np.select(
        [repair_raw.apply(std_repair).notna(), repair_raw.apply(std_repair).isna() & desc.apply(std_repair).notna()],
        ["existing", "description"],
        default="missing",
    )
    out["finish_source"] = np.select(
        [finish_raw.apply(std_finish).notna(), finish_raw.apply(std_finish).isna() & desc.apply(std_finish).notna()],
        ["existing", "description"],
        default="missing",
    )
    out["house_type_source"] = np.select(
        [house_type_raw.apply(std_house_type).notna(), house_type_raw.apply(std_house_type).isna() & desc.apply(std_house_type).notna()],
        ["existing", "description"],
        default="missing",
    )
    out["ceiling_height_source"] = np.select(
        [ceiling_raw.apply(to_float).notna(), ceiling_raw.apply(to_float).isna() & desc.apply(extract_ceiling_height).notna()],
        ["existing", "description"],
        default="missing",
    )

    # ---------- финальная чистка типов ----------
    binary_prefixes = (
        "has_",
        "is_",
        "parking_",
        "yard_",
        "windows_",
        "mortgage_",
        "one_owner",
        "no_encumbrance",
        "quick_deal",
    )
    for col in out.columns:
        if col.startswith(binary_prefixes):
            out[col] = out[col].fillna(0).astype(int)


    out.loc[~out["lat"].between(40, 70), "lat"] = np.nan
    out.loc[~out["lon"].between(20, 80), "lon"] = np.nan
    out.loc[~out["total_area_m2"].between(10, 500), "total_area_m2"] = np.nan
    out.loc[~out["kitchen_area_m2"].between(2, 150), "kitchen_area_m2"] = np.nan
    out.loc[~out["living_area_m2"].between(2, 400), "living_area_m2"] = np.nan
    out.loc[~out["ceiling_height_m"].between(2.0, 6.0), "ceiling_height_m"] = np.nan
    out.loc[~out["floor"].between(1, 100), "floor"] = np.nan
    out.loc[~out["floors_total"].between(1, 100), "floors_total"] = np.nan
    out.loc[(out["floor"].notna()) & (out["floors_total"].notna()) & (out["floor"] > out["floors_total"]), ["floor", "floors_total", "floor_ratio"]] = np.nan
    out.loc[~out["rooms"].between(0, 10), "rooms"] = np.nan
    out.loc[~out["house_year"].between(1800, 2035), "house_year"] = np.nan
    out.loc[~out["ownership_share_fraction"].between(0, 1), "ownership_share_fraction"] = np.nan
    out.loc[~out["balcony_count"].between(0, 10), "balcony_count"] = np.nan
    out.loc[~out["loggia_count"].between(0, 10), "loggia_count"] = np.nan


    out["floor_group"] = [floor_group(f, t) for f, t in zip(out["floor"], out["floors_total"])]


    preferred_order = [
        "source",
        "source_listing_id",
        "url",
        "parsed_at",
        "link_collected_at",
        "price_rub",
        "price_m2_rub",
        "lat",
        "lon",
        "region",
        "okrug",
        "settlement",
        "street",
        "address",
        "rooms",
        "is_studio",
        "total_area_m2",
        "kitchen_area_m2",
        "living_area_m2",
        "kitchen_area_share",
        "living_area_share",
        "floor",
        "floors_total",
        "floor_ratio",
        "floor_group",
        "is_first_floor",
        "is_last_floor",
        "house_year",
        "house_age",
        "is_new_building_year",
        "ceiling_height_m",
        "house_type",
        "property_format",
        "housing_market",
        "housing_class",
        "housing_class_rank",
        "heating_type",
        "overlap_type",
        "gas_supply_type",
        "redevelopment_status",
        "ownership_share_fraction",
        "is_share_sale",
        "repair_type",
        "finish_type",
        "finish_rank",
        "sale_type",
        "bathroom_type",
        "bathroom_combined_count",
        "bathroom_separate_count",
        "bathroom_total_count",
        "passenger_elevator_count",
        "freight_elevator_count",
        "elevator_total_count",
        "has_elevator",
        "metro_min_listed",
        "metro_count_listed",
        "mkad_distance_km",
        "balcony_count",
        "loggia_count",
        "balcony_loggia_type",
    ]

    remaining = [c for c in out.columns if c not in preferred_order]
    out = out[[c for c in preferred_order if c in out.columns] + remaining]

    return out


def is_osm_feature_column(col: str) -> bool:
    """
    Берём только признаки, созданные fast_osm_enrichment.py,
    а не все сырые колонки из OSM-файла.
    """
    return (
        col.startswith("osm_")
        or col.startswith("log1p_osm_")
        or col.startswith("infrastructure_score_")
        or bool(re.match(r"^has_(metro|park|school|kindergarten|hospital_clinic|pharmacy|supermarket|public_transport_stop|cafe_restaurant|fitness_sport)_", col))
    )


def normalize_osm_feature_column_name(col: str, radius_m: int) -> str:
    """
    В твоём OSM-файле радиус фактически 1000 м, но часть колонок могла остаться с суффиксом 500m.
    Приводим названия к честному радиусу:
      osm_school_count_500m       -> osm_school_count_1000m
      has_school_500m             -> has_school_1000m
      infrastructure_score_500m   -> infrastructure_score_1000m
      osm_school_nearest_m        -> osm_school_nearest_1000m
    """
    suffix = f"{radius_m}m"
    new_col = col


    new_col = re.sub(r"\d+m", suffix, new_col)


    m = re.match(r"^(osm_.+)_nearest_m$", new_col)
    if m:
        new_col = f"{m.group(1)}_nearest_{suffix}"

    return new_col


def read_osm_features(path: str, radius_m: int) -> pd.DataFrame:
    """
    Читает OSM-файл и возвращает только ключи + ML-признаки OSM.
    Мержим по source + source_listing_id, потому что эта пара уникальна
    и совпадает между regex/std датасетом и OSM-датасетом.
    """
    geo = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    key_cols = ["source", "source_listing_id"]
    missing_keys = [c for c in key_cols if c not in geo.columns]
    if missing_keys:
        raise ValueError(f"В OSM-файле нет ключевых колонок: {missing_keys}")

    feature_cols = [c for c in geo.columns if is_osm_feature_column(c)]
    if not feature_cols:
        raise ValueError("В OSM-файле не найдены OSM-признаки: osm_*, has_metro_*, log1p_osm_* и т.д.")

    osm = geo[key_cols + feature_cols].copy()
    osm = osm.drop_duplicates(key_cols, keep="first")

    rename_map = {
        c: normalize_osm_feature_column_name(c, radius_m)
        for c in feature_cols
    }
    osm = osm.rename(columns=rename_map)


    osm = osm.loc[:, ~osm.columns.duplicated()].copy()

    return osm


def merge_osm_features(ml: pd.DataFrame, osm_path: str, radius_m: int) -> pd.DataFrame:
    """
    Добавляет OSM-признаки к ML-датасету.
    Строки без координат или без OSM-расчёта остаются в датасете,
    но получают osm_features_available = 0.
    """
    if not MERGE_GEO_FEATURES:
        ml = ml.copy()
        ml["osm_features_available"] = 0
        return ml

    if not os.path.exists(osm_path):
        print(f"OSM-файл не найден: {osm_path}")
        print("Скрипт сохранит только regex/std ML-признаки без OSM.")
        ml = ml.copy()
        ml["osm_features_available"] = 0
        return ml

    key_cols = ["source", "source_listing_id"]
    missing_keys = [c for c in key_cols if c not in ml.columns]
    if missing_keys:
        raise ValueError(f"В ML-датасете нет ключевых колонок для merge: {missing_keys}")

    osm = read_osm_features(osm_path, radius_m)

    before_rows = len(ml)
    merged = ml.merge(osm, on=key_cols, how="left", validate="one_to_one")

    osm_feature_cols = [c for c in osm.columns if c not in key_cols]
    merged["osm_features_available"] = merged[osm_feature_cols].notna().any(axis=1).astype(int)
    merged["osm_radius_m"] = np.where(merged["osm_features_available"].eq(1), radius_m, np.nan)

    if len(merged) != before_rows:
        raise RuntimeError("После merge изменилось количество строк. Это нельзя допускать для ML-датасета.")

    matched = int(merged["osm_features_available"].sum())
    missing = int(len(merged) - matched)

    print("\nOSM-признаки подтянуты:")
    print(f"  OSM-файл: {osm_path}")
    print(f"  Радиус: {radius_m} м")
    print(f"  Найдено совпадений: {matched}")
    print(f"  Без OSM-признаков: {missing}")
    print(f"  Добавлено OSM-колонок: {len(osm_feature_cols)}")

    return merged


def main() -> None:
    avito = read_csv_safe(AVITO_PATH, "avito")
    cian = read_csv_safe(CIAN_PATH, "cian")
    raw = pd.concat([avito, cian], ignore_index=True, sort=False)

    ml = build_ml_dataset(raw)
    ml = merge_osm_features(ml, GEO_OSM_PATH, GEO_RADIUS_M)
    ml.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"Готово: {OUTPUT_PATH}")
    print(f"Строк: {len(ml)}")
    print(f"Колонок: {ml.shape[1]}")
    print("\nЗаполненность ключевых признаков:")
    key_cols = [
        "price_rub",
        "price_m2_rub",
        "lat",
        "lon",
        "rooms",
        "total_area_m2",
        "kitchen_area_m2",
        "living_area_m2",
        "floor",
        "floors_total",
        "house_year",
        "ceiling_height_m",
        "repair_type",
        "finish_type",
        "house_type",
        "bathroom_type",
        "property_format",
        "overlap_type",
        "gas_supply_type",
        "balcony_loggia_type",
        "floor_group",
        "osm_features_available",
        f"osm_metro_count_{GEO_RADIUS_M}m",
        f"osm_park_count_{GEO_RADIUS_M}m",
        f"osm_school_count_{GEO_RADIUS_M}m",
        f"osm_supermarket_count_{GEO_RADIUS_M}m",
    ]
    key_cols = [c for c in key_cols if c in ml.columns]
    print((ml[key_cols].notna().mean() * 100).round(1).sort_values(ascending=False))


if __name__ == "__main__":
    main()
