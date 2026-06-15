import argparse
import csv
import random
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


START_URL = (
    "https://www.cian.ru/cat.php?"
    "deal_type=sale&engine_version=2&location%5B0%5D=4593"
    "&offer_type=flat&region=4593&source=search_string"
)


def clean_text(value):
    if value is None:
        return None

    value = str(value).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else None


def clean_description_text(value):
    # Описание сохраняем в одну строку для CSV.
    
    if value is None:
        return None

    value = str(value).replace("\xa0", " ")
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else None


def text_to_int(value):
    """
    '338 577 750 ₽' -> 338577750
    '1 493 962 ₽/м²' -> 1493962
    """
    value = clean_text(value)

    if not value:
        return None

    digits = re.sub(r"\D", "", value)

    return int(digits) if digits else None


def parse_area_from_text(value):
    """
    '60,3 м²' -> 60.3
    """
    value = clean_text(value)

    if not value:
        return None

    value = value.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", value)

    return float(match.group()) if match else None


def format_rub_per_m2(value):
    if value is None:
        return None

    return f"{int(round(value)):,}".replace(",", " ") + " ₽/м²"


def extract_cian_id(url):
    url = clean_text(url)

    if not url:
        return None

    match = re.search(r"/flat/(\d+)", url)

    return match.group(1) if match else None


def is_cian_flat_detail_url(url):
    
    url = clean_text(url) or ""
    return bool(re.search(r"/sale/flat/\d+/?", url)) or bool(re.search(r"/flat/\d+/?", url))


def set_url_page_param(url, page_number):
    
    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True)

    new_items = []

    for key, value in query_items:
        if key != "p":
            new_items.append((key, value))

    if page_number > 1:
        new_items.append(("p", str(page_number)))

    new_query = urlencode(new_items, doseq=True)

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            new_query,
            parts.fragment,
        )
    )


CIAN_ROOM_PARAM_BY_SEGMENT = {
    "1 комнатные": "room1",
    "2 комнатные": "room2",
    "3 комнатные": "room3",
    "4 комнатные": "room4",
    "студия": "room9",
}


def get_room_param_for_segment(segment):
    segment = normalize_quicklink_title(segment)

    if not segment:
        return None

    if segment.startswith("1 комнат"):
        return "room1"

    if segment.startswith("2 комнат"):
        return "room2"

    if segment.startswith("3 комнат"):
        return "room3"

    if segment.startswith("4 комнат"):
        return "room4"

    if "студи" in segment:
        return "room9"

    return CIAN_ROOM_PARAM_BY_SEGMENT.get(segment)


def build_cian_room_listing_url(room_segment, page_number, start_url=None):
    
    segment_name = clean_text(room_segment.get("segment")) if isinstance(room_segment, dict) else clean_text(room_segment)
    room_param = get_room_param_for_segment(segment_name)

    if not room_param:
        raise RuntimeError(f"Не удалось определить room-параметр для сегмента: {segment_name}")


    base_url = start_url or START_URL
    segment_url = clean_text(room_segment.get("url")) if isinstance(room_segment, dict) else None

    merged = []

    def add_query_from(url):
        if not url:
            return

        try:
            for key, value in parse_qsl(urlsplit(url).query, keep_blank_values=True):
                merged.append((key, value))
        except Exception:
            pass

    add_query_from(base_url)
    add_query_from(segment_url)

    skip_keys = {"p", "room1", "room2", "room3", "room4", "room9"}
    query_items = []
    seen_keys = set()

    for key, value in merged:
        if key in skip_keys:
            continue


        if key in seen_keys:
            continue

        seen_keys.add(key)
        query_items.append((key, value))


    required = {
        "deal_type": "sale",
        "engine_version": "2",
        "offer_type": "flat",
        "region": "4593",
    }

    existing = {key for key, _ in query_items}

    for key, value in required.items():
        if key not in existing:
            query_items.append((key, value))

    if page_number > 1:
        query_items.append(("p", str(page_number)))

    query_items.append((room_param, "1"))

    query = urlencode(query_items, doseq=True)

    return urlunsplit(
        (
            "https",
            "www.cian.ru",
            "/cat.php",
            query,
            "",
        )
    )


def listing_page_is_observed(page, expected_page_number=None, expected_room_param=None):
    
    try:
        current_url = page.url
        query = dict(parse_qsl(urlsplit(current_url).query, keep_blank_values=True))

        if expected_room_param and query.get(expected_room_param) != "1":
            return False

        if expected_page_number and expected_page_number > 1:
            if query.get("p") != str(expected_page_number):


                pass

        if page.locator('a[href*="/sale/flat/"]').count() > 0:
            return True

        if page.locator('article[data-name="CardContainer"]').count() > 0:
            return True

        if page.locator('div[data-testid="offer-card"]').count() > 0:
            return True

    except Exception:
        pass

    return False


def normalize_quicklink_title(value):
    value = clean_text(value)

    if not value:
        return None

    value = value.lower().replace("ё", "е")
    value = re.sub(r"\s+", " ", value).strip()

    return value


def quicklink_segment_key(title):
    """
    Фильтруем только нужные сегменты комнатности.
    URL НЕ хардкодится — href берётся из QuickLinks.
    """
    title_norm = normalize_quicklink_title(title)

    if not title_norm:
        return None

    if re.search(r"^1\s*комнат", title_norm):
        return "1 комнатные"

    if re.search(r"^2\s*комнат", title_norm):
        return "2 комнатные"

    if re.search(r"^3\s*комнат", title_norm):
        return "3 комнатные"

    if re.search(r"^4\s*комнат", title_norm):
        return "4 комнатные"

    if "студи" in title_norm:
        return "студия"

    return None


def parse_quicklink_count(value):
    value = clean_text(value)

    if not value:
        return None

    digits = re.sub(r"\D", "", value)

    return int(digits) if digits else None


def scroll_to_quicklinks(page):
    """
    QuickLinks часто находится ниже выдачи.
    Скроллим страницу, пока блок не появится.
    """
    try:
        page.wait_for_selector('[data-name="QuickLinks"]', timeout=7000)
        return True
    except Exception:
        pass

    try:
        page.wait_for_selector('a[data-name="QuickLinkItem"]', timeout=7000)
        return True
    except Exception:
        pass

    for _ in range(18):
        try:
            page.mouse.wheel(0, random.randint(900, 1500))
            page.wait_for_timeout(random.randint(900, 1500))

            if page.locator('[data-name="QuickLinks"]').count() > 0:
                return True

            if page.locator('a[data-name="QuickLinkItem"]').count() > 0:
                return True

        except Exception:
            pass

    return False


def extract_room_segments_from_quicklinks(page, start_url):
    """
    Открывает стартовую страницу и достаёт ссылки на:
    - 1 комнатные
    - 2 комнатные
    - 3 комнатные
    - 4 комнатные
    - студия
    
    """
    print("Открываю стартовую страницу для поиска QuickLinks:")
    print(start_url)

    page.goto(start_url, wait_until="domcontentloaded", timeout=60000)
    time.sleep(random.uniform(4, 7))

    found = scroll_to_quicklinks(page)

    if not found:
        Path("debug_cian_quicklinks_not_found.html").write_text(
            page.content(),
            encoding="utf-8",
        )

        raise RuntimeError(
            "Не удалось найти блок QuickLinks. "
            "HTML сохранён в debug_cian_quicklinks_not_found.html"
        )

    quicklinks = page.evaluate(
        """
        () => {
            const norm = (text) => {
                return (text || '')
                    .replace(/\\u00a0/g, ' ')
                    .replace(/\\s+/g, ' ')
                    .trim();
            };

            const absUrl = (href) => {
                try {
                    return new URL(href, location.origin).href;
                } catch (e) {
                    return href || null;
                }
            };

            const root =
                document.querySelector('[data-name="QuickLinks"]') ||
                document;

            const links = Array.from(
                root.querySelectorAll('a[data-name="QuickLinkItem"]')
            );

            return links.map(a => {
                const title =
                    norm(a.querySelector('[class*="link-title"]')?.innerText) ||
                    norm(a.querySelector('span')?.innerText) ||
                    norm(a.innerText);

                const count =
                    norm(a.querySelector('[class*="link-count"]')?.innerText);

                return {
                    title,
                    count,
                    url: absUrl(a.getAttribute('href'))
                };
            });
        }
        """
    )

    result_by_segment = {}

    for item in quicklinks:
        title = clean_text(item.get("title"))
        url = clean_text(item.get("url"))
        count = parse_quicklink_count(item.get("count"))

        segment = quicklink_segment_key(title)

        if not segment:
            continue

        if not url:
            continue

        result_by_segment[segment] = {
            "segment": segment,
            "quicklink_title": title,
            "quicklink_count": count,
            "url": url,
        }

    order = {
        "1 комнатные": 1,
        "2 комнатные": 2,
        "3 комнатные": 3,
        "4 комнатные": 4,
        "студия": 5,
    }

    result = sorted(
        result_by_segment.values(),
        key=lambda x: order.get(x["segment"], 999),
    )

    if not result:
        Path("debug_cian_quicklinks_empty.html").write_text(
            page.content(),
            encoding="utf-8",
        )

        raise RuntimeError(
            "QuickLinks найден, но нужные сегменты комнатности не извлечены. "
            "HTML сохранён в debug_cian_quicklinks_empty.html"
        )

    print("\nНайдены сегменты из QuickLinks:")

    for item in result:
        print(
            f'  - {item["segment"]}: {item["url"]} '
            f'({item.get("quicklink_count")})'
        )

    print()

    return result


def is_valid_geo_pair(lat, lon):
    try:
        lat = float(str(lat).replace(",", "."))
        lon = float(str(lon).replace(",", "."))
    except Exception:
        return False


    return 40 <= lat <= 82 and 20 <= lon <= 190


def normalize_coord(value):
    if value is None:
        return None

    value = str(value).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", value)

    return match.group() if match else None


def make_geo_data(lat, lon, source):
    lat = normalize_coord(lat)
    lon = normalize_coord(lon)

    if lat and lon and is_valid_geo_pair(lat, lon):
        return {
            "Широта": lat,
            "Долгота": lon,
            "Геопозиция": f"{lat}, {lon}",
            "Источник геопозиции": source,
        }

    return None


def find_geo_in_text(text):
    if not text:
        return None

    patterns = [
        (
            r'"latitude"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"longitude"\s*:\s*(-?\d+(?:\.\d+)?)',
            "lat_lon",
        ),
        (
            r'"longitude"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"latitude"\s*:\s*(-?\d+(?:\.\d+)?)',
            "lon_lat",
        ),
        (
            r'"lat"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"lng"\s*:\s*(-?\d+(?:\.\d+)?)',
            "lat_lon",
        ),
        (
            r'"lng"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"lat"\s*:\s*(-?\d+(?:\.\d+)?)',
            "lon_lat",
        ),
        (
            r'"lat"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"lon"\s*:\s*(-?\d+(?:\.\d+)?)',
            "lat_lon",
        ),
        (
            r'"lon"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"lat"\s*:\s*(-?\d+(?:\.\d+)?)',
            "lon_lat",
        ),
        (
            r'"coordinates"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            "array",
        ),
        (
            r'"geoCoordinates"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            "array",
        ),
        (
            r'"center"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            "array",
        ),
    ]

    for pattern, mode in patterns:
        match = re.search(pattern, text)

        if not match:
            continue

        a = match.group(1)
        b = match.group(2)

        if mode == "lat_lon":
            geo = make_geo_data(a, b, "html/json text")
            if geo:
                return geo

        elif mode == "lon_lat":
            geo = make_geo_data(b, a, "html/json text")
            if geo:
                return geo

        elif mode == "array":

            geo = make_geo_data(b, a, "html/json coordinates")
            if geo:
                return geo

            geo = make_geo_data(a, b, "html/json coordinates")
            if geo:
                return geo

    return None


def find_geo_in_json(obj, depth=0, max_depth=14):
    if depth > max_depth:
        return None

    if isinstance(obj, dict):
        lower_keys = {str(k).lower(): k for k in obj.keys()}

        if "latitude" in lower_keys and "longitude" in lower_keys:
            lat = obj.get(lower_keys["latitude"])
            lon = obj.get(lower_keys["longitude"])

            geo = make_geo_data(lat, lon, "network json latitude/longitude")
            if geo:
                return geo

        if "lat" in lower_keys and "lng" in lower_keys:
            lat = obj.get(lower_keys["lat"])
            lon = obj.get(lower_keys["lng"])

            geo = make_geo_data(lat, lon, "network json lat/lng")
            if geo:
                return geo

        if "lat" in lower_keys and "lon" in lower_keys:
            lat = obj.get(lower_keys["lat"])
            lon = obj.get(lower_keys["lon"])

            geo = make_geo_data(lat, lon, "network json lat/lon")
            if geo:
                return geo

        if "coordinates" in lower_keys:
            coords = obj.get(lower_keys["coordinates"])

            if isinstance(coords, list) and len(coords) >= 2:
                a = coords[0]
                b = coords[1]

                geo = make_geo_data(b, a, "network json coordinates")
                if geo:
                    return geo

                geo = make_geo_data(a, b, "network json coordinates")
                if geo:
                    return geo

        for value in obj.values():
            geo = find_geo_in_json(value, depth + 1, max_depth)
            if geo:
                return geo

    elif isinstance(obj, list):
        for value in obj:
            geo = find_geo_in_json(value, depth + 1, max_depth)
            if geo:
                return geo

    return None


def setup_cian_geo_capture(page, geo_store):
    """
    Перехватывает ответы ЦИАН/карты и ищет координаты в JSON или текстах.
    geo_store очищается перед каждым объявлением.
    """

    def handle_response(response):
        try:
            if geo_store.get("Геопозиция"):
                return

            url = response.url.lower()

            interesting = [
                "infrastructure",
                "map",
                "geo",
                "coordinates",
                "offer",
                "realty",
                "frontend",
                "microfrontend",
            ]

            if not any(word in url for word in interesting):
                return

            content_type = response.headers.get("content-type", "").lower()

            if "json" in content_type:
                obj = response.json()
                geo = find_geo_in_json(obj)

                if geo:
                    geo_store.update(geo)
                    geo_store["Источник геопозиции"] = (
                        f'{geo["Источник геопозиции"]}: {response.url}'
                    )
                    return

            if (
                "text" in content_type
                or "html" in content_type
                or "javascript" in content_type
            ):
                text = response.text()
                geo = find_geo_in_text(text)

                if geo:
                    geo_store.update(geo)
                    geo_store["Источник геопозиции"] = (
                        f'{geo["Источник геопозиции"]}: {response.url}'
                    )
                    return

        except Exception:
            pass

    page.on("response", handle_response)


def parse_geo_from_dom_attrs(soup):
    data = {
        "Широта": None,
        "Долгота": None,
        "Геопозиция": None,
        "Источник геопозиции": None,
    }

    selectors = [
        "[data-map-lat][data-map-lon]",
        "[data-lat][data-lng]",
        "[data-latitude][data-longitude]",
        "[data-lat][data-lon]",
    ]

    for selector in selectors:
        for el in soup.select(selector):
            attrs = el.attrs

            lat = (
                attrs.get("data-map-lat")
                or attrs.get("data-lat")
                or attrs.get("data-latitude")
                or attrs.get("lat")
                or attrs.get("latitude")
            )

            lon = (
                attrs.get("data-map-lon")
                or attrs.get("data-lng")
                or attrs.get("data-lon")
                or attrs.get("data-longitude")
                or attrs.get("lng")
                or attrs.get("lon")
                or attrs.get("longitude")
            )

            geo = make_geo_data(lat, lon, f"DOM attrs {selector}")

            if geo:
                data.update(geo)
                return data

    return data


def scroll_to_cian_infrastructure_map(page):
    # Доскролливает до карты/инфраструктуры, чтобы ЦИАН сделал сетевые запросы.
    
    selectors = [
        '[data-testid="Infrastructure"]',
        '[data-name="InfrastructureWrapper"]',
        '#infrastructure-microfrontend-map',
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector).first

            if locator.count() > 0:
                locator.scroll_into_view_if_needed(timeout=5000)
                page.wait_for_timeout(3000)
                return True

        except Exception:
            pass

    for _ in range(10):
        try:
            page.mouse.wheel(0, 1200)
            page.wait_for_timeout(400)

            for selector in selectors:
                locator = page.locator(selector).first

                if locator.count() > 0:
                    locator.scroll_into_view_if_needed(timeout=5000)
                    page.wait_for_timeout(3000)
                    return True

        except Exception:
            pass

    return False


def parse_cian_geoposition(soup, html, geo_store):
    data = {
        "Широта": None,
        "Долгота": None,
        "Геопозиция": None,
        "Источник геопозиции": None,
    }

    if geo_store.get("Геопозиция"):
        data.update(geo_store)
        return data

    dom_geo = parse_geo_from_dom_attrs(soup)

    if dom_geo.get("Геопозиция"):
        return dom_geo

    geo = find_geo_in_text(html)

    if geo:
        data.update(geo)
        return data

    return data


def debug_geo_blocks(soup, html, geo_store, url=None):
    print("\n--- DEBUG GEO ---")

    if url:
        print("URL:", url)

    print("geo_store:", geo_store)
    print("latitude в HTML:", "latitude" in html)
    print("longitude в HTML:", "longitude" in html)
    print("coordinates в HTML:", "coordinates" in html)
    print("data-lat:", len(soup.select("[data-lat]")))
    print("data-lng:", len(soup.select("[data-lng]")))
    print("data-map-lat:", len(soup.select("[data-map-lat]")))
    print("Infrastructure:", len(soup.select('[data-testid="Infrastructure"]')))
    print("InfrastructureWrapper:", len(soup.select('[data-name="InfrastructureWrapper"]')))
    print("infrastructure map:", len(soup.select("#infrastructure-microfrontend-map")))

    print("--- END DEBUG GEO ---\n")


def parse_price_and_deal_from_page(page):
    """
    Читаем цену, цену за м² и условия сделки напрямую из Playwright DOM.
    Лучше делать до лишних скроллов, пока правый сайдбар точно в DOM.
    """
    data = {
        "Цена": None,
        "Цена числом": None,
        "Цена за м²": None,
        "Цена за м² числом": None,
        "Условия сделки": None,
    }

    try:
        price_locator = page.locator('[data-testid="price-amount"]').first
        price_locator.wait_for(timeout=15000)

        price_text = clean_text(price_locator.inner_text(timeout=5000))

        data["Цена"] = price_text
        data["Цена числом"] = text_to_int(price_text)

    except Exception:
        pass

    try:
        page.locator('[data-testid="offer-facts"]').first.wait_for(timeout=15000)
    except Exception:
        pass

    try:
        facts = page.evaluate(
            """
            () => {
                const result = {};

                const root = document.querySelector('[data-testid="offer-facts"]');

                if (!root) {
                    return result;
                }

                const items = Array.from(
                    root.querySelectorAll('[data-name="OfferFactItem"]')
                );

                for (const item of items) {
                    const spans = Array.from(item.querySelectorAll('span'))
                        .map(span => {
                            return (span.innerText || span.textContent || '')
                                .replace(/\\s+/g, ' ')
                                .trim();
                        })
                        .filter(Boolean);

                    if (spans.length >= 2) {
                        const label = spans[0];
                        const value = spans[spans.length - 1];

                        result[label] = value;
                    }
                }

                return result;
            }
            """
        )

        price_per_m2 = facts.get("Цена за метр")
        deal_type = facts.get("Условия сделки")

        if price_per_m2:
            data["Цена за м²"] = clean_text(price_per_m2)
            data["Цена за м² числом"] = text_to_int(price_per_m2)

        if deal_type:
            data["Условия сделки"] = clean_text(deal_type)

    except Exception:
        pass

    return data


def expand_description_if_needed(page):
    """
    Раскрывает описание по кнопке 'Узнать больше', если кнопка есть.
    """
    try:
        description = page.locator('[data-name="Description"], #description').first

        if description.count() == 0:
            return

        description.scroll_into_view_if_needed(timeout=5000)
        page.wait_for_timeout(1000)

    except Exception:
        pass

    try:
        toggle = page.locator(
            '[data-name="Description"] [data-mark="ShutterToggle"], '
            '#description [data-mark="ShutterToggle"], '
            '[data-name="Description"] [data-id="toggle"], '
            '#description [data-id="toggle"]'
        ).first

        if toggle.count() > 0:
            toggle_text = clean_text(toggle.inner_text(timeout=2000))

            if toggle_text and "узнать больше" in toggle_text.lower():
                toggle.click(timeout=5000)
                page.wait_for_timeout(1500)

    except Exception:
        pass

    try:
        page.evaluate(
            """
            () => {
                const description = document.querySelector('[data-name="Description"], #description');

                if (!description) {
                    return;
                }

                const contentBlocks = description.querySelectorAll(
                    '[data-mark="Shutter"] > div, '
                    '[class*="content"], '
                    '[data-id="content"]'
                );

                for (const block of contentBlocks) {
                    block.style.maxHeight = 'none';
                    block.style.height = 'auto';
                    block.style.overflow = 'visible';
                }
            }
            """
        )
        page.wait_for_timeout(500)

    except Exception:
        pass


def parse_description_from_page(page):
    data = {
        "Описание": None,
    }

    try:
        expand_description_if_needed(page)

        description_text = page.evaluate(
            """
            () => {
                const description = document.querySelector('[data-name="Description"], #description');

                if (!description) {
                    return null;
                }

                const content =
                    description.querySelector('[data-id="content"]') ||
                    description.querySelector('[data-name="Shutter"] [data-id="content"]') ||
                    description;

                let text = content.innerText || content.textContent || '';

                text = text
                    .replace(/\\u00a0/g, ' ')
                    .replace(/\\n\\s*Узнать больше\\s*$/i, '')
                    .replace(/\\n\\s*Свернуть\\s*$/i, '')
                    .trim();

                return text || null;
            }
            """
        )

        data["Описание"] = clean_description_text(description_text)

    except Exception:
        pass

    return data


def parse_description_from_soup(soup):
    data = {
        "Описание": None,
    }

    description_block = soup.select_one('[data-name="Description"], #description')

    if not description_block:
        return data

    content = description_block.select_one('[data-id="content"]')

    if content:
        text = content.get_text(" ", strip=True)
    else:
        text = description_block.get_text(" ", strip=True)

    text = clean_description_text(text)

    if text:
        text = re.sub(r"\s*Узнать больше\s*$", "", text).strip()
        text = re.sub(r"\s*Свернуть\s*$", "", text).strip()

    data["Описание"] = text if text else None

    return data


def parse_factoids(soup):
    data = {}

    for item in soup.select(
        'div[data-name="ObjectFactoidsItem"], '
        'div[data-name="OfferSummaryInfoItem"]'
    ):
        label = item.select_one(
            'span[class*="color_text-secondary-default"], '
            'p[class*="color_text-secondary-default"]'
        )

        value = item.select_one(
            'span[class*="fontWeight_bold"], '
            'p[class*="color_text-primary-default"]'
        )

        if label and value:
            key = clean_text(label.get_text(" ", strip=True))
            val = clean_text(value.get_text(" ", strip=True))

            if key and val:
                data[key] = val

    return data


def parse_address(soup):
    data = {
        "Полный адрес": None,
        "Регион": None,
        "Округ": None,
        "Населенный пункт": None,
        "Улица": None,
        "Дом": None,
        "Шоссе": None,
        "Метро": None,
    }

    address_block = soup.select_one("address")

    if not address_block:
        return data

    address_items = address_block.select('a[data-name="AddressItem"]')

    parts = [
        clean_text(a.get_text(" ", strip=True))
        for a in address_items
        if clean_text(a.get_text(" ", strip=True))
    ]

    if parts:
        data["Полный адрес"] = ", ".join(parts)
        data["Регион"] = parts[0] if len(parts) > 0 else None
        data["Округ"] = parts[1] if len(parts) > 1 else None
        data["Населенный пункт"] = parts[2] if len(parts) > 2 else None
        data["Улица"] = parts[3] if len(parts) > 3 else None
        data["Дом"] = parts[4] if len(parts) > 4 else None

    highway_items = address_block.select('li[data-name="HighwayItem"]')

    if not highway_items:
        highway_items = soup.select('li[data-name="HighwayItem"]')

    highways = []

    for item in highway_items:
        name_tag = item.select_one(
            'a[class*="highway_link"], '
            'a[data-name="Link"], '
            'a[href*="shosse"], '
            'a'
        )

        dist_tag = item.select_one(
            'span[class*="highway_distance"]'
        )

        name = clean_text(name_tag.get_text(" ", strip=True)) if name_tag else None
        dist = clean_text(dist_tag.get_text(" ", strip=True)) if dist_tag else None

        if name:
            highways.append(f"{name} — {dist}" if dist else name)

    data["Шоссе"] = "; ".join(highways) if highways else None

    underground_items = address_block.select('li[data-name="UndergroundItem"]')

    if not underground_items:
        underground_items = soup.select('li[data-name="UndergroundItem"]')

    metro_list = []

    for item in underground_items:
        station_tag = item.select_one(
            'a[class*="underground_link"], '
            'a[href*="metro"], '
            'a'
        )

        time_tag = item.select_one(
            'span[class*="underground_time"]'
        )

        station = clean_text(station_tag.get_text(" ", strip=True)) if station_tag else None
        time_text = clean_text(time_tag.get_text(" ", strip=True)) if time_tag else None

        if station:
            metro_list.append(f"{station} ({time_text})" if time_text else station)

    data["Метро"] = "; ".join(metro_list) if metro_list else None

    return data


def fill_price_per_m2_from_area(row):
    if row.get("Цена за м²") or row.get("Цена за м² числом"):
        return row

    price = row.get("Цена числом")

    if not price:
        return row

    possible_area_keys = [
        "Общая площадь",
        "Площадь",
        "Площадь квартиры",
        "Площадь комнат",
    ]

    area = None

    for key in possible_area_keys:
        if row.get(key):
            area = parse_area_from_text(row.get(key))
            if area:
                break

    if not area:
        for key, value in row.items():
            if "площад" in str(key).lower():
                area = parse_area_from_text(value)
                if area:
                    break

    if not area:
        return row

    price_per_m2 = price / area

    row["Цена за м² числом"] = int(round(price_per_m2))
    row["Цена за м²"] = format_rub_per_m2(price_per_m2)

    return row


def debug_address_blocks(soup, url=None):
    address_block = soup.select_one("address")

    print("\n--- DEBUG ADDRESS ---")

    if url:
        print("URL:", url)

    print("address найден:", address_block is not None)

    if address_block:
        print(
            "AddressItem внутри address:",
            len(address_block.select('a[data-name="AddressItem"]'))
        )
        print(
            "HighwayItem внутри address:",
            len(address_block.select('li[data-name="HighwayItem"]'))
        )
        print(
            "UndergroundItem внутри address:",
            len(address_block.select('li[data-name="UndergroundItem"]'))
        )

    print(
        "HighwayItem во всём soup:",
        len(soup.select('li[data-name="HighwayItem"]'))
    )
    print(
        "UndergroundItem во всём soup:",
        len(soup.select('li[data-name="UndergroundItem"]'))
    )

    print("--- END DEBUG ---\n")


def debug_price_blocks(page, soup, url=None):
    print("\n--- DEBUG PRICE ---")

    if url:
        print("URL:", url)

    html = page.content()

    print("price-amount в HTML:", "price-amount" in html)
    print("offer-facts в HTML:", "offer-facts" in html)
    print("OfferFactItem в HTML:", "OfferFactItem" in html)
    print("Цена за метр в HTML:", "Цена за метр" in html)
    print("Условия сделки в HTML:", "Условия сделки" in html)

    print(
        'Soup price-amount:',
        len(soup.select('[data-testid="price-amount"]'))
    )
    print(
        'Soup offer-facts:',
        len(soup.select('[data-testid="offer-facts"]'))
    )
    print(
        'Soup OfferFactItem:',
        len(soup.select('[data-name="OfferFactItem"]'))
    )

    print("--- END DEBUG ---\n")


def debug_description_blocks(page, soup, url=None):
    print("\n--- DEBUG DESCRIPTION ---")

    if url:
        print("URL:", url)

    html = page.content()

    print("Description в HTML:", "Description" in html)
    print("description id в HTML:", 'id="description"' in html)
    print("ShutterToggle в HTML:", "ShutterToggle" in html)
    print("Узнать больше в HTML:", "Узнать больше" in html)

    print(
        'Soup Description:',
        len(soup.select('[data-name="Description"], #description'))
    )

    print("--- END DEBUG ---\n")


def wait_listing_page(page):
    selectors = [
        'article[data-name="CardContainer"]',
        'div[data-testid="offer-card"]',
        'a[href*="/sale/flat/"]',
        'a[href*="/flat/"]',
    ]

    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=15000)
            return True
        except Exception:
            pass

    return False


def scroll_listing_page(page, scrolls=12):
    for _ in range(scrolls):
        page.mouse.wheel(0, random.randint(700, 1200))
        page.wait_for_timeout(random.randint(700, 1400))


def collect_cards_from_listing(page):
    cards_data = page.evaluate(
        """
        () => {
            const absUrl = (href) => {
                try {
                    return new URL(href, location.origin).href;
                } catch (e) {
                    return href || '';
                }
            };

            const result = [];
            const seen = new Set();

            const containers = Array.from(
                document.querySelectorAll(
                    'article[data-name="CardContainer"], ' +
                    'div[data-testid="offer-card"], ' +
                    '[data-name="CardComponent"], ' +
                    '[data-name="LinkArea"]'
                )
            );

            for (const card of containers) {
                const link =
                    card.querySelector('a[href*="/sale/flat/"]') ||
                    card.querySelector('a[href*="/flat/"]');

                if (!link) {
                    continue;
                }

                const url = absUrl(link.getAttribute('href') || link.href || '');

                if (!url || seen.has(url)) {
                    continue;
                }

                if (!/\\/sale\\/flat\\/\\d+/.test(url)) {
                    continue;
                }

                seen.add(url);
                result.push({url});
            }

            // Fallback: если контейнеры изменились, ищем ссылки напрямую.
            if (result.length === 0) {
                const links = Array.from(
                    document.querySelectorAll('a[href*="/sale/flat/"]')
                );

                for (const link of links) {
                    const url = absUrl(link.getAttribute('href') || link.href || '');

                    if (!url || seen.has(url)) {
                        continue;
                    }

                    if (!/\\/sale\\/flat\\/\\d+/.test(url)) {
                        continue;
                    }

                    seen.add(url);
                    result.push({url});
                }
            }

            return result;
        }
        """
    )

    return cards_data


def move_description_column_to_end(df):
    if "Описание" in df.columns:
        columns = [col for col in df.columns if col != "Описание"] + ["Описание"]
        df = df[columns]

    return df


def normalize_description_column(df):
    if "Описание" in df.columns:
        df["Описание"] = df["Описание"].apply(clean_description_text)

    return df


def save_results(results, csv_path):
    df = pd.DataFrame(results)

    if df.empty:
        print("   Нет данных для сохранения.")
        return df

    if "Ссылка" in df.columns:
        df = df.drop_duplicates(subset=["Ссылка"], keep="first")

    df = normalize_description_column(df)
    df = move_description_column_to_end(df)

    df.to_csv(
        csv_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )

    print(f"   Сохранено: {len(df)} записей")
    print(f"   CSV: {csv_path}")

    return df


def save_debug_html(page, full_url):
    try:
        offer_id = extract_cian_id(full_url) or "unknown"

        filename = f"debug_cian_{offer_id}.html"
        Path(filename).write_text(page.content(), encoding="utf-8")

        print(f"   Debug HTML сохранён: {filename}")

    except Exception as e:
        print(f"   Не удалось сохранить debug HTML: {e}")


def wait_detail_page(page):
    try:
        page.wait_for_selector("address", timeout=20000)
    except Exception:
        pass

    try:
        page.wait_for_selector('[data-testid="price-amount"]', timeout=15000)
    except Exception:
        pass

    try:
        page.wait_for_selector('[data-testid="offer-facts"]', timeout=15000)
    except Exception:
        pass

    time.sleep(random.uniform(0.8, 1.5))


def parse_detail_card(
    page,
    full_url,
    geo_store,
    room_name,
    room_url,
    room_quicklink_count,
    debug_address=False,
    debug_price=False,
    debug_description=False,
    debug_geo=False,
):
    geo_store.clear()

    page.goto(full_url, wait_until="domcontentloaded", timeout=60000)

    wait_detail_page(page)

    price_data = parse_price_and_deal_from_page(page)

    print("      Цена:", price_data.get("Цена"))
    print("      Цена за м²:", price_data.get("Цена за м²"))
    print("      Условия сделки:", price_data.get("Условия сделки"))

    description_data = parse_description_from_page(page)


    scroll_to_cian_infrastructure_map(page)


    for _ in range(3):
        page.mouse.wheel(0, random.randint(700, 1200))
        time.sleep(random.uniform(0.3, 0.6))

    html = page.content()
    soup = BeautifulSoup(html, "lxml")

    if not description_data.get("Описание"):
        description_data = parse_description_from_soup(soup)

    geo_data = parse_cian_geoposition(soup, html, geo_store)

    if debug_address:
        debug_address_blocks(soup, full_url)

    if debug_price:
        debug_price_blocks(page, soup, full_url)

    if debug_description:
        debug_description_blocks(page, soup, full_url)

    if debug_geo:
        debug_geo_blocks(soup, html, geo_store, full_url)

    offer_id = extract_cian_id(full_url)

    row = {
        "ID": offer_id,
        "Ссылка": full_url,
        "Сегмент комнатности": room_name,
        "URL сегмента": room_url,
        "Количество в QuickLinks": room_quicklink_count,
    }

    row.update(parse_factoids(soup))
    row.update(price_data)
    row.update(parse_address(soup))
    row.update(geo_data)

    row = fill_price_per_m2_from_area(row)

    row.update(description_data)

    print("      Итоговая цена:", row.get("Цена"))
    print("      Итоговая цена за м²:", row.get("Цена за м²"))
    print("      Геопозиция:", row.get("Геопозиция"))
    print("      Шоссе:", row.get("Шоссе"))
    print("      Метро:", row.get("Метро"))
    print("      Описание:", "есть" if row.get("Описание") else None)

    if (
        not row.get("Шоссе")
        and not row.get("Метро")
        and not row.get("Цена")
        and not row.get("Цена за м²")
        and not row.get("Условия сделки")
        and not row.get("Описание")
    ):
        save_debug_html(page, full_url)

    return row


STOP_MARK = "STOP"

LINK_COLUMNS = [
    "source",
    "record_id",
    "listing_id",
    "canonical_url",
    "url",
    "segment",
    "segment_url",
    "quicklink_count",
    "page_number",
    "position_on_page",
    "status",
    "run_marker",
    "collected_at",
    "parsed_at",
    "error",
]


def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_id(value):
    value = clean_text(value)

    if not value:
        return ""

    value = str(value).strip()

    # На случай если Excel/CSV где-то превратил ID в 330123456.0
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]

    return value


def make_record_id(url=None, listing_id=None):
    # record_id создаём ТОЛЬКО для реальной карточки квартиры ЦИАН.
    listing_id = normalize_id(listing_id or extract_cian_id(url))

    if listing_id:
        return f"cian_{listing_id}"

    return ""


def canonical_cian_url(url=None, listing_id=None):
    listing_id = normalize_id(listing_id or extract_cian_id(url))

    if listing_id:
        return f"https://www.cian.ru/sale/flat/{listing_id}/"

    return clean_text(url) or ""


def read_csv_safely(path):
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    for enc in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return pd.read_csv(path, sep=";", dtype=str, encoding=enc).fillna("")
        except UnicodeDecodeError:
            continue

    return pd.read_csv(path, sep=";", dtype=str, encoding="utf-8", errors="replace").fillna("")


def atomic_write_csv(df, path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    df.to_csv(
        tmp_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )

    tmp_path.replace(path)


def get_link_record_id_from_row(row):
    record_id = clean_text(row.get("record_id"))

    if record_id:
        return record_id

    listing_id = normalize_id(row.get("listing_id"))
    url = clean_text(row.get("url"))

    return make_record_id(url=url, listing_id=listing_id)


def ensure_link_columns(df):
    if df.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)

    df = df.copy().fillna("")


    for col in LINK_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    for index, row in df.iterrows():
        url = clean_text(row.get("url")) or ""
        listing_id = normalize_id(row.get("listing_id") or extract_cian_id(url))


        if listing_id and is_cian_flat_detail_url(url):
            record_id = make_record_id(url=url, listing_id=listing_id)
        else:
            record_id = ""

        df.at[index, "source"] = clean_text(row.get("source")) or "cian"
        df.at[index, "listing_id"] = listing_id
        df.at[index, "record_id"] = record_id
        df.at[index, "canonical_url"] = canonical_cian_url(url=url, listing_id=listing_id) if record_id else ""

        if not clean_text(row.get("status")):
            df.at[index, "status"] = "pending"

    ordered = LINK_COLUMNS + [col for col in df.columns if col not in LINK_COLUMNS]

    return df[ordered].fillna("")


def load_links_df(links_csv):
    return ensure_link_columns(read_csv_safely(links_csv))


def save_links_df(df, links_csv):
    df = ensure_link_columns(df)
    atomic_write_csv(df, links_csv)


def get_existing_raw_record_ids(raw_csv):
    df = read_csv_safely(raw_csv)

    if df.empty:
        return set()

    ids = set()

    if "record_id" in df.columns:
        for value in df["record_id"].tolist():
            record_id = clean_text(value)
            if record_id:
                ids.add(record_id)

    for col in ("ID", "listing_id", "source_listing_id"):
        if col in df.columns:
            for value in df[col].tolist():
                listing_id = normalize_id(value)
                if listing_id:
                    ids.add(make_record_id(listing_id=listing_id))

    for col in ("Ссылка", "url", "canonical_url"):
        if col in df.columns:
            for url in df[col].tolist():
                listing_id = extract_cian_id(url)
                if listing_id:
                    ids.add(make_record_id(listing_id=listing_id))

    return ids


def normalize_raw_df(df):
    if df.empty:
        return df

    df = df.copy().fillna("")

    for col in ("record_id", "source", "source_listing_id", "canonical_url"):
        if col not in df.columns:
            df[col] = ""

    for index, row in df.iterrows():
        url = (
            clean_text(row.get("Ссылка"))
            or clean_text(row.get("url"))
            or clean_text(row.get("canonical_url"))
            or ""
        )

        listing_id = normalize_id(
            row.get("source_listing_id")
            or row.get("ID")
            or row.get("listing_id")
            or extract_cian_id(url)
        )

        record_id = clean_text(row.get("record_id")) or make_record_id(url=url, listing_id=listing_id)

        df.at[index, "record_id"] = record_id
        df.at[index, "source"] = clean_text(row.get("source")) or "cian"
        df.at[index, "source_listing_id"] = listing_id
        df.at[index, "canonical_url"] = clean_text(row.get("canonical_url")) or canonical_cian_url(url=url, listing_id=listing_id)

        if listing_id and "ID" in df.columns and not clean_text(row.get("ID")):
            df.at[index, "ID"] = listing_id

    return df.fillna("")


def remove_raw_record(raw_csv, record_id=None, listing_id=None, url=None):
    raw_path = Path(raw_csv)

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return

    df = read_csv_safely(raw_csv)

    if df.empty:
        return

    df = normalize_raw_df(df)

    record_id = clean_text(record_id) or make_record_id(url=url, listing_id=listing_id)
    listing_id = normalize_id(listing_id or extract_cian_id(url))
    url = clean_text(url) or ""

    before = len(df)
    mask_keep = pd.Series([True] * len(df), index=df.index)

    if record_id and "record_id" in df.columns:
        mask_keep = mask_keep & (df["record_id"].astype(str) != record_id)

    if listing_id:
        for col in ("ID", "listing_id", "source_listing_id"):
            if col in df.columns:
                mask_keep = mask_keep & (df[col].astype(str) != listing_id)

    if url:
        for col in ("Ссылка", "url", "canonical_url"):
            if col in df.columns:
                mask_keep = mask_keep & (df[col].astype(str) != url)

    df = df[mask_keep].copy()

    if len(df) != before:
        atomic_write_csv(df, raw_csv)


def upsert_raw_row(raw_csv, row):
    """
    Запись идёт сразу после карточки.
    Одна карточка = один record_id.
    Повторная запись той же карточки перезаписывает строку, а не плодит дубли.
    """
    row = dict(row)

    url = clean_text(row.get("Ссылка") or row.get("url"))
    listing_id = normalize_id(
        row.get("source_listing_id")
        or row.get("ID")
        or row.get("listing_id")
        or extract_cian_id(url)
    )
    record_id = make_record_id(url=url, listing_id=listing_id)

    row["source"] = "cian"
    row["record_id"] = record_id
    row["source_listing_id"] = listing_id
    row["canonical_url"] = canonical_cian_url(url=url, listing_id=listing_id)

    if listing_id and not clean_text(row.get("ID")):
        row["ID"] = listing_id

    remove_raw_record(raw_csv, record_id=record_id, listing_id=listing_id, url=url)

    raw_path = Path(raw_csv)

    if raw_path.exists() and raw_path.stat().st_size > 0:
        df = read_csv_safely(raw_csv)
    else:
        df = pd.DataFrame()

    new_row_df = pd.DataFrame([row])

    if df.empty:
        out_df = new_row_df
    else:
        out_df = pd.concat([df, new_row_df], ignore_index=True, sort=False).fillna("")

    out_df = normalize_raw_df(out_df)

    if "record_id" in out_df.columns:
        out_df = out_df.drop_duplicates(subset=["record_id"], keep="last").copy()

    out_df = normalize_description_column(out_df)
    out_df = move_description_column_to_end(out_df)

    atomic_write_csv(out_df, raw_csv)


def is_valid_detail_row(row):
    if not isinstance(row, dict):
        return False

    strong_fields = [
        "Цена",
        "Цена числом",
        "Цена за м²",
        "Цена за м² числом",
        "Полный адрес",
        "Метро",
        "Описание",
        "Общая площадь",
        "Площадь",
        "Широта",
        "Долгота",
    ]

    return any(clean_text(row.get(field)) for field in strong_fields)


def is_cian_listing_unavailable(page):
    try:
        text = page.locator("body").inner_text(timeout=5000)
    except Exception:
        return False

    text = clean_text(text)

    if not text:
        return False

    text = text.lower().replace("ё", "е")

    phrases = [
        "объявление снято",
        "объявление удалено",
        "объявление не найдено",
        "страница не найдена",
        "такой страницы нет",
        "предложение снято",
        "предложение больше не актуально",
    ]

    return any(phrase in text for phrase in phrases)


def dedupe_links_df(df):
    """
    Убирает дубли по record_id.
    Если есть STOP — сохраняет STOP-строку.
    Иначе сохраняет первую строку данного record_id.
    """
    df = ensure_link_columns(df)

    if df.empty:
        return df, 0

    selected = {}
    order = []

    for index, row in df.iterrows():
        record_id = get_link_record_id_from_row(row)

        if not record_id:
            continue

        marker = str(row.get("run_marker", "")).strip().upper()
        current = selected.get(record_id)

        if record_id not in selected:
            selected[record_id] = (index, row.to_dict())
            order.append(record_id)
            continue

        current_marker = str(current[1].get("run_marker", "")).strip().upper()

        if marker == STOP_MARK and current_marker != STOP_MARK:
            selected[record_id] = (index, row.to_dict())

    rows = [selected[record_id][1] for record_id in order]
    out = pd.DataFrame(rows)

    out = ensure_link_columns(out)
    removed = len(df) - len(out)

    return out, removed


def sync_links_with_raw(df, raw_csv):
    # Если карточка уже есть в raw, links помечается done.
    df = ensure_link_columns(df).copy()
    raw_ids = get_existing_raw_record_ids(raw_csv)
    changed = 0

    if not raw_ids:
        return df, changed

    for index, row in df.iterrows():
        record_id = get_link_record_id_from_row(row)
        marker = str(row.get("run_marker", "")).strip().upper()

        if not record_id:
            continue

        if record_id in raw_ids:

            if (
                str(row.get("status", "")).strip().lower() != "done"
                or marker
                or str(row.get("error", "")).strip()
            ):
                df.at[index, "status"] = "done"
                df.at[index, "run_marker"] = ""
                df.at[index, "error"] = ""
                changed += 1

    return df, changed


def repair_done_without_raw(df, raw_csv):
    
    # Если links говорит done, но raw не содержит record_id — возвращаем в pending.
    
    df = ensure_link_columns(df).copy()
    raw_ids = get_existing_raw_record_ids(raw_csv)
    changed = 0

    for index, row in df.iterrows():
        record_id = get_link_record_id_from_row(row)
        status = str(row.get("status", "")).strip().lower()

        if status == "done" and record_id and record_id not in raw_ids:
            df.at[index, "status"] = "pending"
            df.at[index, "run_marker"] = ""
            df.at[index, "error"] = "requeued: done in links but missing in raw"
            changed += 1

    if changed:
        print(f"Возвращено в pending, потому что нет в raw: {changed}")

    return df


def mark_only_one_stop(df, index):
    df = ensure_link_columns(df).copy()
    df["run_marker"] = ""
    df.at[index, "run_marker"] = STOP_MARK
    return df


def clear_stop_at_index(df, index):
    df = ensure_link_columns(df).copy()
    df.at[index, "run_marker"] = ""
    return df


def find_stop_index(df):
    if df.empty or "run_marker" not in df.columns:
        return None

    stop_rows = df.index[df["run_marker"].astype(str).str.upper() == STOP_MARK].tolist()

    return stop_rows[0] if stop_rows else None


def find_first_pending_index(df, raw_csv, start_index=0):
    df = ensure_link_columns(df)
    raw_ids = get_existing_raw_record_ids(raw_csv)

    for index in range(max(0, start_index), len(df)):
        row = df.iloc[index]
        url = clean_text(row.get("url"))
        record_id = get_link_record_id_from_row(row)
        status = str(row.get("status", "")).strip().lower()

        if not url:
            continue

        if record_id and record_id in raw_ids:
            continue

        if status == "done":

            return index

        return index

    return None


def mark_next_pending_stop(df, links_csv, raw_csv, start_index):
    next_index = find_first_pending_index(df, raw_csv, start_index=start_index)

    if next_index is None:
        return df

    df = mark_only_one_stop(df, next_index)
    save_links_df(df, links_csv)

    print(f"STOP поставлен на следующую необработанную строку links.csv: index={next_index}")

    return df


def keep_stop_and_abort(df, links_csv, index, error_text):
    df = mark_only_one_stop(df, index)
    df.at[index, "status"] = "error"
    df.at[index, "parsed_at"] = now_iso()
    df.at[index, "error"] = str(error_text)[:1000]
    save_links_df(df, links_csv)


def export_xlsx_if_needed(csv_path):
    csv_path = Path(csv_path)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    df = read_csv_safely(csv_path)

    if df.empty:
        return

    xlsx_path = csv_path.with_suffix(".xlsx")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"Excel сохранён: {xlsx_path}")


def build_link_row(room_segment, page_number, position_on_page, url):
    listing_id = extract_cian_id(url)

    if not listing_id or not is_cian_flat_detail_url(url):
        return None

    record_id = make_record_id(url=url, listing_id=listing_id)

    if not record_id:
        return None

    return {
        "source": "cian",
        "record_id": record_id,
        "listing_id": listing_id,
        "canonical_url": canonical_cian_url(url=url, listing_id=listing_id),
        "url": url,
        "segment": room_segment.get("segment"),
        "segment_url": room_segment.get("url"),
        "quicklink_count": room_segment.get("quicklink_count"),
        "page_number": page_number,
        "position_on_page": position_on_page,
        "status": "pending",
        "run_marker": "",
        "collected_at": now_iso(),
        "parsed_at": "",
        "error": "",
    }


def collect_links_mode(args):

    # Собирает ссылки по ЦИАН
    
    links_df = load_links_df(args.links_csv)

    links_df = repair_done_without_raw(links_df, args.csv)
    links_df, synced = sync_links_with_raw(links_df, args.csv)
    links_df, removed = dedupe_links_df(links_df)

    if synced:
        print(f"Синхронизировано с raw как done: {synced}")

    if removed:
        print(f"Удалено дублей из links.csv по record_id: {removed}")

    save_links_df(links_df, args.links_csv)

    seen_record_ids = set(
        get_link_record_id_from_row(row)
        for _, row in links_df.iterrows()
        if get_link_record_id_from_row(row)
    )

    print("Режим: collect-links")
    print(f"Файл ссылок: {args.links_csv}")
    print(f"Уже есть уникальных record_id: {len(seen_record_ids)}")
    print("Страницы выдачи строятся через cat.php + room-параметр + p=N")
    print("Карточки в links.csv сохраняются только как /sale/flat/<ID>/")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=args.profile_dir,
            headless=args.headless,
            viewport={"width": 1440, "height": 960},
            locale="ru-RU",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/134.0.0.0 Safari/537.36"
            ),
            args=["--no-sandbox"],
        )

        page = context.pages[0] if context.pages else context.new_page()

        room_segments = extract_room_segments_from_quicklinks(page, args.start_url)
        if args.max_segments > 0:
            room_segments = room_segments[: args.max_segments]

        total_new_links = 0

        for room_index, room_segment in enumerate(room_segments, start=1):
            room_name = room_segment["segment"]
            room_param = get_room_param_for_segment(room_name)

            print("\n" + "=" * 100)
            print(f"Блок {room_index}/{len(room_segments)}: {room_name}")
            print(f"room-параметр: {room_param}=1")
            print("=" * 100)

            not_observed_pages = 0
            pages_read = 0

            for page_number in range(1, args.max_pages_per_group + 1):
                listing_url = build_cian_room_listing_url(
                    room_segment=room_segment,
                    page_number=page_number,
                    start_url=args.start_url,
                )

                print(f"\nСтраница выдачи {page_number}/{args.max_pages_per_group}: {listing_url}")

                try:
                    page.goto(listing_url, wait_until="domcontentloaded", timeout=60000)
                    time.sleep(random.uniform(args.page_delay_min, args.page_delay_max))

                    wait_listing_page(page)
                    scroll_listing_page(page, scrolls=args.scrolls)

                    observed = listing_page_is_observed(
                        page,
                        expected_page_number=page_number,
                        expected_room_param=room_param,
                    )

                    cards_data = collect_cards_from_listing(page)

                    if not observed or not cards_data:
                        not_observed_pages += 1
                        print(
                            f"Страница выдачи не наблюдается или карточек нет: "
                            f"{not_observed_pages}/{args.no_new_pages}"
                        )

                        if not_observed_pages >= args.no_new_pages:
                            print(f"Сегмент {room_name}: выдача больше не наблюдается. Перехожу к следующему блоку.")
                            break

                        continue

                    not_observed_pages = 0
                    pages_read += 1

                except Exception as e:
                    print(f"Ошибка загрузки/сбора страницы выдачи: {e}")
                    not_observed_pages += 1

                    if not_observed_pages >= args.no_new_pages:
                        print(f"Сегмент {room_name}: достигнут лимит ненаблюдаемых/ошибочных страниц.")
                        break

                    continue

                new_rows = []
                page_seen_ids = set()

                for position, card in enumerate(cards_data, start=1):
                    url = clean_text(card.get("url"))

                    if not url:
                        continue

                    listing_id = extract_cian_id(url)

                    if not listing_id or not is_cian_flat_detail_url(url):
                        print(f"Пропускаю не карточку квартиры: {url}")
                        continue

                    record_id = make_record_id(url=url, listing_id=listing_id)

                    if not record_id:
                        print(f"Пропускаю URL без ID квартиры: {url}")
                        continue

                    if record_id in seen_record_ids or record_id in page_seen_ids:
                        continue

                    link_row = build_link_row(
                        room_segment=room_segment,
                        page_number=page_number,
                        position_on_page=position,
                        url=url,
                    )

                    if not link_row:
                        print(f"Пропускаю невалидную ссылку карточки: {url}")
                        continue

                    seen_record_ids.add(record_id)
                    page_seen_ids.add(record_id)
                    new_rows.append(link_row)

                    if args.max_links and total_new_links + len(new_rows) >= args.max_links:
                        break

                print(f"Найдено ссылок-карточек на странице: {len(cards_data)}")
                print(f"Новых уникальных record_id: {len(new_rows)}")

                if new_rows:
                    links_df = pd.concat(
                        [links_df, pd.DataFrame(new_rows)],
                        ignore_index=True,
                        sort=False,
                    ).fillna("")

                    links_df = ensure_link_columns(links_df)
                    links_df, removed_after_append = dedupe_links_df(links_df)

                    if removed_after_append:
                        print(f"Удалено дублей после добавления: {removed_after_append}")

                    save_links_df(links_df, args.links_csv)

                    total_new_links += len(new_rows)

                    print(f"Всего новых ссылок за запуск: {total_new_links}")
                    print(f"Всего строк в links.csv: {len(links_df)}")
                else:
                    print("Новых уникальных карточек на странице нет, но продолжаю до пропажи выдачи или лимита страниц.")

                if args.max_links and total_new_links >= args.max_links:
                    print("Достигнут лимит --max-links.")
                    break

                time.sleep(random.uniform(args.between_pages_delay_min, args.between_pages_delay_max))

            print(f"Блок {room_name} завершён. Прочитано наблюдаемых страниц: {pages_read}")

            if args.max_links and total_new_links >= args.max_links:
                break

        if args.keep_open:
            input("Браузер оставлен открытым. Нажмите Enter, чтобы закрыть...")

        context.close()

    print("\nСбор ссылок завершён.")
    print(f"Файл ссылок: {args.links_csv}")
    print(f"Всего строк в links.csv: {len(load_links_df(args.links_csv))}")


def analyze_mode(args):
    """
    Читает links.csv и парсит карточки.
    Запись raw происходит сразу после каждой успешной карточки.
    """
    links_df = load_links_df(args.links_csv)

    if links_df.empty:
        raise RuntimeError(
            f"Файл ссылок пустой или не найден: {args.links_csv}. "
            f"Сначала запусти --mode collect-links."
        )

    print("Режим: analyze")
    print(f"Файл ссылок: {args.links_csv}")
    print(f"Файл результата: {args.csv}")
    print(f"Ссылок в links.csv: {len(links_df)}")

    links_df = repair_done_without_raw(links_df, args.csv)
    links_df, synced = sync_links_with_raw(links_df, args.csv)
    links_df, removed = dedupe_links_df(links_df)

    if synced:
        print(f"Синхронизировано с raw как done: {synced}")

    if removed:
        print(f"Удалено дублей из links.csv по record_id: {removed}")

    save_links_df(links_df, args.links_csv)

    stop_index = find_stop_index(links_df)
    raw_ids = get_existing_raw_record_ids(args.csv)

    if stop_index is not None:
        stop_row = links_df.iloc[stop_index]
        stop_record_id = get_link_record_id_from_row(stop_row)
        stop_listing_id = normalize_id(stop_row.get("listing_id"))
        stop_url = clean_text(stop_row.get("url"))

        print(f"Найдена STOP-метка: index={stop_index}, record_id={stop_record_id}, URL={stop_url}")

        if stop_record_id in raw_ids and not args.force_reparse:
            print("STOP указывает на карточку, которая уже есть в raw. Очищаю STOP и считаю done.")
            links_df.at[stop_index, "status"] = "done"
            links_df.at[stop_index, "run_marker"] = ""
            links_df.at[stop_index, "error"] = ""
            save_links_df(links_df, args.links_csv)
            stop_index = None
        else:
            print("Строка будет перепарсена заново.")
            remove_raw_record(
                args.csv,
                record_id=stop_record_id,
                listing_id=stop_listing_id,
                url=stop_url,
            )

            links_df = clear_stop_at_index(links_df, stop_index)
            links_df.at[stop_index, "status"] = "pending"
            links_df.at[stop_index, "error"] = ""
            save_links_df(links_df, args.links_csv)

    if stop_index is not None:
        start_index = stop_index
    elif args.start_index is not None:
        start_index = max(0, args.start_index)
        print(f"STOP не найден. Старт по --start-index: {start_index}")
    else:
        first_pending = find_first_pending_index(links_df, args.csv, start_index=0)

        if first_pending is None:
            print("Нет pending/error записей для анализа. Всё уже обработано.")

            if args.save_xlsx:
                export_xlsx_if_needed(args.csv)

            return

        start_index = first_pending
        print(f"Старт с первой необработанной строки: index={start_index}")

    processed_this_run = 0
    skipped_existing = 0
    closed_this_run = 0
    consecutive_errors = 0
    raw_ids_cache = get_existing_raw_record_ids(args.csv)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=args.profile_dir,
            headless=args.headless,
            viewport={"width": 1440, "height": 960},
            locale="ru-RU",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/134.0.0.0 Safari/537.36"
            ),
            args=["--no-sandbox"],
        )

        page = context.pages[0] if context.pages else context.new_page()

        geo_store = {}
        setup_cian_geo_capture(page, geo_store)

        try:
            for index in range(start_index, len(links_df)):
                link_row = links_df.iloc[index]

                url = clean_text(link_row.get("url"))
                listing_id = normalize_id(link_row.get("listing_id") or extract_cian_id(url))
                record_id = get_link_record_id_from_row(link_row)
                status = str(link_row.get("status", "")).strip().lower()
                marker = str(link_row.get("run_marker", "")).strip().upper()

                if not url:
                    links_df.at[index, "status"] = "error"
                    links_df.at[index, "error"] = "empty url"
                    links_df.at[index, "parsed_at"] = now_iso()
                    save_links_df(links_df, args.links_csv)
                    continue

                if not listing_id or not is_cian_flat_detail_url(url):
                    links_df.at[index, "status"] = "error"
                    links_df.at[index, "run_marker"] = ""
                    links_df.at[index, "error"] = f"not a flat detail url: {url}"
                    links_df.at[index, "parsed_at"] = now_iso()
                    save_links_df(links_df, args.links_csv)
                    print(f"Пропускаю строку links.csv: это не карточка квартиры: {url}")
                    continue

                if (
                    record_id
                    and record_id in raw_ids_cache
                    and marker != STOP_MARK
                    and not args.force_reparse
                ):
                    if status != "done" or marker or str(link_row.get("error", "")).strip():
                        links_df.at[index, "status"] = "done"
                        links_df.at[index, "run_marker"] = ""
                        links_df.at[index, "error"] = ""
                        save_links_df(links_df, args.links_csv)

                    skipped_existing += 1
                    continue

                if args.max_items and processed_this_run >= args.max_items:
                    links_df = mark_next_pending_stop(
                        links_df,
                        args.links_csv,
                        args.csv,
                        start_index=index,
                    )
                    print("Достигнут лимит --max-items.")
                    break

                print("\n" + "=" * 100)
                print(f"Карточка links.csv index={index}, processed={processed_this_run + 1}")
                print(f"record_id: {record_id}")
                print(f"listing_id: {listing_id}")
                print(f"URL: {url}")
                print("=" * 100)

                links_df = mark_only_one_stop(links_df, index)
                links_df.at[index, "status"] = "processing"
                links_df.at[index, "error"] = ""
                save_links_df(links_df, args.links_csv)

                remove_raw_record(
                    args.csv,
                    record_id=record_id,
                    listing_id=listing_id,
                    url=url,
                )

                try:
                    row = parse_detail_card(
                        page=page,
                        full_url=url,
                        geo_store=geo_store,
                        room_name=link_row.get("segment"),
                        room_url=link_row.get("segment_url"),
                        room_quicklink_count=link_row.get("quicklink_count"),
                        debug_address=args.debug_address,
                        debug_price=args.debug_price,
                        debug_description=args.debug_description,
                        debug_geo=args.debug_geo,
                    )

                    if not is_valid_detail_row(row):
                        if is_cian_listing_unavailable(page):
                            print("Карточка недоступна/снята. Помечаю closed и иду дальше.")
                            links_df.at[index, "status"] = "closed"
                            links_df.at[index, "run_marker"] = ""
                            links_df.at[index, "parsed_at"] = now_iso()
                            links_df.at[index, "error"] = "listing unavailable or closed"
                            save_links_df(links_df, args.links_csv)
                            closed_this_run += 1
                            continue

                        raise RuntimeError(
                            "Карточка распарсилась пустой: вероятно капча/блокировка/непрогруженная страница."
                        )

                    row["source"] = "cian"
                    row["record_id"] = record_id
                    row["source_listing_id"] = listing_id
                    row["canonical_url"] = canonical_cian_url(url=url, listing_id=listing_id)
                    row["source_segment"] = link_row.get("segment")
                    row["listing_page_number"] = link_row.get("page_number")
                    row["listing_position_on_page"] = link_row.get("position_on_page")
                    row["link_collected_at"] = link_row.get("collected_at")
                    row["parsed_at"] = now_iso()

                    upsert_raw_row(args.csv, row)

                    raw_ids_after_write = get_existing_raw_record_ids(args.csv)

                    if record_id and record_id not in raw_ids_after_write:
                        raise RuntimeError(
                            f"Raw verification failed: после записи record_id {record_id} не найден в {args.csv}"
                        )

                    links_df.at[index, "status"] = "done"
                    links_df.at[index, "run_marker"] = ""
                    links_df.at[index, "parsed_at"] = now_iso()
                    links_df.at[index, "error"] = ""
                    save_links_df(links_df, args.links_csv)

                    processed_this_run += 1

                    if record_id:
                        raw_ids_cache.add(record_id)

                    consecutive_errors = 0

                    print(f"Готово: {record_id}. За запуск записано: {processed_this_run}")

                except KeyboardInterrupt:
                    print("\nОстановлено пользователем. STOP оставлен на текущей строке.")
                    save_links_df(links_df, args.links_csv)
                    raise

                except Exception as e:
                    error_text = f"{type(e).__name__}: {e}"
                    print(f"Ошибка карточки: {error_text}")
                    traceback.print_exc()

                    try:
                        save_debug_html(page, url)
                    except Exception:
                        pass

                    consecutive_errors += 1

                    if args.continue_errors and consecutive_errors < args.max_consecutive_errors:
                        links_df.at[index, "status"] = "error"
                        links_df.at[index, "run_marker"] = ""
                        links_df.at[index, "parsed_at"] = now_iso()
                        links_df.at[index, "error"] = error_text[:1000]
                        save_links_df(links_df, args.links_csv)
                        continue

                    keep_stop_and_abort(links_df, args.links_csv, index, error_text)
                    print("STOP оставлен на текущей строке. Запусти --mode analyze снова после проверки причины.")
                    raise SystemExit(1)

                time.sleep(random.uniform(args.card_delay_min, args.card_delay_max))

        finally:
            if args.keep_open:
                input("Браузер оставлен открытым. Нажмите Enter, чтобы закрыть...")

            context.close()

    if args.save_xlsx:
        export_xlsx_if_needed(args.csv)

    print("\nАнализ завершён.")
    print(f"За запуск записано новых/перепарсенных карточек: {processed_this_run}")
    print(f"Пропущено, потому что уже есть в raw CSV: {skipped_existing}")
    print(f"Недоступных/закрытых карточек: {closed_this_run}")
    print(f"Raw CSV: {args.csv}")


def clean_links_mode(args):
    """
    Только чистит существующий links.csv от дублей.

    Уникальность считается по record_id/listing_id, а не по сырому URL.
    Это важно, потому что ЦИАН может отдавать разные URL одной квартиры с разными query-параметрами,
    но listing_id у них один.
    """
    links_df = load_links_df(args.links_csv)

    if links_df.empty:
        print(f"Файл ссылок пустой или не найден: {args.links_csv}")
        return

    before = len(links_df)
    links_df, removed = dedupe_links_df(links_df)


    invalid_mask = (
        links_df["record_id"].astype(str).str.strip().eq("")
        | ~links_df["url"].astype(str).apply(is_cian_flat_detail_url)
    )
    invalid_removed = int(invalid_mask.sum())

    if invalid_removed:
        links_df = links_df[~invalid_mask].copy()

    after = len(links_df)

    save_links_df(links_df, args.links_csv)

    print("Режим: clean-links")
    print(f"Файл ссылок: {args.links_csv}")
    print(f"Было строк: {before}")
    print(f"Удалено дублей по record_id/listing_id: {removed}")
    print(f"Удалено не-карточек / поисковых URL: {invalid_removed}")
    print(f"Осталось уникальных ссылок/квартир: {after}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cian.ru — парсер: "
            "collect-links собирает ссылки, analyze парсит карточки с записью после каждой строки."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["collect-links", "analyze", "all", "clean-links"],
        default=None,
        help="collect-links = собрать уникальные ссылки; clean-links = очистить существующий links.csv от дублей; analyze = парсить карточки; all = сначала ссылки, потом карточки.",
    )

    parser.add_argument("--start-url", default=START_URL)
    parser.add_argument("--links-csv", default="cian_links.csv")
    parser.add_argument("--csv", default="cian_raw.csv")
    parser.add_argument("--profile-dir", default="cian_profile")

    parser.add_argument("--max-pages-per-group", type=int, default=10)
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Лимит по сегментам за запуск collect-links. 0 = all segments.",
    )
    parser.add_argument("--no-new-pages", type=int, default=2)
    parser.add_argument("--scrolls", type=int, default=12)

    parser.add_argument("--max-links", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)

    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--force-reparse", action="store_true")
    parser.add_argument("--continue-errors", action="store_true")
    parser.add_argument("--max-consecutive-errors", type=int, default=3)

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--save-xlsx", action="store_true")

    parser.add_argument("--debug-address", action="store_true")
    parser.add_argument("--debug-price", action="store_true")
    parser.add_argument("--debug-description", action="store_true")
    parser.add_argument("--debug-geo", action="store_true")

    parser.add_argument("--page-delay-min", type=float, default=4.0)
    parser.add_argument("--page-delay-max", type=float, default=7.0)
    parser.add_argument("--between-pages-delay-min", type=float, default=6.0)
    parser.add_argument("--between-pages-delay-max", type=float, default=12.0)
    parser.add_argument("--card-delay-min", type=float, default=3.0)
    parser.add_argument("--card-delay-max", type=float, default=5.0)

    args = parser.parse_args()

    if args.mode is None:
        print("\nВыбери режим:")
        print("1 — collect-links: собрать ссылки")
        print("2 — analyze: проанализировать собранные карточки")
        print("3 — all: сначала собрать ссылки, потом проанализировать")
        print("4 — clean-links: только очистить links.csv от дублей")

        choice = input("Введите 1, 2, 3 или 4: ").strip()

        if choice == "1":
            args.mode = "collect-links"
        elif choice == "2":
            args.mode = "analyze"
        elif choice == "3":
            args.mode = "all"
        elif choice == "4":
            args.mode = "clean-links"
        else:
            raise SystemExit("Некорректный выбор режима.")

    print("\n" + "=" * 100)
    print("CIAN reliable parser")
    print("=" * 100)
    print(f"Mode: {args.mode}")
    print(f"Links CSV: {Path(args.links_csv).resolve()}")
    print(f"Raw CSV: {Path(args.csv).resolve()}")
    print("=" * 100)

    if args.mode == "collect-links":
        collect_links_mode(args)

    elif args.mode == "clean-links":
        clean_links_mode(args)

    elif args.mode == "analyze":
        analyze_mode(args)

    elif args.mode == "all":
        collect_links_mode(args)
        analyze_mode(args)


if __name__ == "__main__":
    main()
