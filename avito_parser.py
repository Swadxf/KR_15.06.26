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
    "https://www.avito.ru/moskovskaya_oblast/kvartiry/prodam-ASgBAgICAUSSA8YQ"
    "?context=H4sIAAAAAAAA_wEmANn_YToxOntzOjE6InkiO3M6MTY6IlpObWZoMnhFNW9DdktDRlgiO32ikMmMJgAAAA"
    "&localPriority=0"
)


def clean_text(value):
    if value is None:
        return None
    value = str(value).replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else None


def clean_description_text(value):
    if value is None:
        return None
    value = str(value).replace("\xa0", " ")
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value if value else None


def text_to_int(value):
    value = clean_text(value)
    if not value:
        return None
    digits = re.sub(r"\D", "", value)
    return int(digits) if digits else None


def extract_avito_id(url):
    url = clean_text(url)
    if not url:
        return None
    match = re.search(r"_(\d+)(?:\?|$)", url)
    return match.group(1) if match else None


def extract_rooms_from_title(title):
    title = clean_text(title)
    if not title:
        return None
    title_lower = title.lower()
    if "студ" in title_lower:
        return "Студия"
    match = re.search(r"(\d+)-к\.", title_lower)
    return match.group(1) if match else None


def extract_area_from_title(title):
    title = clean_text(title)
    if not title:
        return None
    match = re.search(r"(\d+(?:[,.]\d+)?)\s*м²", title)
    return match.group(1).replace(",", ".") if match else None


def extract_floor_from_title(title):
    title = clean_text(title)
    if not title:
        return None
    match = re.search(r"(\d+\s*/\s*\d+)\s*эт", title)
    return match.group(1).replace(" ", "") if match else None


def set_page_param(url, page_number):
    parts = urlsplit(url)
    query_items = parse_qsl(parts.query, keep_blank_values=True)
    new_items = [(key, value) for key, value in query_items if key != "p"]
    if page_number > 1:
        new_items.append(("p", str(page_number)))
    new_query = urlencode(new_items, doseq=True)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, new_query, parts.fragment))


def accept_cookies_if_any(page):
    selectors = [
        'button:has-text("Принять")',
        'button:has-text("Согласен")',
        'button:has-text("Хорошо")',
        'button:has-text("Понятно")',
        '[data-marker="cookie-policy/accept"]',
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.click(timeout=2000)
                page.wait_for_timeout(1000)
                return
        except Exception:
            pass


def is_avito_blocked(page):
    try:
        text = page.locator("body").inner_text(timeout=5000).lower()
        phrases = [
            "доступ ограничен",
            "проблема с ip",
            "решения капчи",
            "переместите слайдер",
            "слишком много запросов",
            "подтвердите, что вы не робот",
        ]
        return any(phrase in text for phrase in phrases)
    except Exception:
        return False


def wait_manual_captcha_if_needed(page):
    if not is_avito_blocked(page):
        return
    print("\n!!!Авито показал капчу / ограничение.!!!")
    print("Решить капчу вручную в открытом браузере.")
    print("После успешного возврата на сайт нажми Enter в консоли.\n")
    input("Нажми Enter после прохождения капчи...")
    page.wait_for_timeout(3000)
    if is_avito_blocked(page):
        print("!!!Капча или блокировка всё ещё активна. Лучше остановить парсер и подождать.!!!")
    else:
        print("Капча пройдена, продолжаю...")


def is_avito_listing_closed(page):
    """
    Проверяет страницу-заглушку Авито:
    "Объявление не посмотреть" / "Объявление закрыто" / "Перейти к поиску".
    Если такая страница открылась, ссылку надо пропустить и пометить как closed.
    """
    try:
        text = page.locator("body").inner_text(timeout=7000)
    except Exception:
        return False

    text = clean_text(text)

    if not text:
        return False

    text_lower = text.lower().replace("ё", "е")

    has_closed_title = (
        "объявление не посмотреть" in text_lower
        or "объявление нельзя посмотреть" in text_lower
    )

    has_closed_reason = (
        "объявление закрыто" in text_lower
        or "объявление снято" in text_lower
        or "объявление удалено" in text_lower
        or "уже не актуально" in text_lower
    )

    has_search_button = "перейти к поиску" in text_lower

    return has_closed_title and (has_closed_reason or has_search_button)


def mark_link_closed(links_df, index, links_csv, message="listing closed"):
    """
    Помечает ссылку как закрытую и убирает STOP.
    """
    links_df.at[index, "status"] = "closed"
    links_df.at[index, "run_marker"] = ""
    links_df.at[index, "parsed_at"] = now_iso()
    links_df.at[index, "error"] = message
    save_links_df(links_df, links_csv)
    return links_df


def normalize_room_filter_text(value):
    value = clean_text(value)
    if not value:
        return None
    value = value.lower().replace("ё", "е")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def room_segment_key(label_text):
    text = normalize_room_filter_text(label_text)
    if not text:
        return None
    if "студи" in text:
        return "студия"
    if re.search(r"^1\s+комнат", text):
        return "1 комната"
    if re.search(r"^2\s+комнат", text):
        return "2 комнаты"
    if re.search(r"^3\s+комнат", text):
        return "3 комнаты"
    if re.search(r"^4\s+комнат", text):
        return "4 комнаты"
    return None


def room_segment_order(segment):
    return {
        "1 комната": 1,
        "2 комнаты": 2,
        "3 комнаты": 3,
        "4 комнаты": 4,
        "студия": 5,
    }.get(segment, 999)


def wait_filters_or_listing(page):
    selectors = [
        'label[data-marker^="params[549]/checkbox/"]',
        '[data-marker="item"]',
        '[itemscope][itemtype="http://schema.org/Product"]',
        'a[href*="/kvartiry/"]',
    ]
    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=15000)
            return True
        except Exception:
            pass
    return False


def scroll_to_room_checkboxes(page):
    try:
        page.wait_for_selector('label[data-marker^="params[549]/checkbox/"]', timeout=8000)
        return True
    except Exception:
        pass
    for _ in range(8):
        try:
            page.mouse.wheel(0, random.randint(500, 900))
            page.wait_for_timeout(random.randint(600, 1100))
            if page.locator('label[data-marker^="params[549]/checkbox/"]').count() > 0:
                return True
        except Exception:
            pass
    return False


def extract_room_segments_from_checkboxes(page, start_url):
    print("Открываю стартовую страницу для поиска чекбоксов комнатности:")
    print(start_url)
    page.goto(start_url, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_timeout(random.randint(4000, 7000))
    accept_cookies_if_any(page)
    wait_manual_captcha_if_needed(page)
    wait_filters_or_listing(page)
    found = scroll_to_room_checkboxes(page)
    if not found:
        Path("debug_avito_room_checkboxes_not_found.html").write_text(page.content(), encoding="utf-8")
        raise RuntimeError(
            "Не удалось найти чекбоксы комнатности params[549]. "
            "HTML сохранён в debug_avito_room_checkboxes_not_found.html"
        )
    raw_segments = page.evaluate(
        r"""
        () => {
            const norm = (text) => {
                return (text || '')
                    .replace(/\u00a0/g, ' ')
                    .replace(/\s+/g, ' ')
                    .trim();
            };
            const labels = Array.from(
                document.querySelectorAll('label[data-marker^="params[549]/checkbox/"][role="checkbox"]')
            );
            return labels.map(label => {
                const input = label.querySelector('input[type="checkbox"]');
                return {
                    labelText: norm(label.innerText),
                    dataMarker: label.getAttribute('data-marker'),
                    ariaChecked: label.getAttribute('aria-checked'),
                    inputName: input ? input.getAttribute('name') : null,
                    inputValue: input ? input.getAttribute('value') : null
                };
            });
        }
        """
    )
    result_by_segment = {}
    for item in raw_segments:
        label_text = clean_text(item.get("labelText"))
        segment = room_segment_key(label_text)
        if not segment:
            continue
        result_by_segment[segment] = {
            "segment": segment,
            "label_text": label_text,
            "data_marker": clean_text(item.get("dataMarker")),
            "input_name": clean_text(item.get("inputName")),
            "input_value": clean_text(item.get("inputValue")),
        }
    result = sorted(result_by_segment.values(), key=lambda x: room_segment_order(x["segment"]))
    if not result:
        Path("debug_avito_room_checkboxes_empty.html").write_text(page.content(), encoding="utf-8")
        raise RuntimeError(
            "Чекбоксы params[549] найдены, но нужные сегменты не распознаны. "
            "HTML сохранён в debug_avito_room_checkboxes_empty.html"
        )
    print("\nНайдены сегменты комнатности из чекбоксов:")
    for item in result:
        print(
            f'  - {item["segment"]}: '
            f'label="{item["label_text"]}", '
            f'value={item["input_value"]}, '
            f'marker={item["data_marker"]}'
        )
    print()
    return result


def find_room_checkbox_locator(page, room_segment):
    data_marker = room_segment.get("data_marker")
    label_text = room_segment.get("label_text")
    if data_marker:
        locator = page.locator(f'label[data-marker="{data_marker}"]').first
        try:
            if locator.count() > 0:
                return locator
        except Exception:
            pass
    if label_text:
        locator = page.locator(f'label[role="checkbox"]:has-text("{label_text}")').first
        try:
            if locator.count() > 0:
                return locator
        except Exception:
            pass
    segment = room_segment.get("segment")
    if segment:
        locator = page.locator(f'label[role="checkbox"]:has-text("{segment}")').first
        try:
            if locator.count() > 0:
                return locator
        except Exception:
            pass
    return None


def apply_room_segment_filter(page, start_url, room_segment):
    segment = room_segment["segment"]
    print(f"\nНастраиваю фильтр сегмента: {segment}")
    page.goto(start_url, wait_until="domcontentloaded", timeout=90000)
    page.wait_for_timeout(random.randint(4000, 7000))
    accept_cookies_if_any(page)
    wait_manual_captcha_if_needed(page)
    wait_filters_or_listing(page)
    scroll_to_room_checkboxes(page)
    locator = find_room_checkbox_locator(page, room_segment)
    if locator is None:
        safe_segment = re.sub(r"\W+", "_", segment)
        Path(f"debug_avito_checkbox_{safe_segment}.html").write_text(page.content(), encoding="utf-8")
        raise RuntimeError(
            f"Не удалось найти чекбокс для сегмента {segment}. "
            f"HTML сохранён в debug_avito_checkbox_{safe_segment}.html"
        )
    try:
        locator.scroll_into_view_if_needed(timeout=5000)
        page.wait_for_timeout(700)
    except Exception:
        pass
    try:
        checked = locator.get_attribute("aria-checked")
        if checked != "true":
            locator.click(timeout=5000)
            page.wait_for_timeout(random.randint(3000, 5000))
    except Exception:
        try:
            locator.locator('input[type="checkbox"]').click(force=True, timeout=5000)
            page.wait_for_timeout(random.randint(3000, 5000))
        except Exception:
            raise
    try:
        page.wait_for_load_state("networkidle", timeout=15000)
    except Exception:
        pass
    wait_manual_captcha_if_needed(page)
    wait_listing_page(page)
    filtered_url = page.url
    print(f"URL после установки фильтра {segment}:")
    print(filtered_url)
    if filtered_url == start_url:
        print("!!! URL после клика не изменился. Авито мог держать фильтр только в состоянии страницы. !!!")
    return filtered_url


def wait_listing_page(page):
    selectors = [
        '[data-marker="item"]',
        '[data-marker="item-title"]',
        'a[data-marker="title"]',
        '[itemscope][itemtype="http://schema.org/Product"]',
        'a[href*="/kvartiry/"]',
    ]
    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=15000)
            return True
        except Exception:
            pass
    return False


def scroll_listing_page(page, scrolls=10):
    for _ in range(scrolls):
        page.mouse.wheel(0, random.randint(800, 1400))
        page.wait_for_timeout(random.randint(900, 1600))


def collect_links_from_listing(page):
    rows = page.evaluate(
        r"""
        () => {
            const norm = (text) => {
                return (text || '')
                    .replace(/\u00a0/g, ' ')
                    .replace(/\s+/g, ' ')
                    .trim();
            };
            const absUrl = (href) => {
                try { return new URL(href, location.origin).href; }
                catch (e) { return href || null; }
            };
            const candidates = Array.from(
                document.querySelectorAll(
                    '[data-marker="item"], ' +
                    '[itemscope][itemtype="http://schema.org/Product"], ' +
                    'div[data-testid="item"], ' +
                    'article'
                )
            );
            const result = [];
            const seen = new Set();
            for (const card of candidates) {
                const link =
                    card.querySelector('a[data-marker="item-title"]') ||
                    card.querySelector('a[data-marker="title"]') ||
                    card.querySelector('a[itemprop="url"][href*="/kvartiry/"]') ||
                    card.querySelector('a[href*="/kvartiry/"]');
                if (!link) continue;
                const href = link.getAttribute('href') || '';
                const url = absUrl(href);
                if (!url || seen.has(url)) continue;
                if (!/_(\d+)(\?|$)/.test(url)) continue;
                seen.add(url);
                const title =
                    norm(link.innerText) ||
                    norm(link.getAttribute('title')) ||
                    norm(card.querySelector('[itemprop="name"]')?.innerText);
                const priceText =
                    norm(card.querySelector('[data-marker="item-price-value"]')?.innerText) ||
                    norm(card.querySelector('meta[itemprop="price"]')?.getAttribute('content'));
                const priceMeta =
                    norm(card.querySelector('meta[itemprop="price"]')?.getAttribute('content'));
                const locationBlock =
                    card.querySelector('[data-marker="item-location"]') ||
                    card.querySelector('[itemprop="address"]');
                const address = locationBlock ? norm(locationBlock.innerText) : null;
                const description =
                    norm(card.querySelector('meta[itemprop="description"]')?.getAttribute('content'));
                result.push({ url, title, priceText, priceMeta, address, description });
            }
            return result;
        }
        """
    )
    parsed = []
    for row in rows:
        url = clean_text(row.get("url"))
        title = clean_text(row.get("title"))
        price_text = clean_text(row.get("priceText"))
        price_meta = clean_text(row.get("priceMeta"))
        if price_meta and str(price_meta).isdigit():
            price_numeric = int(price_meta)
        else:
            price_numeric = text_to_int(price_text)
        parsed.append(
            {
                "ID": extract_avito_id(url),
                "Ссылка": url,
                "Название из выдачи": title,
                "Цена из выдачи": price_text,
                "Цена из выдачи числом": price_numeric,
                "Адрес из выдачи": clean_text(row.get("address")),
                "Описание из выдачи": clean_description_text(row.get("description")),
            }
        )
    return parsed


def wait_detail_page(page):
    selectors = [
        '[data-marker="item-view/item-price-container"]',
        '[data-marker="item-view/item-price"]',
        '[data-marker="item-view/item-params"]',
        '[data-marker="item-view/item-description"]',
        '[itemprop="address"]',
        'h1',
    ]
    for selector in selectors:
        try:
            page.wait_for_selector(selector, timeout=20000)
            return True
        except Exception:
            pass
    return False


def expand_description_if_needed(page):
    selectors = [
        'button:has-text("Показать полностью")',
        'button:has-text("Развернуть")',
        'button:has-text("Ещё")',
        '[data-marker="item-view/item-description"] button',
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if locator.count() > 0:
                locator.click(timeout=3000)
                page.wait_for_timeout(1000)
                break
        except Exception:
            pass


def parse_title(soup):
    selectors = [
        'h1[itemprop="name"]',
        'h1',
        '[data-marker="item-view/title-info"]',
        '[data-marker="item-view/item-title"]',
    ]
    for selector in selectors:
        el = soup.select_one(selector)
        if el:
            title = clean_text(el.get_text(" ", strip=True))
            if title:
                return title
    return None


def parse_price(soup):
    data = {
        "Цена": None,
        "Цена числом": None,
        "Цена за м²": None,
        "Цена за м² числом": None,
    }
    price_el = soup.select_one('[data-marker="item-view/item-price"]')
    if price_el:
        price_text = clean_text(price_el.get_text(" ", strip=True))
        price_content = price_el.get("content")
        data["Цена"] = price_text
        data["Цена числом"] = text_to_int(price_content or price_text)
    if not data["Цена числом"]:
        price_meta = soup.select_one('[itemprop="price"][content]')
        if price_meta:
            data["Цена числом"] = text_to_int(price_meta.get("content"))
    if not data["Цена"] and data["Цена числом"]:
        data["Цена"] = f'{data["Цена числом"]:,}'.replace(",", " ") + " ₽"
    price_per_m2 = None
    for el in soup.find_all(["p", "span", "div"]):
        text = clean_text(el.get_text(" ", strip=True))
        if not text or len(text) > 80:
            continue
        match = re.search(r"\d[\d\s]*₽\s*за\s*м²", text, flags=re.I)
        if match:
            price_per_m2 = clean_text(match.group())
            break
    if not price_per_m2:
        full_text = clean_text(soup.get_text(" ", strip=True))
        if full_text:
            match = re.search(r"\d[\d\s]*₽\s*за\s*м²", full_text, flags=re.I)
            if match:
                price_per_m2 = clean_text(match.group())
    if price_per_m2:
        data["Цена за м²"] = price_per_m2
        data["Цена за м² числом"] = text_to_int(price_per_m2)
    return data


def parse_params_sections(soup):
    data = {}
    sections = soup.find_all(attrs={"data-marker": "item-view/item-params"})
    for section in sections:
        title_el = section.find("h2")
        section_title = clean_text(title_el.get_text(" ", strip=True)) if title_el else "Параметры"
        for li in section.find_all("li"):
            full_text = clean_text(li.get_text(" ", strip=True))
            if not full_text:
                continue
            label_el = li.find("span")
            label = clean_text(label_el.get_text(" ", strip=True)) if label_el else None
            if label:
                label = label.replace(":", "")
                label = clean_text(label)
                value = full_text
                value = re.sub(r"^" + re.escape(label) + r"\s*:?\s*", "", value)
                value = clean_text(value)
            else:
                if ":" not in full_text:
                    continue
                label, value = full_text.split(":", 1)
                label = clean_text(label)
                value = clean_text(value)
            if not label or not value:
                continue
            if len(label) > 80:
                continue
            if section_title == "О квартире":
                data[label] = value
            else:
                data[f"{section_title}: {label}"] = value
    return data


def parse_address(soup):
    data = {
        "Полный адрес": None,
        "Метро": None,
        "Широта": None,
        "Долгота": None,
    }
    address_block = soup.select_one('[itemprop="address"]')
    if address_block:
        first_span = address_block.find("span")
        if first_span:
            data["Полный адрес"] = clean_text(first_span.get_text(" ", strip=True))
        else:
            data["Полный адрес"] = clean_text(address_block.get_text(" ", strip=True))
        texts = [clean_text(text) for text in address_block.stripped_strings if clean_text(text)]
        if data["Полный адрес"] in texts:
            texts = texts[texts.index(data["Полный адрес"]) + 1:]
        metro_items = []
        i = 0
        while i < len(texts):
            current = texts[i]
            next_text = texts[i + 1] if i + 1 < len(texts) else None
            if not current:
                i += 1
                continue
            if next_text and re.search(r"мин|пешком|транспорт", next_text, flags=re.I):
                metro_items.append(f"{current} ({next_text})")
                i += 2
            else:
                if not re.search(r"д\.|ул\.|корп|строен|обл|район", current, flags=re.I):
                    metro_items.append(current)
                i += 1
        data["Метро"] = "; ".join(metro_items) if metro_items else None
    map_el = soup.select_one('[data-marker="item-map-wrapper"]')
    if map_el:
        data["Широта"] = clean_text(map_el.get("data-map-lat"))
        data["Долгота"] = clean_text(map_el.get("data-map-lon"))
    return data


def parse_description(soup):
    data = {"Описание": None}
    description_el = soup.select_one('[data-marker="item-view/item-description"]')
    if description_el:
        data["Описание"] = clean_description_text(description_el.get_text(" ", strip=True))
    if not data["Описание"]:
        meta = soup.select_one('meta[itemprop="description"]')
        if meta and meta.get("content"):
            data["Описание"] = clean_description_text(meta.get("content"))
    return data


def safe_call(name, func, default):
    try:
        return func()
    except Exception as e:
        print(f"      Ошибка в {name}: {e}")
        traceback.print_exc()
        return default


def parse_detail_page(page, full_url):
    page.wait_for_timeout(random.randint(1500, 3000))
    accept_cookies_if_any(page)
    wait_manual_captcha_if_needed(page)
    wait_detail_page(page)
    for _ in range(4):
        try:
            page.mouse.wheel(0, random.randint(500, 900))
            page.wait_for_timeout(random.randint(700, 1300))
        except Exception:
            break
    expand_description_if_needed(page)
    html = page.content()
    soup = BeautifulSoup(html, "lxml")
    title = safe_call("parse_title", lambda: parse_title(soup), None)
    price_data = safe_call("parse_price", lambda: parse_price(soup), {})
    params_data = safe_call("parse_params_sections", lambda: parse_params_sections(soup), {})
    address_data = safe_call("parse_address", lambda: parse_address(soup), {})
    description_data = safe_call("parse_description", lambda: parse_description(soup), {})
    row = {
        "ID": extract_avito_id(full_url),
        "Ссылка": full_url,
        "Название": title,
        "Комнат": extract_rooms_from_title(title),
        "Площадь м² из названия": extract_area_from_title(title),
        "Этаж из названия": extract_floor_from_title(title),
    }
    row.update(price_data)
    row.update(params_data)
    row.update(address_data)
    row.update(description_data)
    return row


def move_description_to_end(df):
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
        print("Нет данных для сохранения.")
        return df
    if "Ссылка" in df.columns:
        df = df.drop_duplicates(subset=["Ссылка"], keep="first")
    df = normalize_description_column(df)
    df = move_description_to_end(df)
    df.to_csv(
        csv_path,
        sep=";",
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    print(f"CSV сохранён: {csv_path}")

    print(f"Строк сохранено: {len(df)}")
    return df


def save_debug_html(page, full_url):
    try:
        offer_id = extract_avito_id(full_url) or "unknown"
        filename = f"debug_avito_{offer_id}.html"
        Path(filename).write_text(page.content(), encoding="utf-8")
        print(f"      Debug HTML сохранён: {filename}")
    except Exception as e:
        print(f"      Не удалось сохранить debug HTML: {e}")


LINK_COLUMNS = [
    "source",
    "segment",
    "segment_url",
    "checkbox_label",
    "checkbox_value",
    "checkbox_marker",
    "page_number",
    "position_on_page",
    "listing_id",
    "url",
    "title_from_listing",
    "price_from_listing",
    "price_from_listing_num",
    "address_from_listing",
    "description_from_listing",
    "status",
    "run_marker",
    "collected_at",
    "parsed_at",
    "error",
]

STOP_MARK = "STOP"


def now_iso():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
    # Атомарная запись CSV: сначала пишем во временный файл, потом заменяем основной.
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


def ensure_link_columns(df):
    if df.empty:
        return pd.DataFrame(columns=LINK_COLUMNS)

    for col in LINK_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    ordered = LINK_COLUMNS + [col for col in df.columns if col not in LINK_COLUMNS]

    return df[ordered].fillna("")


def load_links_df(links_csv):
    return ensure_link_columns(read_csv_safely(links_csv))


def save_links_df(df, links_csv):
    df = ensure_link_columns(df)
    atomic_write_csv(df, links_csv)


def normalize_id(value):
    value = clean_text(value)

    if not value:
        return ""

    return str(value)


def get_existing_raw_ids(raw_csv):
    df = read_csv_safely(raw_csv)

    if df.empty:
        return set()

    ids = set()

    for col in ("ID", "listing_id", "source_listing_id"):
        if col in df.columns:
            ids.update(
                normalize_id(value)
                for value in df[col].tolist()
                if normalize_id(value)
            )

    for col in ("Ссылка", "url"):
        if col in df.columns:
            for url in df[col].tolist():
                listing_id = extract_avito_id(url)
                if listing_id:
                    ids.add(normalize_id(listing_id))

    return ids


def remove_raw_record(raw_csv, listing_id=None, url=None):
    """
    Удаляет строку из raw CSV перед повторным парсингом STOP-записи.
    Защита от дублей на стыке.
    """
    raw_path = Path(raw_csv)

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        return

    df = read_csv_safely(raw_csv)

    if df.empty:
        return

    before = len(df)

    listing_id = normalize_id(listing_id)
    url = clean_text(url) or ""

    mask_keep = pd.Series([True] * len(df))

    if listing_id:
        for col in ("ID", "listing_id", "source_listing_id"):
            if col in df.columns:
                mask_keep = mask_keep & (df[col].astype(str) != listing_id)

    if url:
        for col in ("Ссылка", "url"):
            if col in df.columns:
                mask_keep = mask_keep & (df[col].astype(str) != url)

    df = df[mask_keep].copy()

    if len(df) != before:
        atomic_write_csv(df, raw_csv)


def upsert_raw_row(raw_csv, row):
    # Перезаписывает строку по ID/Ссылке, если она уже есть.
    listing_id = normalize_id(row.get("ID") or row.get("listing_id") or row.get("source_listing_id"))
    url = clean_text(row.get("Ссылка") or row.get("url"))

    remove_raw_record(raw_csv, listing_id=listing_id, url=url)

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

    out_df = normalize_description_column(out_df)
    out_df = move_description_to_end(out_df)

    atomic_write_csv(out_df, raw_csv)


def mark_only_one_stop(df, index):
    df = df.copy()

    if "run_marker" not in df.columns:
        df["run_marker"] = ""

    df["run_marker"] = ""
    df.at[index, "run_marker"] = STOP_MARK

    return df


def clear_stop_at_index(df, index):
    df = df.copy()

    if "run_marker" not in df.columns:
        df["run_marker"] = ""

    df.at[index, "run_marker"] = ""

    return df


def find_stop_index(df):
    if df.empty or "run_marker" not in df.columns:
        return None

    stop_rows = df.index[df["run_marker"].astype(str).str.upper() == STOP_MARK].tolist()

    return stop_rows[0] if stop_rows else None


def find_first_pending_index(df, raw_csv, start_index=0):
    raw_done_ids = get_existing_raw_ids(raw_csv)

    for index in range(max(0, start_index), len(df)):
        row = df.iloc[index]

        status = str(row.get("status", "")).strip().lower()
        listing_id = normalize_id(row.get("listing_id"))
        url = clean_text(row.get("url"))

        if not url:
            continue

        if listing_id and listing_id in raw_done_ids and status == "done":
            continue

        if status == "done":
            continue

        return index

    return None


def mark_next_pending_stop(df, links_csv, raw_csv, start_index):
    #Повторный запуск
    next_index = find_first_pending_index(df, raw_csv, start_index=start_index)

    if next_index is None:
        return df

    df = mark_only_one_stop(df, next_index)
    save_links_df(df, links_csv)

    print(f"STOP поставлен на следующую необработанную строку links.csv: index={next_index}")

    return df


def export_xlsx_if_needed(csv_path):
    # Отключена
    csv_path = Path(csv_path)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return

    df = read_csv_safely(csv_path)

    if df.empty:
        return

    xlsx_path = csv_path.with_suffix(".xlsx")

    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    print(f"Excel сохранён: {xlsx_path}")


def build_link_row(room_segment, filtered_url, page_number, position_on_page, preview):
    url = clean_text(preview.get("Ссылка"))
    listing_id = extract_avito_id(url)

    return {
        "source": "avito",
        "segment": room_segment.get("segment"),
        "segment_url": filtered_url,
        "checkbox_label": room_segment.get("label_text"),
        "checkbox_value": room_segment.get("input_value"),
        "checkbox_marker": room_segment.get("data_marker"),
        "page_number": page_number,
        "position_on_page": position_on_page,
        "listing_id": listing_id,
        "url": url,
        "title_from_listing": preview.get("Название из выдачи"),
        "price_from_listing": preview.get("Цена из выдачи"),
        "price_from_listing_num": preview.get("Цена из выдачи числом"),
        "address_from_listing": preview.get("Адрес из выдачи"),
        "description_from_listing": preview.get("Описание из выдачи"),
        "status": "pending",
        "run_marker": "",
        "collected_at": now_iso(),
        "parsed_at": "",
        "error": "",
    }


def collect_links_mode(args):
    
    # Режим 1: Собирает ссылки со всех страниц всех сегментов Авито.

    links_df = load_links_df(args.links_csv)

    seen_urls = set(
        clean_text(url)
        for url in links_df["url"].tolist()
        if clean_text(url)
    )

    print("Режим: collect-links")
    print(f"Файл ссылок: {args.links_csv}")
    print(f"Уже есть ссылок в файле: {len(seen_urls)}")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=args.profile_dir,
            headless=args.headless,
            viewport={"width": 1440, "height": 1000},
            locale="ru-RU",
        )

        page = context.pages[0] if context.pages else context.new_page()

        room_segments = extract_room_segments_from_checkboxes(page, args.url)
        if args.max_segments > 0:
            room_segments = room_segments[: args.max_segments]

        total_new_links = 0

        for room_index, room_segment in enumerate(room_segments, start=1):
            room_name = room_segment["segment"]

            print("\n" + "=" * 100)
            print(f"Блок {room_index}/{len(room_segments)}: {room_name}")
            print(
                f'Чекбокс: {room_segment.get("label_text")} / '
                f'value={room_segment.get("input_value")}'
            )
            print("=" * 100)

            try:
                filtered_url = apply_room_segment_filter(page, args.url, room_segment)
            except Exception as e:
                print(f"Не удалось настроить сегмент {room_name}: {e}")
                traceback.print_exc()
                continue

            no_new_rounds = 0

            for page_number in range(1, args.max_pages_per_group + 1):
                listing_url = set_page_param(filtered_url, page_number)

                print(f"\nСтраница {page_number}/{args.max_pages_per_group}: {listing_url}")

                try:
                    page.goto(listing_url, wait_until="domcontentloaded", timeout=90000)
                    page.wait_for_timeout(random.randint(args.page_delay_min_ms, args.page_delay_max_ms))

                    accept_cookies_if_any(page)
                    wait_manual_captcha_if_needed(page)
                    wait_listing_page(page)
                    scroll_listing_page(page, scrolls=args.scrolls)

                    preview_rows = collect_links_from_listing(page)

                except Exception as e:
                    print(f"Ошибка загрузки/сбора страницы выдачи: {e}")
                    traceback.print_exc()

                    no_new_rounds += 1

                    if no_new_rounds >= args.no_new_pages:
                        print(f"Сегмент {room_name}: достигнут лимит пустых/ошибочных страниц.")
                        break

                    continue

                new_rows = []

                for position, preview in enumerate(preview_rows, start=1):
                    url = clean_text(preview.get("Ссылка"))

                    if not url:
                        continue

                    if url in seen_urls:
                        continue

                    seen_urls.add(url)
                    new_rows.append(
                        build_link_row(
                            room_segment=room_segment,
                            filtered_url=filtered_url,
                            page_number=page_number,
                            position_on_page=position,
                            preview=preview,
                        )
                    )

                    if args.max_links and total_new_links + len(new_rows) >= args.max_links:
                        break

                print(f"Найдено ссылок на странице: {len(preview_rows)}")
                print(f"Новых ссылок: {len(new_rows)}")

                if new_rows:
                    links_df = pd.concat(
                        [links_df, pd.DataFrame(new_rows)],
                        ignore_index=True,
                        sort=False,
                    ).fillna("")

                    links_df = ensure_link_columns(links_df)
                    save_links_df(links_df, args.links_csv)

                    total_new_links += len(new_rows)

                    print(f"Всего новых ссылок за запуск: {total_new_links}")
                    print(f"Всего ссылок в links.csv: {len(links_df)}")

                    no_new_rounds = 0
                else:
                    no_new_rounds += 1
                    print(f"Нет новых страниц подряд: {no_new_rounds}/{args.no_new_pages}")

                    if no_new_rounds >= args.no_new_pages:
                        print(f"Сегмент {room_name} завершён.")
                        break

                if args.max_links and total_new_links >= args.max_links:
                    print("Достигнут лимит --max-links.")
                    break

                time.sleep(random.uniform(args.between_pages_delay_min, args.between_pages_delay_max))

            if args.max_links and total_new_links >= args.max_links:
                break

            time.sleep(random.uniform(args.between_segments_delay_min, args.between_segments_delay_max))

        if args.keep_open:
            input("Браузер оставлен открытым. Нажмите Enter, чтобы закрыть...")

        context.close()

    print("\nСбор ссылок завершён.")
    print(f"Файл ссылок: {args.links_csv}")
    print(f"Всего ссылок в файле: {len(load_links_df(args.links_csv))}")


def analyze_mode(args):
    """
    Режим 2:
    Читает links.csv и парсит карточки.
    Resume-логика:
    - перед карточкой ставит STOP в links.csv;
    - после успешной карточки убирает STOP и ставит status=done;
    - если скрипт упал, STOP остаётся;
    - при новом запуске скрипт начинает с STOP-строки;
    - raw-строка по этой карточке перед повторным парсингом удаляется и пишется заново.
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

    stop_index = find_stop_index(links_df)

    if stop_index is not None:
        start_index = stop_index

        stop_row = links_df.iloc[start_index]
        stop_id = normalize_id(stop_row.get("listing_id"))
        stop_url = clean_text(stop_row.get("url"))

        print(f"Найдена STOP-метка: index={start_index}, ID={stop_id}, URL={stop_url}")
        print("Строка будет перепарсена заново, чтобы не было дублей/битой записи.")

        remove_raw_record(args.csv, listing_id=stop_id, url=stop_url)

        links_df = clear_stop_at_index(links_df, start_index)
        links_df.at[start_index, "status"] = "pending"
        links_df.at[start_index, "error"] = ""
        save_links_df(links_df, args.links_csv)

    elif args.start_index is not None:
        start_index = max(0, args.start_index)
        print(f"STOP не найден. Старт по --start-index: {start_index}")
    else:
        first_pending = find_first_pending_index(links_df, args.csv, start_index=0)

        if first_pending is None:
            print("Нет pending/error записей для анализа. Всё уже обработано.")
            export_xlsx_if_needed(args.csv)
            return

        start_index = first_pending
        print(f"STOP не найден. Старт с первой необработанной строки: index={start_index}")

    processed_this_run = 0
    closed_this_run = 0

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=args.profile_dir,
            headless=args.headless,
            viewport={"width": 1440, "height": 1000},
            locale="ru-RU",
        )

        page = context.pages[0] if context.pages else context.new_page()

        try:
            for index in range(start_index, len(links_df)):
                link_row = links_df.iloc[index]

                url = clean_text(link_row.get("url"))
                listing_id = normalize_id(link_row.get("listing_id") or extract_avito_id(url))
                status = str(link_row.get("status", "")).strip().lower()

                if not url:
                    links_df.at[index, "status"] = "error"
                    links_df.at[index, "error"] = "empty url"
                    links_df.at[index, "parsed_at"] = now_iso()
                    save_links_df(links_df, args.links_csv)
                    continue

                raw_done_ids = get_existing_raw_ids(args.csv)

                if status == "done" and listing_id and listing_id in raw_done_ids:
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
                print(f"ID: {listing_id}")
                print(f"URL: {url}")
                print("=" * 100)


                links_df = mark_only_one_stop(links_df, index)
                links_df.at[index, "status"] = "processing"
                links_df.at[index, "error"] = ""
                save_links_df(links_df, args.links_csv)


                remove_raw_record(args.csv, listing_id=listing_id, url=url)

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=90000)
                    wait_manual_captcha_if_needed(page)

                    if is_avito_listing_closed(page):
                        print("Карточка закрыта на Авито. Пропускаю:", url)
                        links_df = mark_link_closed(
                            links_df=links_df,
                            index=index,
                            links_csv=args.links_csv,
                            message="listing closed: Объявление не посмотреть / Объявление закрыто",
                        )
                        closed_this_run += 1
                        continue

                    detail_row = parse_detail_page(page, url)

                    detail_row["Сегмент комнатности"] = link_row.get("segment")
                    detail_row["Чекбокс комнатности"] = link_row.get("checkbox_label")
                    detail_row["Значение чекбокса"] = link_row.get("checkbox_value")
                    detail_row["URL сегмента"] = link_row.get("segment_url")


                    if not detail_row.get("Название"):
                        detail_row["Название"] = link_row.get("title_from_listing")

                    if not detail_row.get("Цена"):
                        detail_row["Цена"] = link_row.get("price_from_listing")
                        detail_row["Цена числом"] = link_row.get("price_from_listing_num")

                    if not detail_row.get("Полный адрес"):
                        detail_row["Полный адрес"] = link_row.get("address_from_listing")

                    if not detail_row.get("Описание"):
                        detail_row["Описание"] = link_row.get("description_from_listing")

                    detail_row["source"] = "avito"
                    detail_row["source_listing_id"] = listing_id
                    detail_row["source_segment"] = link_row.get("segment")
                    detail_row["listing_page_number"] = link_row.get("page_number")
                    detail_row["listing_position_on_page"] = link_row.get("position_on_page")
                    detail_row["link_collected_at"] = link_row.get("collected_at")
                    detail_row["parsed_at"] = now_iso()

                    upsert_raw_row(args.csv, detail_row)

                    links_df.at[index, "status"] = "done"
                    links_df.at[index, "run_marker"] = ""
                    links_df.at[index, "parsed_at"] = now_iso()
                    links_df.at[index, "error"] = ""
                    save_links_df(links_df, args.links_csv)

                    processed_this_run += 1

                    print("Название:", detail_row.get("Название"))
                    print("Цена:", detail_row.get("Цена"))
                    print("Цена за м²:", detail_row.get("Цена за м²"))
                    print("Адрес:", detail_row.get("Полный адрес"))
                    print("Гео:", detail_row.get("Широта"), detail_row.get("Долгота"))
                    print(f"Готово: {listing_id}. За запуск обработано: {processed_this_run}")

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

                    links_df.at[index, "status"] = "error"
                    links_df.at[index, "run_marker"] = ""
                    links_df.at[index, "parsed_at"] = now_iso()
                    links_df.at[index, "error"] = error_text[:1000]
                    save_links_df(links_df, args.links_csv)

                time.sleep(random.uniform(args.card_delay_min, args.card_delay_max))

        finally:
            if args.keep_open:
                input("Браузер оставлен открытым. Нажмите Enter, чтобы закрыть...")

            context.close()

    if args.save_xlsx:
        export_xlsx_if_needed(args.csv)

    print("\nАнализ завершён.")
    print(f"За запуск обработано: {processed_this_run}")
    print(f"Закрытых объявлений пропущено: {closed_this_run}")
    print(f"Raw CSV: {args.csv}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Avito.ru — парсер: "
            "collect-links собирает ссылки, analyze парсит карточки с resume через STOP."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["collect-links", "analyze", "all"],
        default=None,
        help="collect-links = собрать ссылки; analyze = парсить карточки; all = сначала ссылки, потом карточки.",
    )

    parser.add_argument("--url", default=START_URL)
    parser.add_argument("--links-csv", default="avito_links.csv")
    parser.add_argument("--csv", default="avito_raw.csv")
    parser.add_argument("--profile-dir", default="avito_profile")

    parser.add_argument("--max-pages-per-group", type=int, default=20)
    parser.add_argument(
        "--max-segments",
        type=int,
        default=0,
        help="Лимит новых сегментов за запуск collect-links. 0 = all segments.",
    )
    parser.add_argument("--no-new-pages", type=int, default=2)
    parser.add_argument("--scrolls", type=int, default=10)

    parser.add_argument(
        "--max-links",
        type=int,
        default=0,
        help="Лимит новых ссылок за запуск collect-links. 0 = без лимита.",
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=0,
        help="Лимит карточек за запуск analyze. 0 = без лимита.",
    )

    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Начать analyze с конкретного индекса links.csv, если STOP не найден.",
    )

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--save-xlsx", action="store_true")

    parser.add_argument("--page-delay-min-ms", type=int, default=4000)
    parser.add_argument("--page-delay-max-ms", type=int, default=7000)
    parser.add_argument("--between-pages-delay-min", type=float, default=8.0)
    parser.add_argument("--between-pages-delay-max", type=float, default=15.0)
    parser.add_argument("--between-segments-delay-min", type=float, default=10.0)
    parser.add_argument("--between-segments-delay-max", type=float, default=20.0)
    parser.add_argument("--card-delay-min", type=float, default=6.0)
    parser.add_argument("--card-delay-max", type=float, default=12.0)

    args = parser.parse_args()

    if args.mode is None:
        print("\nВыбери режим:")
        print("1 — collect-links: собрать ссылки")
        print("2 — analyze: проанализировать собранные карточки")
        print("3 — all: сначала собрать ссылки, потом проанализировать")

        choice = input("Введите 1, 2 или 3: ").strip()

        if choice == "1":
            args.mode = "collect-links"
        elif choice == "2":
            args.mode = "analyze"
        elif choice == "3":
            args.mode = "all"
        else:
            raise SystemExit("Некорректный выбор режима.")

    print("\n" + "=" * 100)
    print("AVITO reliable parser")
    print("=" * 100)
    print(f"Mode: {args.mode}")
    print(f"Links CSV: {args.links_csv}")
    print(f"Raw CSV: {args.csv}")
    print("=" * 100)

    if args.mode == "collect-links":
        collect_links_mode(args)

    elif args.mode == "analyze":
        analyze_mode(args)

    elif args.mode == "all":
        collect_links_mode(args)
        analyze_mode(args)


if __name__ == "__main__":
    main()
