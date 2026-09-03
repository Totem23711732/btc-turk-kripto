"""
BTCTurk'teki TÜM TL (TRY) karşılığı kripto paraları tarar. Üç ayrı sistem
birlikte çalışır:

1. STRATEJİ SİNYALİ (mevcut sistem): MA9/MA21 kesişimi + EMA200 trend +
   MACD + RSI + ADX + hacim onayının HEPSİNİN sağlandığı, sıkı filtreli
   AL/SAT sinyali.

2. ANORMAL HACİM ALARMI ("balina vekili"): Bir coin'in işlem hacmi kendi
   ortalamasının belirgin şekilde üzerine çıkarsa (büyük bir oyuncunun
   piyasaya girmiş/çıkmış olabileceğinin dolaylı bir işareti) bildirim
   gönderir. Gerçek cüzdan takibi DEĞİLDİR, sadece işlem hacmindeki
   anormalliği yakalar.

3. HABER TAKİBİ (filtreli): Her coin için Google Haberler'den yeni haber
   çıktığında, SADECE "büyük hareket" ile ilişkili anahtar kelimeler
   (listelenme, hack, yasaklama, ortaklık, rekor, çöküş, SEC/düzenleme vb.)
   içeren başlıkları bildirim olarak gönderir. Sıradan yorum/analiz
   haberleri otomatik elenir.

ÖNEMLİ: Bu üç sistem de bilgi sağlar, hiçbiri fiyatın ne yöne gideceğini
GARANTİ ETMEZ. Amaç, kararınızı vermeniz için daha fazla ve daha hızlı
bilgiye ulaşmanızı sağlamaktır.
"""
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

# ============================================================
# AYARLAR
# ============================================================

NTFY_TOPIC = "osman-btc-turk-kripto-efe"

# --- Mum / strateji ayarları ---
CANDLE_INTERVAL_MINUTES = 60
BARS_TO_FETCH = 260

MA_SHORT_PERIOD = 9
MA_LONG_PERIOD = 21
EMA_TREND_PERIOD = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
ADX_PERIOD = 14
ADX_THRESHOLD = 25
VOLUME_MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2

# --- Anormal hacim alarmı ("balina vekili") ---
WHALE_VOLUME_MULTIPLIER = 3.0   # Hacim, ortalamanın kaç katına çıkarsa alarm versin

# --- Haber takibi ---
NEWS_MAX_ITEMS = 5              # Her taramada en fazla kaç haber kontrol edilsin
NEWS_ENABLED = False           # Haberler kapalı - sadece strateji sinyali + hacim alarmı aktif

# Sadece bu kelimelerden en az birini içeren başlıklar "büyük hareket" ile
# ilişkili kabul edilip bildirim olarak gönderilir. Sıradan yorum/analiz
# haberleri bu listeye girmediği için otomatik elenir.
NEWS_KEY_TERMS = [
    "listelendi", "listeleniyor", "listeleme", "listing",
    "borsadan kaldır", "delist",
    "hack", "hacklendi", "çalındı", "siber saldırı",
    "yasak", "yasakla", "ban ",
    "sec ", "düzenleme", "regülasyon", "regulation",
    "iflas", "bankrupt", "çöktü", "çöküş", "crash",
    "ortaklık", "partnership", "anlaşma imzaladı",
    "rekor", "zirve", "ath", "tüm zamanların",
    "patladı", "fırladı", "yüzde yüz", "%100", "ikiye katla",
    "sert düşüş", "değer kaybetti",
    "dolandırıcılık", "scam", "vurgun",
    "etf", "onay", "sec onayı",
]

REQUEST_DELAY_SECONDS = 0.25

# ============================================================
# Buradan sonrasını değiştirmenize gerek yok
# ============================================================

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")


def get_try_pairs() -> list:
    resp = requests.get("https://api.btcturk.com/api/v2/ticker", timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success", False):
        raise RuntimeError(f"Parite listesi alınamadı: {payload}")
    pairs = [item["pair"] for item in payload["data"] if item["pair"].endswith("TRY")]
    return sorted(set(pairs))


def get_ohlc_history(pair_symbol: str, bars: int = BARS_TO_FETCH) -> list:
    now = int(time.time())
    span_seconds = CANDLE_INTERVAL_MINUTES * 60 * (bars + 5)
    start = now - span_seconds

    resp = requests.get(
        "https://graph-api.btcturk.com/v1/klines/history",
        params={
            "symbol": pair_symbol,
            "resolution": CANDLE_INTERVAL_MINUTES,
            "from": start,
            "to": now,
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("s") != "ok":
        raise RuntimeError(f"OHLC verisi alınamadı ({pair_symbol}): {payload}")

    candles = []
    for t, o, h, l, c, v in zip(
        payload["t"], payload["o"], payload["h"], payload["l"], payload["c"], payload["v"]
    ):
        candles.append({"time": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return candles[-bars:]


def _rsi(closes: pd.Series, period: int) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


def _macd(closes: pd.Series):
    ema_fast = closes.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = closes.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    return macd_line, signal_line


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, 1e-12)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-12)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx


def compute_strategy_signal(df: pd.DataFrame) -> dict:
    min_needed = max(EMA_TREND_PERIOD + 20, MACD_SLOW + MACD_SIGNAL, ADX_PERIOD * 3, VOLUME_MA_PERIOD) + 2
    if len(df) < min_needed:
        return {"signal": "HOLD", "price": float(df["close"].iloc[-1]) if len(df) else None, "detail": "Yeterli veri yok"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_up = prev["ma_short"] <= prev["ma_long"] and last["ma_short"] > last["ma_long"]
    crossed_down = prev["ma_short"] >= prev["ma_long"] and last["ma_short"] < last["ma_long"]

    if not (crossed_up or crossed_down):
        return {"signal": "HOLD", "price": float(last["close"]), "detail": "Kesişim yok"}

    direction = "BUY" if crossed_up else "SELL"

    checks = {}
    if direction == "BUY":
        checks["trend"] = last["close"] > last["ema_trend"]
        checks["macd"] = last["macd_line"] > last["macd_signal"]
        checks["rsi"] = 50 < last["rsi"] < RSI_OVERBOUGHT
        checks["adx"] = last["adx"] > ADX_THRESHOLD
        checks["volume"] = last["volume"] > last["volume_ma"] * VOLUME_MULTIPLIER
    else:
        checks["trend"] = last["close"] < last["ema_trend"]
        checks["macd"] = last["macd_line"] < last["macd_signal"]
        checks["rsi"] = RSI_OVERSOLD < last["rsi"] < 50
        checks["adx"] = last["adx"] > ADX_THRESHOLD
        checks["volume"] = last["volume"] > last["volume_ma"] * VOLUME_MULTIPLIER

    passed = sum(checks.values())
    total = len(checks)
    detail_parts = [f"{name}={'✓' if ok else '✗'}" for name, ok in checks.items()]
    detail = f"Kesişim + {passed}/{total} onay ({', '.join(detail_parts)}) RSI={last['rsi']:.1f} ADX={last['adx']:.1f}"

    if passed == total:
        return {"signal": direction, "price": float(last["close"]), "detail": detail}

    return {"signal": "HOLD", "price": float(last["close"]), "detail": f"Kesişim var ama filtre geçmedi: {detail}"}


def check_volume_spike(df: pd.DataFrame):
    """Hacim, kendi ortalamasının WHALE_VOLUME_MULTIPLIER katından fazlaysa anormal kabul edilir."""
    if len(df) < VOLUME_MA_PERIOD + 2:
        return False, None, None
    last_vol = df["volume"].iloc[-1]
    last_vol_ma = df["volume_ma"].iloc[-1]
    last_price = float(df["close"].iloc[-1])
    if pd.isna(last_vol_ma) or last_vol_ma <= 0:
        return False, None, last_price
    ratio = last_vol / last_vol_ma
    return ratio >= WHALE_VOLUME_MULTIPLIER, ratio, last_price


def check_news(pair: str, pair_state: dict) -> list:
    """
    Google Haberler'den o coin ile ilgili son haberleri çeker.
    Sadece DAHA ÖNCE görülmemiş haberleri döner. İlk çalıştırmada
    (referans yokken) bildirim spam'i olmasın diye boş liste döner.
    """
    base_symbol = pair[:-3] if pair.endswith("TRY") else pair
    query = f"{base_symbol} kripto"
    url = f"https://news.google.com/rss/search?q={quote(query)}&hl=tr&gl=TR&ceid=TR:tr"

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    items = root.findall(".//item")[:NEWS_MAX_ITEMS]

    is_first_run = "seen_news_links" not in pair_state
    seen_links = set(pair_state.get("seen_news_links", []))

    current_links = []
    new_articles = []
    for item in items:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not link:
            continue
        current_links.append(link)
        if not is_first_run and link not in seen_links:
            title_lower = title.lower()
            is_relevant = any(term in title_lower for term in NEWS_KEY_TERMS)
            if is_relevant:
                new_articles.append((title, link))

    pair_state["seen_news_links"] = current_links
    return new_articles


def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(state: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def send_notification(title: str, message: str, priority: str = "default"):
    if "buraya-kendi" in NTFY_TOPIC:
        print("⚠️ UYARI: NTFY_TOPIC hâlâ varsayılan değerde. check_signal.py içinde değiştirin!")
        return
    requests.post(
        f"https://ntfy.sh/{NTFY_TOPIC}",
        data=message.encode("utf-8"),
        headers={"Title": title.encode("utf-8"), "Priority": priority},
        timeout=15,
    )


def main():
    state = load_state()

    try:
        pairs = get_try_pairs()
    except Exception as exc:
        print(f"Parite listesi alınamadı (sonraki denemede tekrar denenecek): {exc}")
        sys.exit(0)

    print(f"{len(pairs)} parite (TL karşılığı coin) taranacak.")

    for pair in pairs:
        pair_state = state.get(pair, {})

        # --- Fiyat verisini bir kere çek, hem strateji hem hacim alarmı için kullan ---
        try:
            candles = get_ohlc_history(pair)
            df = pd.DataFrame(candles)
            df["ma_short"] = df["close"].rolling(window=MA_SHORT_PERIOD).mean()
            df["ma_long"] = df["close"].rolling(window=MA_LONG_PERIOD).mean()
            df["ema_trend"] = df["close"].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
            df["rsi"] = _rsi(df["close"], RSI_PERIOD)
            df["macd_line"], df["macd_signal"] = _macd(df["close"])
            df["adx"] = _adx(df, ADX_PERIOD)
            df["volume_ma"] = df["volume"].rolling(window=VOLUME_MA_PERIOD).mean()
        except Exception as exc:
            print(f"{pair}: fiyat verisi alınamadı, atlanıyor -> {exc}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        # --- 1) STRATEJİ SİNYALİ ---
        result = compute_strategy_signal(df)
        print(f"{pair}: {result['signal']} | {result['detail']}")

        if result["signal"] != "HOLD" and pair_state.get("last_signal") != result["signal"]:
            action_tr = "AL" if result["signal"] == "BUY" else "SAT"
            send_notification(
                title=f"🔔 {action_tr} sinyali (sıkı filtre) — {pair}",
                message=(
                    f"Fiyat: {result['price']:,.2f} TL\n{result['detail']}\n\n"
                    f"BTCTurk uygulamasını açıp işlemi kendiniz yapabilirsiniz."
                ),
                priority="high",
            )
            print(f"  ✅ Strateji bildirimi gönderildi ({pair}).")
            pair_state["last_signal"] = result["signal"]
        elif result["signal"] != "HOLD":
            print(f"  (Aynı strateji sinyali zaten gönderilmişti: {pair})")

        # --- 2) ANORMAL HACİM ALARMI ("balina vekili") ---
        is_spike, ratio, price = check_volume_spike(df)
        if is_spike and not pair_state.get("volume_alert_active", False):
            send_notification(
                title=f"🐋 Anormal hacim artışı — {pair}",
                message=(
                    f"İşlem hacmi ortalamanın {ratio:.1f} katına çıktı "
                    f"(olası büyük oyuncu hareketi). Fiyat: {price:,.2f} TL.\n\n"
                    f"Bu kesin bir sinyal değildir, sadece dikkat çekici bir "
                    f"anormalliktir."
                ),
                priority="high",
            )
            print(f"  🐋 Hacim alarmı gönderildi ({pair}, oran={ratio:.1f}).")
            pair_state["volume_alert_active"] = True
        elif not is_spike:
            pair_state["volume_alert_active"] = False

        # --- 3) HABER TAKİBİ ---
        if NEWS_ENABLED:
            try:
                new_articles = check_news(pair, pair_state)
                for title, link in new_articles:
                    send_notification(
                        title=f"📰 Yeni haber — {pair}",
                        message=f"{title}\n{link}",
                        priority="default",
                    )
                    print(f"  📰 Haber bildirimi gönderildi ({pair}): {title[:60]}")
            except Exception as exc:
                print(f"{pair}: haber kontrolü başarısız, atlanıyor -> {exc}")

        pair_state["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        state[pair] = pair_state
        time.sleep(REQUEST_DELAY_SECONDS)

    save_state(state)
    print("Tarama tamamlandı.")


if __name__ == "__main__":
    main()
