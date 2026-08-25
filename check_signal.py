"""
BTCTurk'teki TÜM TL (TRY) karşılığı kripto paraları tarar. MA kesişimi +
RSI stratejisine göre herhangi birinde AL/SAT sinyali oluşursa, telefonunuza
(ntfy uygulaması üzerinden) ücretsiz bildirim gönderir. Emri siz BTCTurk
uygulamasından kendiniz verirsiniz.

Bu dosyayı elle çalıştırmanıza gerek yok — GitHub, .github/workflows/check.yml
sayesinde bunu sizin için otomatik, düzenli aralıklarla çalıştırır.
"""
import json
import os
import sys
import time

import pandas as pd
import requests

# ============================================================
# AYARLAR — bunları GitHub üzerinden (dosyayı düzenleyerek) değiştirebilirsiniz
# ============================================================

# 👇 BUNU MUTLAKA KENDİ SEÇTİĞİNİZ İSİMLE DEĞİŞTİRİN (ntfy uygulamasında
# abone olduğunuz konu adı). Daha önce girdiyseniz, onu buraya tekrar yazın.
NTFY_TOPIC = "osman-btc-turk-kripto-efe"

CANDLE_INTERVAL_MINUTES = 60        # Mum periyodu (60 = saatlik mumlar)
MA_SHORT_PERIOD = 9                 # Kısa hareketli ortalama periyodu
MA_LONG_PERIOD = 21                 # Uzun hareketli ortalama periyodu
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

REQUEST_DELAY_SECONDS = 0.25        # BTCTurk'ü yormamak için istekler arası bekleme

# ============================================================
# Buradan sonrasını değiştirmenize gerek yok
# ============================================================

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")


def get_try_pairs() -> list:
    """BTCTurk'te işlem gören TÜM TL (TRY) karşılığı pariteleri getirir."""
    resp = requests.get("https://api.btcturk.com/api/v2/ticker", timeout=15)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success", False):
        raise RuntimeError(f"Parite listesi alınamadı: {payload}")
    pairs = [item["pair"] for item in payload["data"] if item["pair"].endswith("TRY")]
    return sorted(set(pairs))


def get_ohlc_history(pair_symbol: str, bars: int = 100) -> list:
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


def compute_signal(candles: list) -> dict:
    min_needed = max(MA_LONG_PERIOD, RSI_PERIOD) + 2
    if len(candles) < min_needed:
        return {
            "signal": "HOLD",
            "price": candles[-1]["close"] if candles else None,
            "detail": "Yeterli veri yok",
        }

    df = pd.DataFrame(candles)
    df["ma_short"] = df["close"].rolling(window=MA_SHORT_PERIOD).mean()
    df["ma_long"] = df["close"].rolling(window=MA_LONG_PERIOD).mean()
    df["rsi"] = _rsi(df["close"], RSI_PERIOD)

    last = df.iloc[-1]
    prev = df.iloc[-2]

    crossed_up = prev["ma_short"] <= prev["ma_long"] and last["ma_short"] > last["ma_long"]
    crossed_down = prev["ma_short"] >= prev["ma_long"] and last["ma_short"] < last["ma_long"]

    if crossed_up and last["rsi"] < RSI_OVERBOUGHT:
        return {
            "signal": "BUY",
            "price": float(last["close"]),
            "detail": f"MA{MA_SHORT_PERIOD}, MA{MA_LONG_PERIOD}'i yukarı kesti. RSI={last['rsi']:.1f}",
        }
    if crossed_down and last["rsi"] > RSI_OVERSOLD:
        return {
            "signal": "SELL",
            "price": float(last["close"]),
            "detail": f"MA{MA_SHORT_PERIOD}, MA{MA_LONG_PERIOD}'i aşağı kesti. RSI={last['rsi']:.1f}",
        }
    return {
        "signal": "HOLD",
        "price": float(last["close"]),
        "detail": f"Kesişim yok. RSI={last['rsi']:.1f}",
    }


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
        headers={
            "Title": title.encode("utf-8"),
            "Priority": priority,
        },
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
        try:
            candles = get_ohlc_history(pair)
            result = compute_signal(candles)
        except Exception as exc:
            print(f"{pair}: hata, atlanıyor -> {exc}")
            time.sleep(REQUEST_DELAY_SECONDS)
            continue

        pair_state = state.get(pair, {})
        print(f"{pair}: {result['signal']} | {result['detail']}")

        if result["signal"] != "HOLD" and pair_state.get("last_signal") != result["signal"]:
            action_tr = "AL" if result["signal"] == "BUY" else "SAT"
            send_notification(
                title=f"🔔 {action_tr} sinyali — {pair}",
                message=(
                    f"Fiyat: {result['price']:,.2f} TL\n{result['detail']}\n\n"
                    f"BTCTurk uygulamasını açıp işlemi kendiniz yapabilirsiniz."
                ),
                priority="high",
            )
            print(f"  ✅ Bildirim gönderildi ({pair}).")
            pair_state["last_signal"] = result["signal"]
        elif result["signal"] != "HOLD":
            print(f"  (Aynı sinyal zaten gönderilmişti, tekrar gönderilmiyor: {pair})")

        pair_state["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        state[pair] = pair_state

        time.sleep(REQUEST_DELAY_SECONDS)

    save_state(state)
    print("Tarama tamamlandı.")


if __name__ == "__main__":
    main()
