"""
BTCTurk piyasasını kontrol eder, MA kesişimi + RSI stratejisine göre
AL/SAT sinyali oluşursa telefonunuza (ntfy uygulaması üzerinden) ücretsiz
bildirim gönderir. Emri siz BTCTurk uygulamasından kendiniz verirsiniz.

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

# 👇 BUNU MUTLAKA DEĞİŞTİRİN: ntfy uygulamasında seçtiğiniz "konu" (topic) adı.
# Uzun ve tahmin edilmesi zor bir isim seçin (örn. isminiz + rastgele sayılar),
# çünkü bu isim aynı zamanda bildirimlerinize kimin ulaşabileceğini belirliyor.
NTFY_TOPIC = "buraya-kendi-gizli-konu-adinizi-yazin-8291"

PAIR_SYMBOL = "BTCTRY"              # Hangi parite izlensin (örn. ETHTRY de olabilir)
CANDLE_INTERVAL_MINUTES = 60        # Mum periyodu (60 = saatlik mumlar)
MA_SHORT_PERIOD = 9                 # Kısa hareketli ortalama periyodu
MA_LONG_PERIOD = 21                 # Uzun hareketli ortalama periyodu
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# ============================================================
# Buradan sonrasını değiştirmenize gerek yok
# ============================================================

STATE_FILE = os.path.join(os.path.dirname(__file__), "state.json")


def get_ohlc_history(bars: int = 100) -> list:
    graph_symbol = PAIR_SYMBOL[:-3] + "_" + PAIR_SYMBOL[-3:]  # BTCTRY -> BTC_TRY
    now = int(time.time())
    span_seconds = CANDLE_INTERVAL_MINUTES * 60 * (bars + 5)
    start = now - span_seconds

    resp = requests.get(
        "https://graph-api.btcturk.com/v1/klines/history",
        params={
            "symbol": graph_symbol,
            "resolution": CANDLE_INTERVAL_MINUTES,
            "from": start,
            "to": now,
        },
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("s") != "ok":
        raise RuntimeError(f"OHLC verisi alınamadı: {payload}")

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
        return {"last_signal": None}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_signal": None}


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
        candles = get_ohlc_history()
        result = compute_signal(candles)
    except Exception as exc:
        # Geçici bir ağ/veri hatasında workflow'u kırmadan sessizce çık,
        # bir sonraki çalıştırmada tekrar denenecek.
        print(f"Hata (yoksayıldı, sonraki denemede tekrar denenecek): {exc}")
        sys.exit(0)

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sinyal: {result['signal']} | {result['detail']}")

    state["last_checked"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    if result["signal"] != "HOLD" and state.get("last_signal") != result["signal"]:
        action_tr = "AL" if result["signal"] == "BUY" else "SAT"
        send_notification(
            title=f"🔔 {action_tr} sinyali — {PAIR_SYMBOL}",
            message=f"Fiyat: {result['price']:,.2f} TL\n{result['detail']}\n\nBTCTurk uygulamasını açıp işlemi kendiniz yapabilirsiniz.",
            priority="high",
        )
        print("✅ Bildirim gönderildi.")
        state["last_signal"] = result["signal"]
    elif result["signal"] != "HOLD":
        print("Aynı sinyal zaten gönderilmişti, tekrar gönderilmiyor.")

    save_state(state)


if __name__ == "__main__":
    main()
