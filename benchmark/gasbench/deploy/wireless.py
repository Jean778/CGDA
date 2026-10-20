"""Wireless-link latency and energy for the system-level deployment view.

Two deployment modes are modelled for an inspection-robot gas node:

* edge mode    : inference on-device, transmit only the label (1 byte + header)
* cloud mode   : transmit the raw 128-feature vector (fp32 512 B / int8 128 B)
                 and infer off-node

Link parameters follow the actual inspection-robot radio plan:

* Wi-Fi  802.11n 2.4 GHz   -- primary data/video backhaul; effective
  throughput ~20 Mbps, TX current ~135 mA @ 3.3 V (ESP32-class measured)
* LoRa  433 MHz industrial -- long-range telemetry/backup (SF9 BW125,
  9.6 kbps, 20 dBm ~ 130 mA @ 3.3 V)
* NB-IoT (BC95-class)      -- wide-area backup uplink, ~25 kbps, peak TX
  ~200 mA @ 3.3 V
* BLE 5.0 (1 M PHY)        -- on-site commissioning / nearby node link,
  TX ~30 mW

Airtime uses the raw throughput plus a fixed protocol overhead per packet.
"""
from __future__ import annotations

LINKS = {
    "ble":    {"throughput_bps": 1.0e6,  "tx_mw": 30.0,   "overhead_s": 0.002},
    "wifi":   {"throughput_bps": 20.0e6, "tx_mw": 450.0,  "overhead_s": 0.005},
    "lora":   {"throughput_bps": 9.6e3,  "tx_mw": 430.0,  "overhead_s": 0.120},  # SF9 BW125 + preamble
    "nb_iot": {"throughput_bps": 25.0e3, "tx_mw": 660.0,  "overhead_s": 0.060},  # incl. RRC setup
}


def transmit(payload_bytes: int, link: str) -> dict:
    l = LINKS[link]
    airtime = payload_bytes * 8 / l["throughput_bps"]
    total_s = airtime + l["overhead_s"]
    energy_mj = l["tx_mw"] * total_s / 1e3
    return {"link": link, "payload_bytes": payload_bytes, "airtime_s": airtime,
            "total_s": total_s, "energy_mj": energy_mj}


def system_latency_energy(inference_ms: float, inference_mj: float,
                          payload_bytes: int, link: str) -> dict:
    """End-to-end per-inference latency/energy including the wireless hop."""
    t = transmit(payload_bytes, link)
    return {
        "inference_ms": inference_ms,
        "wireless_s": t["total_s"],
        "system_latency_ms": inference_ms + t["total_s"] * 1e3,
        "inference_mj": inference_mj,
        "wireless_mj": t["energy_mj"],
        "system_energy_mj": inference_mj + t["energy_mj"],
        **{k: v for k, v in t.items() if k != "link"},
    }
