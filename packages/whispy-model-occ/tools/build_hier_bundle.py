"""Assemble models/hier_late.joblib from the frozen revision-3 ICC
artifacts (radar/E2/outputs/frozen_v3 + radar/E1/outputs/pca_deep).

The bundle carries everything OccHierModel needs at predict time:
  views    trained {model, scaler, fill, columns} dicts, 5 RF seeds for
           CSI/radar, 5 GBM seeds for early fusion
  hierarchy  level-2/level-3 scaler+PCA pairs per modality
  level1   (sc, pca) level-1 bases fitted on training seconds
  thresholds per-view validation-tuned cutoffs (selection.json)

Weights were trained on the balanced train split and every threshold
was tuned on the validation split by the frozen_v3 pipeline — the
bundle only repackages them, nothing is refit here.

Run:  python tools/build_hier_bundle.py [radar_root]
"""
import json
import sys
from pathlib import Path

import joblib

RADAR = Path(sys.argv[1] if len(sys.argv) > 1
             else r"C:\Users\ggad\Desktop\radar")
FV3 = RADAR / "E2" / "outputs" / "frozen_v3"
PCAS = RADAR / "E1" / "outputs" / "pca_deep" / "pcas_deep.joblib"
OUT = (Path(__file__).resolve().parent.parent
       / "whispy_model_occ" / "models" / "hier_late.joblib")


def main():
    hierarchy = joblib.load(FV3 / "hierarchy.joblib")
    deep = joblib.load(PCAS)
    basis = deep["train"] if isinstance(deep, dict) else deep
    sel = json.loads((FV3 / "selection.json").read_text())
    cand = {r["key"]: r for r in sel["all_validation_candidates"]}

    def objs(name, view):
        return [joblib.load(
            FV3 / "models" / f"{name}__{view.replace(' ', '_')}__s{s}.joblib")
            for s in range(5)]

    def thr(key):
        seeds = cand[key]["seeds"]
        ts = [float(s["threshold"]) for s in seeds]
        return {"per_seed": ts, "mean": sum(ts) / len(ts)}

    bundle = {
        "kind": "occ-hier",
        "window_s": 15,
        "views": {
            "csi": objs("RF", "CSI"),
            "radar": objs("RF", "radar"),
            "early": objs("GBM", "early fusion"),
        },
        "hierarchy": {"csi": hierarchy["csi"], "rad": hierarchy["rad"]},
        "level1": {
            "csi": (basis["sc_c"], basis["pca_c"]),
            "rad": (basis["sc_r"], basis["pca_r"]),
        },
        "thresholds": {
            "late": thr("RF|late probability fusion"),
            "csi": thr("RF|CSI"),
            "radar": thr("RF|radar"),
            "early": thr("GBM|early fusion"),
        },
        "provenance": {
            "source": "radar/E2 frozen_v3 (icc_multimodal2)",
            "train": "train_minutes/train (balanced, augmented)",
            "validation": "validation_minutes (thresholds, balanced acc)",
            "test_reference": {
                "late_fusion_rf": 0.975, "radar_rf": 0.969,
                "early_fusion_gbm": 0.808, "csi_rf": 0.687,
            },
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, OUT, compress=3)
    print("wrote", OUT, f"({OUT.stat().st_size/1e6:.1f} MB)")
    print("thresholds:", {k: round(v['mean'], 3)
                          for k, v in bundle['thresholds'].items()})


if __name__ == "__main__":
    main()
