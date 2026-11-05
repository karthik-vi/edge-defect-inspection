"""README figure: operator heatmap overlays (one defect per category) from the deployed model."""
import cv2
import numpy as np

from edgeinspect.config import load_settings
from edgeinspect.data import list_test
from edgeinspect.engine import Engine, overlay_png

cfg = load_settings()
rows = []
for cat in cfg.categories:
    eng = Engine.load(cat, cfg)
    defects = [t for t in list_test(cfg.data_root, cat) if t[1] == 1]
    pick = [defects[i] for i in (0, len(defects) // 2, len(defects) - 1)]
    tiles = []
    for p, _, kind in pick:
        img = cv2.imread(str(p))
        v = eng.inspect(img)
        ov = cv2.imdecode(np.frombuffer(overlay_png(img, v.anomaly_map, v.threshold), np.uint8), 1)
        tile = np.hstack([cv2.resize(img, (200, 200)), cv2.resize(ov, (200, 200))])
        label = f"{cat}/{kind} s={v.score:.2f} thr={v.threshold:.2f} {'REJECT' if v.reject else 'ACCEPT'}"
        cv2.putText(tile, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    rows.append(np.hstack(tiles))
cv2.imwrite("results/plots/heatmaps.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
