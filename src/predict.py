"""
predict.py — Run inference on raw Titanic data using the saved pipeline.
The pipeline handles preprocessing automatically — pass raw feature values.
"""

import joblib
import pandas as pd
from pathlib import Path


# ── Load pipeline ─────────────────────────────────────────────────────────────

pipeline_path = Path("models/pipeline.pkl")

if not pipeline_path.exists():
    raise FileNotFoundError(
        f"No model found at {pipeline_path}. Run train.py first."
    )

pipeline = joblib.load(pipeline_path)
print(f"Model loaded: {type(pipeline.named_steps['model']).__name__}\n")


# ── Sample input — raw Titanic features (no preprocessing needed) ─────────────

sample_data = pd.DataFrame({
    "Pclass":   [3,       1,       2      ],
    "Age":      [22.0,    38.0,    26.0   ],
    "SibSp":    [1,       1,       0      ],
    "Parch":    [0,       0,       0      ],
    "Fare":     [7.25,    71.28,   13.00  ],
    "Sex":      ["male",  "female","female"],
    "Embarked": ["S",     "C",     "S"    ],
})

print("Input Data:")
print(sample_data.to_string(index=False))
print()

# ── Predict ───────────────────────────────────────────────────────────────────

predictions = pipeline.predict(sample_data)

label_map = {0: "Did not survive", 1: "Survived"}

print("Predictions:")
for i, pred in enumerate(predictions):
    label = label_map.get(int(pred), str(pred))
    print(f"  Passenger {i + 1}: {label}  (raw={pred})")
