# Dataset Loader

Utilities for loading, filtering, transforming, and representing touch gesture datasets.

The repository provides a `Dataset` abstraction for gesture data consisting of one or more strokes and optional class or condition information.

## Installation

```python
pip install -r requirements.txt
```

## Dataset Format

Datasets are stored as a JSON array. Each entry contains a gesture followed by
one or more class or condition values:

```text
[gesture, class_0, class_1, ...]
```

The class or condition values are optional in number, but the gesture must
always be the first element of an entry.

### Single-stroke gestures

A single-stroke gesture can be stored directly as a list of samples:

```json
[
  [
    [
      [0.0, 0.0, 0.00],
      [1.2, 0.5, 0.01],
      [2.4, 1.1, 0.02]
    ],
    4
  ],
  [
    [
      [5.0, 3.0, 0.00],
      [5.8, 3.7, 0.01],
      [6.6, 4.2, 0.02]
    ],
    7
  ]
]
```

In this example, each dataset entry contains:

```text
[gesture, class_0]
```

The first gesture belongs to class `4` and the second gesture belongs to class
`7`.

### Multi-stroke gestures

For multi-stroke gestures, the gesture is represented as a list of strokes:

```json
[
  [
    [
      [
        [0.0, 0.0, 0.00],
        [1.0, 0.5, 0.01],
        [2.0, 1.0, 0.02]
      ],
      [
        [3.0, 2.0, 0.10],
        [4.0, 2.5, 0.11],
        [5.0, 3.0, 0.12]
      ]
    ],
    4,
    2
  ]
]
```

This entry contains two strokes and two class or condition values:

```text
[gesture, class_0, class_1]
```

Here, `class_0` is `4` and `class_1` is `2`.

### Sample representation

Samples can be represented either as positions:

```text
[x, y]
```

or as timestamped positions:

```text
[x, y, t]
```

All gestures within one dataset should use the same sample representation.

When a single-stroke gesture is loaded, the loader automatically wraps it as a
one-stroke gesture. Therefore, gestures returned by `Dataset` always follow the
same hierarchy:

```text
gesture
└── stroke
    └── sample
        └── [x, y] or [x, y, t]
```

## Loading a Dataset

```python
from src.dataset_loader import Dataset

dataset = Dataset.from_json("data.json")
```

Timestamped data is preserved by default. Timestamps can be removed while
loading using:

```python
dataset = Dataset.from_json("data.json", drop_timestamps=True)
```

Gestures can also be filtered by their number of samples or strokes during
loading:

```python
dataset = Dataset.from_json(
    "data.json",
    min_size=10,
    max_size=500,
    min_strokes=1,
    max_strokes=3,
)
```

`min_size` and `max_size` are applied to every stroke of a gesture.

## Accessing the Data

The number of loaded gestures can be obtained using:

```python
len(dataset)
```

Gestures are accessed using:

```python
gestures = dataset.get_gestures()
```

For example:

```python
gesture = gestures[0]
stroke = gesture[0]
sample = stroke[0]
```

For timestamped data, a sample contains:

```python
x, y, t = sample
```

The loader records whether timestamps are available:

```python
dataset.has_timestamps
```

### Classes and conditions

Class and condition values are stored by dimension.

For an input entry of the form:

```text
[gesture, class_0, class_1]
```

the first class dimension can be accessed using:

```python
classes = dataset.get_class(0)
```

and the second using:

```python
classes = dataset.get_class(1)
```

All class or condition values can be accessed using:

```python
conditions = dataset.get_conditions()
```

The number of class or condition dimensions is available through:

```python
dataset.num_classes()
```

A dataset can be filtered by a value in a particular class dimension:

```python
filtered = dataset.filter_by_class(0, 4)
```

This returns a new `Dataset` containing only samples for which class dimension
`0` has the value `4`.

Multiple categorical filters can be applied using:

```python
filtered = dataset.filter(
    class_filters={
        0: [4, 5],
        1: [1, 2],
    }
)
```

## License

This project is licensed under the GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`). See `LICENSE` for details.