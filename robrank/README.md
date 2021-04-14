### Naming conventions for these models

```
<task><backbone><modifier>
```

* `task` in {c, r, h} for classification, ranking, hybrid
* `backbone` e.g. lenet, res18, mnas10
* `moddifier` in {, d, e} for vanilla, advrank:defense, experimental defense

This rules applies to `configs` and `models`.
