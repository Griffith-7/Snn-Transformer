# Examples

Two styles, matching the plugin-vs-framework split of the package:

- `plugin_block.py` — **plugin usage**: import `AstrocyteHebbianAttention` /
  `AstrocyteHebbianBlock` and wire them into your own model. This is the
  intended way to use the package.
- `wrappers.py` — **example wrappers**: full models (`AstrocyteHebbianClassifier`,
  `CausalAstrocyteLanguageModel`) assembled from the plugin components. Useful
  starting points, not a framework you build inside.

Run either directly:

```bash
python examples/plugin_block.py
python examples/wrappers.py
```