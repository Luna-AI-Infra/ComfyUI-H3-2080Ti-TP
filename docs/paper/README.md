# Bilingual experiment paper

- [English PDF](H3-TP-Experiments.en.pdf) / [editable Markdown](paper.en.md)
- [中文PDF](H3-TP-Experiments.zh-CN.pdf) / [可编辑Markdown](paper.zh-CN.md)

Both versions contain the same eight-section engineering record and the same measurement values. Figures were redrawn as single-question charts with explicit scopes; they are generated from `measurements.json`, not estimated from screenshots. Selected historical reports are linked from `docs/RESULTS.md`.

The PDFs are engineering reports, not peer-reviewed papers or a sustained-serving certification. Read the cache-state labels, numerical caveats, unsuccessful candidates and limitations together with the speed tables.

To rebuild in a separate documentation environment:

```bash
python -m pip install -r requirements-docs.txt
python scripts/plot_paper_figures.py
python scripts/build_papers.py --cjk-font FONT_FILE --cjk-bold BOLD_FONT_FILE
```

Supply TrueType/TTC fonts containing Simplified Chinese that permit PDF embedding. Font files are not included in this repository. Render and review every resulting page after changing content; source-text extraction alone is not a layout check. No PDF build is required for ComfyUI inference.
