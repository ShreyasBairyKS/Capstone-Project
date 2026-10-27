# Research Paper Source — Final Scope

This paper was revised after an evidence audit of the on-disk datasets, SegX workspace, saved ONNX artefacts, dashboard source, and prior review response. It reports only the audited six-class cap-segmentation dataset and recorded validation run, a reproduced three-class fill-classifier validation result (135/145; 93.10%), and source-backed dashboard/API functionality.

It intentionally excludes unverified four-class cap-classifier benchmarks, latency figures, end-to-end/factory metrics, and Himalaya label-detection results. Segmentation remains explicitly labelled as validation-only evidence.

Compile from this directory with:

```text
pdflatex paper
biber paper
pdflatex paper
pdflatex paper
```

The bibliography is shared from `../Final_Report/references.bib` to keep citations consistent.
