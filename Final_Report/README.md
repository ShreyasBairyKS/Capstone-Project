# VisionFood QAI — Final Thesis Source

This is a new final-report package; the earlier `Report/` submission is unchanged.

## Compile

Use a TeX distribution with Biber:

```text
pdflatex main
biber main
pdflatex main
pdflatex main
```

The report was revised after an evidence audit. It retains the audited six-class cap-segmentation validation record, the independently reproduced fill-classifier validation result, source-backed dashboard functionality, and Himalaya ONNX export-equivalence evidence. It intentionally excludes the untraceable four-class cap-classifier benchmark, latency claims, and unverified Himalaya detection-performance table. Himalaya remains a separate thesis/report module and is not used to support research-paper claims.

Before submission, replace the final `TBD` date/signature fields and insert any mandatory college cover-page wording requested by the department.
