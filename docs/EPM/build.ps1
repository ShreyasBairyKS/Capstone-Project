# Regenerates all calculated tables, then compiles main.pdf.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
python tools/build_tables.py
if ($LASTEXITCODE -ne 0) { throw "A consistency check failed" }
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
if ($LASTEXITCODE -ne 0) { throw "LaTeX build failed; see main.log" }
