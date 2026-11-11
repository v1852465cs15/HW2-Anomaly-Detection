# HW2 - Phishing Website Anomaly Detection (G01)

The repository root corresponds to the HW2 submission directory.

| Path | Contents |
|---|---|
| requirements.txt | Dependencies for the supplied Python analysis source |
| configs/ | Configuration documentation |
| data/ | Previously published dataset/source workbooks and split notes |
| src/ | Published Part C source archive; integration notes for A/B/C |
| results/ | Previously published feature rankings and association/redundancy workbooks |
| figures/ | Report figures |
| report/ | ACL source, bibliography, style files and final PDF |

## Report

[Final PDF](report/HW2_G01_report.pdf): six body pages and references on page 7.
[Latest source](report/HW2_Integrated_v10.tex); the default equivalent build target is report/HW2_Integrated.tex.

Compile with XeLaTeX, BibTeX, then XeLaTeX twice from report/. Figure references resolve through ../figures/.

## Available data and source

The original workbooks are in data/source/. The shared split has 7,738 training, 1,658 validation and 1,659 test records. Result=1 denotes legitimate and Result=-1 phishing. The benchmark has 30 categorical features.

Part C is preserved in [src/PartC.zip](src/PartC.zip), containing error_analysis.py, interpretability.py and plot_ablation.py.

Additional local A/B integration scripts, CSV splits, model predictions and analysis artifacts are prepared separately but are not published by this directory-only change. The complete runnable A/B/C pipeline requires those inputs. No model training was rerun.

## Install dependencies

Use Python 3.11 or 3.12 in a virtual environment:

```bash
python -m pip install -r requirements.txt
```
