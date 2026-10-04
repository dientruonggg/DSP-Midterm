# DSP Midterm – Speech/Silence Segmentation

This branch contains notebook-only implementations for the three assigned VAD tasks. Each notebook is self-contained, executes all four test recordings in one **Run All**, displays four result figures inline, and saves the same figures with `metrics.json` in its own folder.

## Structure

```text
src/
├── TT1/
│   ├── 1.ipynb       # teacher baseline: global STE threshold
│   ├── 2.ipynb       # DSP extension: voiced/unvoiced features
│   └── output/
├── TT2/
│   ├── 1.ipynb       # teacher baseline: 1-D histogram
│   ├── 2.ipynb       # DSP extension: H[STE, ZCR]
│   └── output/
└── TT3/
    ├── 1.ipynb       # teacher baseline: Gaussian STE
    ├── 2.ipynb       # DSP extension: multivariate Gaussian
    └── output/
```

`1.ipynb` is the teacher-aligned baseline. `2.ipynb` is the additional DSP experiment. The test set is never used to select parameters.

## Run

```powershell
uv sync
uv run jupyter lab
```

Open one notebook and choose **Restart Kernel and Run All Cells**. No separate Python source file is required.

## Submission outputs

- Presentation slides (Canva): [Canva Slide VAD Midterm](https://canva.link/jhh1gex7ybd87yk)
- English report source: `report/VAD_MIDTERM_REPORT.md`
- Vietnamese explanation source: `report/VAD_MIDTERM_REPORT_VI.md`
- Vietnamese defense notes: `report/DEFENSE_NOTES_VI.md`
- Vietnamese defense appendix (Backup Q&A): `report/PHU_LUC_VAN_DAP_VI.md`
- Report PDF: `output/pdf/VAD_MIDTERM_REPORT.pdf`
- Vietnamese explanation PDF: `output/pdf/VAD_MIDTERM_REPORT_VI.pdf`

Replace the student-name placeholders before submission.


