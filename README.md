# ColorChecker Analysis

[![DOI](https://zenodo.org/badge/938052346.svg)](https://doi.org/10.5281/zenodo.15646472)

A Python-based application for performing DeltaE analysis on ColorChecker charts. This tool offers both automatic detection and manual selection of the ColorChecker pattern from an image, computes the DeltaE differences for each color patch, supports batch processing of images after pattern selection, and displays detailed visualizations.

<p align="center">
  <img src=https://github.com/user-attachments/assets/0297b31a-a8d8-49ef-a141-6b4a4e2062c1 width="49.5%" />
  <img src=https://github.com/user-attachments/assets/8da6f279-b346-4c9b-8679-de871d55a9ea width="49.5%" />
</p>


## Features

- **Image Loading:** Browse and load images via an integrated file explorer.
- **File Filtering:** Filter images in the file explorer using Python-style expressions.
- **ColorChecker Detection:**
  - **Auto-detection:** Uses OpenCV's MCC detector to automatically find the ColorChecker pattern.
  - **Manual Selection:** Adjust or specify the ColorChecker vertices using a simple Tkinter interface.
- **DeltaE Analysis:** Computes the DeltaE (CIE2000) differences between detected patches and a reference.
- **Batch Processing:** Perform DeltaE analysis on multiple images after selecting the pattern.
- **Visualization:** Displays analysis results and histograms using matplotlib.
- **Plot Saving:** Save generated plots with custom DPI and figure dimensions.
- **Logging:** Real-time logging of analysis steps directly within the GUI.

## Installation

### Prerequisites

- **Python 3.x**

### Clone the Repository

```bash
git clone git@github.com:RISE-Dependable-Transport-Systems/ColorCheckerAnalysis.git
cd ColorCheckerAnalysis
```

### Install the required packages via pip

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
python src/colorchecker_analysis.py
```

### How It Works

1. **Load an Image:**  
   Use the file explorer on the left to select an image containing a ColorChecker chart.

2. **File Filtering:**  
   The file explorer allows the filtering of files using Python-like expressions with tokens and logical operators such as "not example and sample".
4. **Detect the ColorChecker Pattern:**  
   - Click **Auto detect** to let the application attempt automatic pattern detection.
   - If detection fails or needs fine-tuning, click **Manual selection** and adjust the vertices manually.

5. **Compute DeltaE:**  
   Once the pattern is set, click **Compute DeltaE** to perform the analysis. Results and a histogram of the DeltaE values will be displayed on the analysis tab.

6. **Batch Processing:**  
   After selecting the pattern, optionally select multiple images (with the ColorChecker pattern assumed to be in the same pixel region) in the file explorer and click **Compute DeltaE** to perform the batch analysis using parallel threads.

7. **Save Your Plot:**  
   Right-click on any plot to open a context menu with the option to save your figure.

## Contributing

Contributions are welcome! Feel free to fork the repository, make improvements, and submit pull requests. When contributing, please follow the coding style and ensure that new features are well documented.

## License

This project is licensed under the [GPL-3.0 license](LICENSE).

## Acknowledgments

- **colormath:** For the color conversion and DeltaE calculation routines.
- **OpenCV:** For powerful image processing capabilities.
- **matplotlib & Tkinter:** For providing an interactive UI and visualization tools.

## Contact

For questions or feedback, please open an issue in the GitHub repository.

## Funded by

<img src="https://user-images.githubusercontent.com/2404625/202213271-a4006999-49d5-4e61-9f3d-867a469238d1.png" width="120" height="81" align="left" alt="EU logo" />
This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement nº 101095835. The results reflect only the authors' view and the Agency is not responsible
for any use that may be made of the information it contains.
