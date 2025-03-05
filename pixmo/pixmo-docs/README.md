# pixmo-docs

This is the repository for the generation system of the [PixMo-Docs](https://huggingface.co/datasets/allenai/pixmo-docs), [CoSyn-400K](https://huggingface.co/datasets/allenai/CoSyn-400K), and [CoSyn-point](https://huggingface.co/datasets/allenai/CoSyn-point) datasets. PixMo-Docs was used to train the [Molmo](https://arxiv.org/abs/2409.17146) model, and the CoSyn datasets are an expanded version that use an improved pipeline and more types of documents. More details can be found in our [paper](https://arxiv.org/pdf/2502.14846).

## Installation
After cloning the repo, you can install the required dependencies using the following commands:

```bash
conda create --name pixmo-doc python=3.10
conda activate pixmo-doc
pip install -r requirements.txt
```

Then export your API key as an environment variable:

```bash
export OPENAI_API_KEY=your-api-key
export ANTHROPIC_API_KEY=your-api-key
export HF_TOKEN=your-api-key # only if you want to upload the dataset to the Hugging Face Hub
```

You need to install the following packages to use some of the pipelines:
1. LaTeX: the installation depends on your operating system, you can refer to the [official LaTeX website](https://www.latex-project.org/get/) for more details.

2. Mermaid: you can refer to [here](https://github.com/mermaid-js/mermaid-cli) to install the Mermaid CLI using npm:
    ```bash
    npm install -g @mermaid-js/mermaid-cli
    ```

3. HTML: install playwright with:

    ```bash
    pip install playwright
    playwright install
    ```

4. mplfinance:

   ```
   pip install mpl_finance<=0.10.1 mplfinance<=0.12.10b0
   ```

5. cairosvg:

   ```
   pip install cairosvg<=2.7.1
   ```

## Quick Start
The [main.py](main.py) script is the entry point for the generation of the dataset. You can use the following main arguments to control the generation process:

```python
python main.py -p {PIPELINE} \
               -t {TYPE_OF_DATA_YOU_WANT_TO_GENERATE} \
               -n {NUMBER_OF_SAMPLES} \
               -m {NAME_OF_DATASET} \
```

For example, `python main.py -p "MatplotlibChartPipeline" -n 5 -m "matplotlib_test" -t "bar chart"`, will generate 5 bar charts using the MatplotlibChartPipeline and save them with the name "matplotlib_test".

You can use comma separated values for the `-p` and `-t` arguments to generate multiple types of data using different pipelines at the same time.

Please refer to the [main.py](main.py) script for more details on the available arguments and their usage.


## Pipelines  
We released 25 pipelines to generate eight main categories of text-rich images: charts, tables, documents, diagrams, circuits, specialized graphics, and pointing. Each pipeline uses one renderer/programming language to generate the images.  

* **Chart**:  
    * *MatplotlibChartPipeline*: using [Matplotlib](https://matplotlib.org/) to generate charts like bar charts, line charts, etc. You can check the [Matplotlib gallery](https://matplotlib.org/stable/gallery/index.html) for possible charts.  
    * *PlotlyChartPipeline*: using [Plotly](https://plotly.com/python/) to generate charts. You can check the [Plotly gallery](https://plotly.com/python/) for possible charts.  
    * *VegaLiteChartPipeline*: using [Vega-Lite](https://vega.github.io/vega-lite/) to generate charts. You can check the [Vega-Lite gallery](https://vega.github.io/vega-lite/examples/) for possible usage.  
    * *LaTeXChartPipeline*: using TikZ to generate charts. This pipeline only works for simple charts like bar charts, line charts, etc.  
    * *HTMLChartPipeline*: using HTML and CSS to generate charts. This pipeline only works for simple charts like bar charts, line charts, etc.  

* **Table**:  
    * *LaTeXTablePipeline*: best for tables with complex structures.  
    * *MatplotlibTablePipeline*: uses Matplotlib to render tables within figures.  
    * *PlotlyTablePipeline*: only works for simple tables like single-header tables.  
    * *HTMLTablePipeline*: only works for simple tables like single-header tables.  

* **Document**:  
    * *LaTeXDocumentPipeline*: works for diverse types of documents like reports, articles, etc.  
    * *HTMLDocumentPipeline*: can create documents with complex styles and structures.  
    * *DOCXDocumentPipeline*: generates Microsoft Word-compatible `.docx` documents.  

* **Diagram**:  
    * *GraphvizDiagramPipeline*: using [Graphviz](https://graphviz.org/) to generate diagrams like directed graphs, trees, etc.  
    * *MermaidDiagramPipeline*: using [Mermaid](https://mermaid-js.github.io/mermaid/#/) to generate diagrams like flowcharts, sequence diagrams, etc.  
    * *LaTeXDiagramPipeline*: using TikZ to generate diagrams. You can refer to [this](https://texample.net/tikz/examples/tag/diagrams/) for possible diagrams.  

* **Circuit**:  
    * *SchemDrawCircuitPipeline*: uses [SchemDraw](https://schemdraw.readthedocs.io/) to generate electrical circuit diagrams.  
    * *LaTeXCircuitPipeline*: uses TikZ circuit libraries to generate circuit diagrams.  

* **Specialized Graphics**:  
    * *DALLEImagePipeline*: generates images using DALL·E models.  
    * *RdkitChemicalPipeline*: renders chemical structure diagrams using [RDKit](https://www.rdkit.org/).  
    * *LaTeXMathPipeline*: generates mathematical expressions using LaTeX.  
    * *LilyPondMusicPipeline*: generates sheet music using [LilyPond](http://lilypond.org/).  
    * *SVGGraphicPipeline*: creates vector graphics using SVG format.  
    * *AsymptoteGraphicPipeline*: uses [Asymptote](https://asymptote.sourceforge.io/) to generate mathematical and technical graphics.  

* **Web Screens**:
    * *HTMLScreenPipeline*: creates HTML-based screen layouts rendered with [Playwright](https://playwright.dev/) and Google Chrome / Chromium.  

* **Pointing**:  
    * *HTMLDocumentPointPipeline*: generates HTML documents with structured points.  




## Citation
Please cite the following papers if you use this codebase or our datasets in your work.

```bibtex
@article{yang2025scaling,
      title={Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation},
      author={Yang, Yue and Patel, Ajay and Deitke, Matt and Gupta, Tanmay and Weihs, Luca and Head, Andrew and Yatskar, Mark and Callison-Burch, Chris and Krishna, Ranjay and Kembhavi, Aniruddha and others},
      journal={arXiv preprint arXiv:2502.14846},
      year={2025}
}
```

```bibtex
@article{deitke2024molmo,
  title={Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models},
  author={Deitke, Matt and Clark, Christopher and Lee, Sangho and Tripathi, Rohun and Yang, Yue and Park, Jae Sung and Salehi, Mohammadreza and Muennighoff, Niklas and Lo, Kyle and Soldaini, Luca and others},
  journal={arXiv preprint arXiv:2409.17146},
  year={2024}
}
```
