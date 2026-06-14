"""
app.py
------
Interactive Jupyter widget UI for keyword extraction.

Run this file inside a Jupyter Notebook or JupyterLab cell:

    %run app.py

Or import and call directly:

    from app import launch_app
    launch_app()
"""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from sklearn.model_selection import train_test_split

from keyword_extractor import KeywordExtractor, load_dataset, prepare_data

# ---------------------------------------------------------------------------
# Bootstrap: load data and fit model once at import time
# ---------------------------------------------------------------------------

print("Loading dataset and fitting model — this may take a moment …")
_df = load_dataset()
_df = prepare_data(_df)
_train_df, _ = train_test_split(_df, test_size=0.2, random_state=42)

_extractor = KeywordExtractor(max_features=30_000, ngram_range=(1, 3))
_extractor.fit(_train_df["clean_text"])
print("Model ready.\n")


# ---------------------------------------------------------------------------
# Widget UI
# ---------------------------------------------------------------------------

_STYLES = """
<style>
  .kw-title   { font-family: sans-serif; font-size: 22px; font-weight: 600; margin-bottom: 4px; }
  .kw-sub     { font-family: sans-serif; font-size: 13px; color: #888; margin-bottom: 20px; }
  .kw-pill    {
    display: inline-block; padding: 6px 16px; margin: 4px;
    border-radius: 999px; font-size: 13px; font-family: sans-serif;
    border: 1px solid #e0e0e0; background: #fafafa; color: #333;
  }
  .kw-divider { border: none; border-top: 1px solid #f0f0f0; margin: 20px 0; }
  .kw-count   { font-size: 12px; color: #bbb; margin-bottom: 12px; }
</style>
"""


def launch_app(extractor: KeywordExtractor = _extractor) -> None:
    """Render the keyword extractor widget in a Jupyter environment."""

    display(HTML(_STYLES + """
        <div class="kw-title">Keyword Extractor</div>
        <div class="kw-sub">Paste your text, choose how many keywords, and hit Extract.</div>
    """))

    textarea = widgets.Textarea(
        placeholder="Paste or type your text here …",
        layout=widgets.Layout(width="100%", height="160px"),
    )

    slider = widgets.IntSlider(
        value=20, min=2, max=40, step=1,
        description="Keywords:",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="100%", margin="10px 0"),
    )

    btn = widgets.Button(
        description="Extract Keywords",
        layout=widgets.Layout(margin="8px 0"),
    )
    btn.style.button_color = "#7c3aed"

    # ipywidgets ≥ 8 renamed text_color → font_color
    try:
        btn.style.font_color = "white"
    except AttributeError:
        btn.style.text_color = "white"   # ipywidgets < 8 fallback

    output = widgets.Output()

    def on_click(_btn):
        text = textarea.value.strip()
        top_n = slider.value

        with output:
            clear_output()

            if not text:
                display(HTML("<p style='color:#c00;'>⚠ Please enter some text.</p>"))
                return

            display(HTML("<p style='color:#aaa;'>Extracting …</p>"))
            clear_output(wait=True)

            keywords = sorted(extractor.extract(text, top_n=top_n))
            pills = "".join(f'<span class="kw-pill">{kw}</span>' for kw in keywords)

            display(HTML(f"""
                <hr class="kw-divider">
                <div class="kw-count">{len(keywords)} keyword(s) found</div>
                <div>{pills}</div>
            """))

    btn.on_click(on_click)
    display(textarea, slider, btn, output)


# ---------------------------------------------------------------------------
# Entry point (allows `%run app.py` in Jupyter)
# ---------------------------------------------------------------------------

launch_app()
