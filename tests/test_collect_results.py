import numpy as np
import pandas as pd
import pytest

from scripts.collect_results import auroc, html_table, markdown_table, slug


def test_html_table_folds_the_index_name_row():
    markup = """
    <table><thead>
      <tr><th>subset</th><th>all</th><th>qa</th></tr>
      <tr><th>score</th><th></th><th></th></tr>
    </thead><tbody>
      <tr><th>SelfCheck&#45;NLI</th><td>0.594</td><td>0.684</td></tr>
    </tbody></table>
    """
    table = html_table(markup)
    assert table.columns.tolist() == ["score", "all", "qa"]
    assert table.iloc[0].tolist() == ["SelfCheck-NLI", "0.594", "0.684"]


def test_html_table_drops_a_blank_index_column():
    markup = """
    <table><tr><th></th><th>variant</th></tr><tr><th>0</th><td>nli</td></tr></table>
    """
    assert html_table(markup).to_dict("list") == {"variant": ["nli"]}


def test_markdown_table_formats_values():
    frame = pd.DataFrame({"name": ["a"], "n": [1234], "rate": [0.12345], "gap": [np.nan]})
    assert markdown_table(frame).splitlines()[2] == "| a | 1,234 | 0.123 |  |"


def test_auroc_handles_order_ties_and_one_class():
    labels = [0, 0, 1, 1]
    assert auroc([0.1, 0.2, 0.8, 0.9], labels) == pytest.approx(1.0)
    assert auroc([0.9, 0.8, 0.2, 0.1], labels) == pytest.approx(0.0)
    assert auroc([1, 1, 1, 1], labels) == pytest.approx(0.5)
    assert np.isnan(auroc([1, 2], [1, 1]))


def test_slug():
    assert slug("5. Precision–recall (paper Figure 5)") == "5_precision_recall_paper_figure_5"
