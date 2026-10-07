"""textscore: segmentation, aggregation and labelling for span-level classifiers.

The pipeline is deliberately split at the model boundary. Anything that knows
how to produce a probability per span (a transformer, an API, a heuristic) is
yours; everything after it — splitting prose, weighting spans, deciding how
much to trust the resulting number, painting it back onto the source, and
logging the request without logging the document — lives here, with no runtime
dependencies.
"""

from .aggregate import Config, DocumentScore, SpanScore, aggregate, band_of
from .highlight import render_ansi, render_html, spans_from_sentences
from .logsafe import DEFAULT_ALLOWED_FIELDS, EventLogger, TextLeakError, get_logger
from .metrics import accuracy_at, brier, by_group, class_means, evaluate, roc_auc
from .segment import Sentence, split_sentences, word_count

__version__ = "0.1.0"

__all__ = [
    "Config",
    "DEFAULT_ALLOWED_FIELDS",
    "DocumentScore",
    "EventLogger",
    "Sentence",
    "SpanScore",
    "TextLeakError",
    "__version__",
    "accuracy_at",
    "aggregate",
    "band_of",
    "by_group",
    "class_means",
    "evaluate",
    "get_logger",
    "render_ansi",
    "render_html",
    "roc_auc",
    "spans_from_sentences",
    "split_sentences",
    "word_count",
]
