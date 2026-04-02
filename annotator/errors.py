"""Exception hierarchy for annotator."""

from __future__ import annotations


class AnnotatorError(Exception):
    """Base exception for all annotator errors."""


class AuthError(AnnotatorError):
    """Authentication failures."""


class ResolverError(AnnotatorError):
    """Hardware detection or model selection failures."""


class EngineError(AnnotatorError):
    """Model loading or inference failures."""


class KombinatError(AnnotatorError):
    """kombinat API errors."""
