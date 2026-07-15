"""EN->UZ translation-quality benchmark harness.

Implements METHODOLOGY.md: build a stratified evaluation set, translate it
through every candidate system, and score with structural gates, XCOMET-QE,
a GEMBA-MQM Gemini judge, and chrF++ / reference-mode XCOMET on human refsets.
"""
