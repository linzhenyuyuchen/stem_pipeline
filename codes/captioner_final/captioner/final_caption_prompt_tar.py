#!/usr/bin/env python3
"""Final prompt for tar-backed inputs with educational metadata priors."""

from __future__ import annotations


FINAL_PROMPT_TEMPLATE = """<image 1>
You are an expert image captioner. Inspect the image first; the following educational priors are only soft hints:
Domain: {domain_prior}
Genre: {genre_prior}
Subject: {subject_prior}
Course: {course_prior}
Knowledge points: {knowledge_points_prior}

First check whether the domain and genre priors are consistent with the image. If either prior is clearly incorrect, or if the image has very low information density, poor quality, severe blur, degradation, cropping or otherwise cannot support a reliable caption, output a refusal. If the image is blank, nearly blank, too unclear, severely cropped or degraded, contains too little meaningful visible information, is only a pure text question or textual narrative, or is otherwise too simple for a reliable useful caption, refuse. Do not refuse a clear image merely because it is visually simple. If refusing, output exactly one line beginning with "REFUSAL:" followed by a brief reason.

If not refusing, describe the image in detail and preserve all visible, caption-worthy information and use the priors to add accurate, image-grounded explanations from appropriate knowledge or disciplinary perspectives, but do not force them, repeat or merely list them, and never invent details or conclusions unsupported by the image. 

Output only natural caption prose in one or more paragraphs in English. Do not use Markdown, headings, lists, labels, JSON, code fences or surrounding commentary. Do not prefix the response with "Caption:"."""


def render_final_prompt(
    *,
    domain_prior: str | None = None,
    genre_prior: str | None = None,
    subject_prior: str | None = None,
    course_prior: str | None = None,
    knowledge_points_prior: str | None = None,
    domain_prompt: str | None = None,
    genre_prompt: str | None = None,
) -> str:
    def value(text: str | None) -> str:
        return text.strip() if text and text.strip() else "Unspecified"

    return FINAL_PROMPT_TEMPLATE.format(
        domain_prior=value(domain_prior),
        genre_prior=value(genre_prior),
        subject_prior=value(subject_prior),
        course_prior=value(course_prior),
        knowledge_points_prior=value(knowledge_points_prior),
    ).strip()
