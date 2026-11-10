"""Client OpenRouter (API OpenAI-compatible).

OpenRouter expose un agrégateur de modèles derrière une API compatible OpenAI
(mêmes routes `/chat/completions` et `/models`, même schéma de réponse), avec
une authentification `Authorization: Bearer <clé>` et deux en-têtes
d'attribution facultatifs (`HTTP-Referer`, `X-Title`) servant aux classements.

On réutilise donc intégralement la plomberie HTTP d'`AlbertClient` ; seuls
changent l'URL de base, la variable d'environnement de la clé, et les deux
en-têtes d'attribution. Voir https://openrouter.ai/docs/quickstart.

Intérêt pour ce repo : juger les transcrits avec un modèle TIERS (Claude, GPT,
Gemini…) plutôt qu'un Mistral servi par Albert, afin d'écarter le biais d'un
juge Mistral notant un transcript produit par un modèle Mistral (Voxtral) ou de
la même famille.
"""

from __future__ import annotations

import os

from eval_transcript.albert import AlbertClient

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"

# Modèle juge par défaut côté OpenRouter : volontairement NON-Mistral pour
# éviter le biais d'un juge de la même famille que les transcrits évalués.
# Surchargeable via --judge-model. Catalogue à jour : https://openrouter.ai/models
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4.5"

# Attribution (facultative) : remontée dans les classements OpenRouter.
DEFAULT_APP_TITLE = "eval-transcript"


class OpenRouterClient(AlbertClient):
    """Client OpenRouter, sous-classe d'`AlbertClient` (API OpenAI-compatible)."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        app_title: str | None = None,
        app_url: str | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
            timeout=timeout,
        )
        # Ne PAS laisser la résolution de clé au parent : sans clé OpenRouter, il
        # se rabattrait sur $ALBERT_API_KEY. On résout ici, explicitement.
        self.api_key = api_key if api_key is not None else os.getenv(DEFAULT_API_KEY_ENV)
        self.provider_name = "OpenRouter"
        self.app_title = app_title or os.getenv("OPENROUTER_APP_TITLE") or DEFAULT_APP_TITLE
        self.app_url = app_url or os.getenv("OPENROUTER_APP_URL")

    @property
    def headers(self) -> dict[str, str]:
        headers = dict(super().headers)
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers
