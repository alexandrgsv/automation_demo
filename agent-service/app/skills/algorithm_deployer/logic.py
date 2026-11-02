"""Retrieve one sample algorithm and create its Portainer stack.

The corpus is three JSON files. Scoring is phrase overlap, so the same
sentence always selects the same document. A blank Portainer URL or API key
skips HTTP and returns the compose that would have been posted.

The image name in those files is a sample label. This module does not pull it.
"""

import json
from pathlib import Path
from typing import Any, Literal

import httpx

from app.config import Settings
from app.skills.algorithm_deployer.schema import (
    AlgorithmDeployRequest,
    AlgorithmDeployResponse,
    AlgorithmDocument,
    AlgorithmId,
    PortainerDeployResult,
    RetrievalHit,
    RetrieveAlgorithmResult,
)

_KNOWLEDGE_DIR: Path = Path(__file__).resolve().parent / "knowledge"
_STACK_PATH: str = "/api/stacks/create/standalone/string"


def load_documents() -> tuple[AlgorithmDocument, ...]:
    """Read and validate every JSON document in the knowledge directory.

    Returns
    -------
    tuple[AlgorithmDocument, ...]
        Documents in file-name order.

    Raises
    ------
    ValueError
        If the directory is empty or two files share an algorithm id.
    """
    paths: list[Path] = sorted(_KNOWLEDGE_DIR.glob("*.json"))
    if not paths:
        raise ValueError(f"No algorithm documents in {_KNOWLEDGE_DIR}")
    documents: list[AlgorithmDocument] = []
    seen: set[str] = set()
    for path in paths:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        document: AlgorithmDocument = AlgorithmDocument.model_validate(payload)
        if document.algorithm_id in seen:
            raise ValueError(f"Duplicate algorithm id: {document.algorithm_id}")
        seen.add(document.algorithm_id)
        documents.append(document)
    return tuple(documents)


def document_by_id(algorithm_id: AlgorithmId) -> tuple[AlgorithmDocument, str]:
    """Return one document and its file name.

    Parameters
    ----------
    algorithm_id : {"line_crossing", "zone_presence", "movement_heatmap"}
        Id stored in the document.

    Returns
    -------
    tuple[AlgorithmDocument, str]
        The document and the JSON file name.

    Raises
    ------
    KeyError
        If the id is not in the corpus.
    """
    for path in sorted(_KNOWLEDGE_DIR.glob("*.json")):
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("algorithm_id") == algorithm_id:
            return AlgorithmDocument.model_validate(payload), path.name
    raise KeyError(algorithm_id)


def _phrase_score(query: str, phrases: list[str]) -> tuple[int, list[str]]:
    """Score a query by the phrases it contains.

    A matched phrase adds two points plus one point per word. Longer phrases
    therefore beat a short accidental overlap.

    Parameters
    ----------
    query : str
        Deployment sentence.
    phrases : list[str]
        Phrases from one document, in file order.

    Returns
    -------
    tuple[int, list[str]]
        The score and the phrases that matched.
    """
    folded: str = query.casefold()
    matched: list[str] = []
    total: int = 0
    for phrase in phrases:
        needle: str = phrase.casefold().strip()
        if needle and needle in folded:
            matched.append(phrase)
            total += len(needle.split()) + 2
    return total, matched


def retrieve_algorithm(query: str) -> RetrieveAlgorithmResult:
    """Score every sample document and keep the highest positive score.

    Ties break toward the document with more matched phrases, then toward
    the alphabetically earlier algorithm id.

    Parameters
    ----------
    query : str
        Deployment sentence.

    Returns
    -------
    RetrieveAlgorithmResult
        Every candidate and the winner. ``selected`` is null when every score is zero.
    """
    hits: list[RetrievalHit] = []
    for path in sorted(_KNOWLEDGE_DIR.glob("*.json")):
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        document: AlgorithmDocument = AlgorithmDocument.model_validate(payload)
        score, matched = _phrase_score(query, document.phrases)
        hits.append(
            RetrievalHit(
                algorithm_id=document.algorithm_id,
                title=document.title,
                document=path.name,
                score=score,
                matched_phrases=matched,
            )
        )
    ranked: list[RetrievalHit] = sorted(
        hits,
        key=lambda item: (-item.score, -len(item.matched_phrases), item.algorithm_id),
    )
    selected: RetrievalHit | None = ranked[0] if ranked and ranked[0].score > 0 else None
    return RetrieveAlgorithmResult(query=query, selected=selected, candidates=ranked)


def render_compose(document: AlgorithmDocument) -> str:
    """Render the sample compose file for one document.

    Parameters
    ----------
    document : AlgorithmDocument
        Validated corpus entry. Its image and config are copied, not invented.

    Returns
    -------
    str
        Compose text whose config JSON is a single-quoted scalar.
    """
    config_json: str = json.dumps(
        document.config.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (
        "# Sample stack generated for a demo. The image name is not a published product.\n"
        f"# stack: {document.stack_name}\n"
        "services:\n"
        "  detector:\n"
        f"    image: {document.image}\n"
        "    environment:\n"
        f"      ALGORITHM_ID: {document.algorithm_id}\n"
        f"      ALGORITHM_CONFIG: '{config_json}'\n"
        '    restart: "no"\n'
    )


def _origin(url: str) -> str:
    """Return the Portainer origin without a trailing slash or an ``/api`` suffix.

    Parameters
    ----------
    url : str
        Value of ``PORTAINER_URL``.

    Returns
    -------
    str
        Origin used to join ``_STACK_PATH``.
    """
    origin: str = url.strip().rstrip("/")
    if origin.endswith("/api"):
        origin = origin[: -len("/api")]
    return origin


def _portainer_configured(settings: Settings) -> bool:
    """Return True when both the origin and the API key are present.

    Parameters
    ----------
    settings : Settings
        Process settings.

    Returns
    -------
    bool
        True when a live call should be attempted.
    """
    return bool(settings.portainer_url.strip() and settings.portainer_api_key.strip())


def deploy_document(
    document: AlgorithmDocument,
    settings: Settings,
    endpoint_id: int | None = None,
    client: httpx.Client | None = None,
) -> tuple[str, PortainerDeployResult]:
    """Create the sample stack, or describe the call when Portainer is not configured.

    Parameters
    ----------
    document : AlgorithmDocument
        Document selected by retrieval or by id.
    settings : Settings
        Portainer origin, key, endpoint, and timeout.
    endpoint_id : int | None
        Request override. ``settings.portainer_endpoint_id`` is used when omitted.
    client : httpx.Client | None
        Open client. Tests pass a mock transport. A client is created when omitted.

    Returns
    -------
    tuple[str, PortainerDeployResult]
        Compose text and the outcome. The API key is not copied into the result.
    """
    resolved_endpoint: int = endpoint_id if endpoint_id is not None else settings.portainer_endpoint_id
    compose: str = render_compose(document)
    if not _portainer_configured(settings):
        skipped: PortainerDeployResult = PortainerDeployResult(
            mode="mock",
            deployed=False,
            endpoint_id=resolved_endpoint,
            stack_name=document.stack_name,
            method="POST",
            path=_STACK_PATH,
            stack_id=None,
            message="Portainer URL or API key is empty. The stack was not created.",
        )
        return compose, skipped

    body: dict[str, Any] = {
        "name": document.stack_name,
        "stackFileContent": compose,
        "env": [
            {"name": "ALGORITHM_ID", "value": document.algorithm_id},
            {
                "name": "ALGORITHM_CONFIG",
                "value": json.dumps(document.config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True),
            },
        ],
    }
    owns_client: bool = client is None
    http: httpx.Client = client or httpx.Client(timeout=settings.portainer_timeout_s)
    try:
        response: httpx.Response = http.post(
            f"{_origin(settings.portainer_url)}{_STACK_PATH}",
            params={"endpointId": resolved_endpoint},
            headers={"X-API-Key": settings.portainer_api_key},
            json=body,
        )
    except httpx.HTTPError as exc:
        failed: PortainerDeployResult = PortainerDeployResult(
            mode="live",
            deployed=False,
            endpoint_id=resolved_endpoint,
            stack_name=document.stack_name,
            method="POST",
            path=_STACK_PATH,
            stack_id=None,
            message=f"Portainer request failed: {type(exc).__name__}.",
        )
        return compose, failed
    finally:
        if owns_client:
            http.close()

    if response.status_code >= 400:
        detail: str = _error_detail(response)
        rejected: PortainerDeployResult = PortainerDeployResult(
            mode="live",
            deployed=False,
            endpoint_id=resolved_endpoint,
            stack_name=document.stack_name,
            method="POST",
            path=_STACK_PATH,
            stack_id=None,
            message=f"Portainer returned {response.status_code}. {detail}",
        )
        return compose, rejected

    stack_id: int | None = _stack_id(response)
    created: PortainerDeployResult = PortainerDeployResult(
        mode="live",
        deployed=True,
        endpoint_id=resolved_endpoint,
        stack_name=document.stack_name,
        method="POST",
        path=_STACK_PATH,
        stack_id=stack_id,
        message="Portainer accepted the sample stack.",
    )
    return compose, created


def _error_detail(response: httpx.Response) -> str:
    """Read a short Portainer error message.

    Parameters
    ----------
    response : httpx.Response
        Response whose status is 400 or higher.

    Returns
    -------
    str
        The ``message`` field, or a fixed fallback. Limited to 200 characters.
    """
    try:
        payload: Any = response.json()
    except json.JSONDecodeError:
        return "No JSON error body."
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        return payload["message"][:200]
    return "No JSON error body."


def _stack_id(response: httpx.Response) -> int | None:
    """Read the numeric stack id from a Portainer success body.

    Parameters
    ----------
    response : httpx.Response
        Response whose status is below 400.

    Returns
    -------
    int | None
        ``Id`` when it is an integer. None when the body has another shape.
    """
    try:
        payload: Any = response.json()
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("Id"), int):
        return payload["Id"]
    return None


def deploy_from_request(
    payload: AlgorithmDeployRequest,
    settings: Settings,
    client: httpx.Client | None = None,
) -> AlgorithmDeployResponse:
    """Retrieve a sample config and deploy it when Portainer is configured.

    Parameters
    ----------
    payload : AlgorithmDeployRequest
        Request sentence and optional endpoint id.
    settings : Settings
        Portainer settings for this call.
    client : httpx.Client | None
        Optional HTTP client. Used only when a document matches and Portainer is configured.

    Returns
    -------
    AlgorithmDeployResponse
        Scores, the selected config, and the Portainer outcome.
        A query that matches nothing does not call Portainer.
    """
    retrieval: RetrieveAlgorithmResult = retrieve_algorithm(payload.request)
    if retrieval.selected is None:
        return AlgorithmDeployResponse(
            status="no_match",
            mode="mock",
            retrieval=retrieval,
            algorithm_id=None,
            title=None,
            config=None,
            compose=None,
            portainer=None,
        )

    document, _name = document_by_id(retrieval.selected.algorithm_id)
    compose, portainer = deploy_document(
        document,
        settings,
        endpoint_id=payload.endpoint_id,
        client=client,
    )
    status: Literal["mock", "deployed", "deploy_failed"]
    if portainer.deployed:
        status = "deployed"
    elif portainer.mode == "live":
        status = "deploy_failed"
    else:
        status = "mock"
    return AlgorithmDeployResponse(
        status=status,
        mode=portainer.mode,
        retrieval=retrieval,
        algorithm_id=document.algorithm_id,
        title=document.title,
        config=document.config.model_dump(mode="json"),
        compose=compose,
        portainer=portainer,
    )
