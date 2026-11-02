"""Contracts for the sample algorithm retriever and the Portainer deploy call."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SAMPLE_NOTICE: str = "Demonstrative sample config. Not a production algorithm package."
AlgorithmId = Literal["line_crossing", "zone_presence", "movement_heatmap"]


class NormalizedPoint(BaseModel):
    """One point in normalized image coordinates.

    Attributes
    ----------
    x : float
        Horizontal position from 0 to 1.
    y : float
        Vertical position from 0 to 1.
    """

    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0, le=1, description="Horizontal position from 0 to 1.")
    y: float = Field(ge=0, le=1, description="Vertical position from 0 to 1.")


class LineCrossingConfig(BaseModel):
    """Sample line a person must cross.

    Attributes
    ----------
    camera_id : str
        Sample camera name. It does not address a real device.
    line_start : NormalizedPoint
        First endpoint of the virtual line.
    line_end : NormalizedPoint
        Second endpoint of the virtual line.
    direction : {"in", "out", "in_out"}
        Which crossing direction the sample counts.
    min_confidence : float
        Minimum detection confidence, from 0 to 1.
    min_track_frames : int
        How many frames a track must live before a crossing counts.
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1, max_length=64, description="Sample camera name.")
    line_start: NormalizedPoint = Field(description="First endpoint of the virtual line.")
    line_end: NormalizedPoint = Field(description="Second endpoint of the virtual line.")
    direction: Literal["in", "out", "in_out"] = Field(description="Crossing direction to count.")
    min_confidence: float = Field(gt=0, le=1, description="Minimum detection confidence.")
    min_track_frames: int = Field(ge=1, le=60, description="Minimum track length in frames.")


class ZonePresenceConfig(BaseModel):
    """Sample polygon for a person standing on the sales floor.

    Attributes
    ----------
    camera_id : str
        Sample camera name. It does not address a real device.
    zone : list[NormalizedPoint]
        Polygon vertices, in order. Three to eight points.
    min_dwell_sec : float
        How long a person must stay inside the polygon.
    min_confidence : float
        Minimum detection confidence, from 0 to 1.
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1, max_length=64, description="Sample camera name.")
    zone: list[NormalizedPoint] = Field(
        min_length=3,
        max_length=8,
        description="Polygon vertices in order.",
    )
    min_dwell_sec: float = Field(gt=0, le=3600, description="Minimum time inside the zone, in seconds.")
    min_confidence: float = Field(gt=0, le=1, description="Minimum detection confidence.")


class MovementHeatmapConfig(BaseModel):
    """Sample grid that accumulates movement.

    Attributes
    ----------
    camera_id : str
        Sample camera name. It does not address a real device.
    grid_columns : int
        Horizontal cells.
    grid_rows : int
        Vertical cells.
    window_sec : int
        Accumulation window in seconds.
    min_confidence : float
        Minimum detection confidence, from 0 to 1.
    """

    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(min_length=1, max_length=64, description="Sample camera name.")
    grid_columns: int = Field(ge=4, le=128, description="Horizontal cells.")
    grid_rows: int = Field(ge=4, le=128, description="Vertical cells.")
    window_sec: int = Field(ge=30, le=86400, description="Accumulation window in seconds.")
    min_confidence: float = Field(gt=0, le=1, description="Minimum detection confidence.")


AlgorithmConfig = LineCrossingConfig | ZonePresenceConfig | MovementHeatmapConfig


class AlgorithmDocument(BaseModel):
    """One retrieval document and the config it is allowed to deploy.

    Attributes
    ----------
    algorithm_id : {"line_crossing", "zone_presence", "movement_heatmap"}
        Stable id. It selects both the document and the config shape.
    title : str
        Short name shown in the response.
    summary : str
        One sentence about the sample.
    notice : str
        Fixed sentence that this document is not a production package.
    sample : True
        Always true. The corpus does not carry production documents.
    phrases : list[str]
        Substrings the retriever looks for in the request.
    stack_name : str
        Portainer stack name. It must start with ``sample-``.
    image : str
        Compose image reference. It must start with ``demo/``.
    config : LineCrossingConfig | ZonePresenceConfig | MovementHeatmapConfig
        Parameters copied into the stack. The shape must match ``algorithm_id``.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm_id: AlgorithmId = Field(description="Stable sample algorithm id.")
    title: str = Field(min_length=1, max_length=120, description="Short name.")
    summary: str = Field(min_length=1, max_length=400, description="What this sample illustrates.")
    notice: Literal[_SAMPLE_NOTICE] = Field(description="Fixed reminder that this is a sample.")
    sample: Literal[True] = Field(description="True for every document in this corpus.")
    phrases: list[str] = Field(min_length=1, description="Substrings matched against the request.")
    stack_name: str = Field(
        pattern=r"^sample-[a-z0-9-]{1,40}$",
        description="Portainer stack name.",
    )
    image: str = Field(
        pattern=r"^demo/[a-z0-9._/-]+:[A-Za-z0-9._-]+$",
        description="Sample image reference. It is not a published product image.",
    )
    config: AlgorithmConfig = Field(description="Parameters for this algorithm id.")

    @model_validator(mode="after")
    def config_matches_algorithm(self) -> "AlgorithmDocument":
        """Reject a document whose config shape belongs to another algorithm.

        Returns
        -------
        AlgorithmDocument
            This document, when the config class matches ``algorithm_id``.

        Raises
        ------
        ValueError
            When the parsed config class does not belong to ``algorithm_id``.
        """
        expected: dict[str, type[BaseModel]] = {
            "line_crossing": LineCrossingConfig,
            "zone_presence": ZonePresenceConfig,
            "movement_heatmap": MovementHeatmapConfig,
        }
        if not isinstance(self.config, expected[self.algorithm_id]):
            raise ValueError(f"{self.algorithm_id} does not match the config shape")
        return self


class RetrieveAlgorithmInput(BaseModel):
    """Request text scored against the sample corpus.

    Attributes
    ----------
    query : str
        Sentence such as "deploy line crossing".
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000, description="Deployment request.")


class RetrievalHit(BaseModel):
    """One document score.

    Attributes
    ----------
    algorithm_id : {"line_crossing", "zone_presence", "movement_heatmap"}
        Document id.
    title : str
        Document title.
    document : str
        File name inside ``knowledge/``.
    score : int
        Sum of the matched phrase weights. Zero means no phrase matched.
    matched_phrases : list[str]
        Phrases that occurred in the query, in document order.
    """

    algorithm_id: AlgorithmId = Field(description="Document id.")
    title: str = Field(description="Document title.")
    document: str = Field(description="Knowledge file name.")
    score: int = Field(ge=0, description="Matched-phrase weight.")
    matched_phrases: list[str] = Field(description="Phrases found in the query.")


class RetrieveAlgorithmResult(BaseModel):
    """Scores for every sample document and the winner, if one exists.

    Attributes
    ----------
    query : str
        Text that was scored.
    selected : RetrievalHit | None
        Highest score above zero. None when every score is zero.
    candidates : list[RetrievalHit]
        All documents, highest score first.
    """

    query: str = Field(description="Text that was scored.")
    selected: RetrievalHit | None = Field(description="Winning document, or null when nothing matched.")
    candidates: list[RetrievalHit] = Field(description="Every document score, highest first.")


class DeployAlgorithmInput(BaseModel):
    """A corpus id to render and send to Portainer.

    Attributes
    ----------
    algorithm_id : {"line_crossing", "zone_presence", "movement_heatmap"}
        Document to deploy. The caller does not supply the config.
    endpoint_id : int | None
        Portainer endpoint. The process setting is used when this is omitted.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm_id: AlgorithmId = Field(description="Sample document to deploy.")
    endpoint_id: int | None = Field(
        default=None,
        ge=1,
        description="Portainer endpoint id. The process default is used when omitted.",
    )


class PortainerDeployResult(BaseModel):
    """What the Portainer call did, or why it was skipped.

    Attributes
    ----------
    mode : {"mock", "live"}
        ``mock`` when the URL or API key is blank. ``live`` when a request was sent.
    deployed : bool
        True only when Portainer accepted the stack.
    endpoint_id : int
        Endpoint id that was addressed, or would have been addressed.
    stack_name : str
        Stack name taken from the document.
    method : {"POST"}
        HTTP method of the Portainer call.
    path : str
        Path of the Portainer call, without the origin.
    stack_id : int | None
        Id returned by Portainer. Null in mock mode and on failure.
    message : str
        One sentence about the outcome. It does not include the API key.
    """

    mode: Literal["mock", "live"] = Field(description="mock when no Portainer call was sent.")
    deployed: bool = Field(description="True when Portainer accepted the stack.")
    endpoint_id: int = Field(ge=1, description="Portainer endpoint id.")
    stack_name: str = Field(description="Stack name from the selected document.")
    method: Literal["POST"] = Field(description="HTTP method.")
    path: str = Field(description="Portainer path, without the origin.")
    stack_id: int | None = Field(description="Stack id from Portainer, when the call succeeded.")
    message: str = Field(description="Outcome sentence. The API key is not included.")


class AlgorithmDeployRequest(BaseModel):
    """Body of ``POST /api/v1/deploy-algorithm``.

    Attributes
    ----------
    request : str
        Sentence used only for retrieval.
    endpoint_id : int | None
        Optional Portainer endpoint override.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"request": "Продеплой детекцию пересечения линии на входе"}
            ]
        },
    )

    request: str = Field(
        min_length=1,
        max_length=2000,
        description="Deployment sentence. It is matched against the sample corpus.",
        examples=["Продеплой детекцию пересечения линии на входе"],
    )
    endpoint_id: int | None = Field(
        default=None,
        ge=1,
        description="Portainer endpoint id. The process default is used when omitted.",
    )


class AlgorithmDeployResponse(BaseModel):
    """Selected sample config and the Portainer outcome.

    Attributes
    ----------
    status : {"mock", "deployed", "no_match", "deploy_failed"}
        ``no_match`` deploys nothing. ``mock`` selected a config and skipped HTTP.
    mode : {"mock", "live"}
        Whether a Portainer request was attempted.
    retrieval : RetrieveAlgorithmResult
        Scores that explain the selection.
    algorithm_id : {"line_crossing", "zone_presence", "movement_heatmap"} | None
        Selected id. Null when nothing matched.
    title : str | None
        Selected title. Null when nothing matched.
    config : dict[str, Any] | None
        Selected parameters. Null when nothing matched.
    compose : str | None
        Compose text that was, or would have been, posted. Null when nothing matched.
    portainer : PortainerDeployResult | None
        Deploy call. Null when nothing matched.
    """

    status: Literal["mock", "deployed", "no_match", "deploy_failed"] = Field(
        description="no_match deploys nothing. mock skips the HTTP call."
    )
    mode: Literal["mock", "live"] = Field(description="live when a Portainer request was sent.")
    retrieval: RetrieveAlgorithmResult = Field(description="Document scores.")
    algorithm_id: AlgorithmId | None = Field(description="Selected algorithm, or null.")
    title: str | None = Field(description="Selected title, or null.")
    config: dict[str, Any] | None = Field(description="Selected parameters, or null.")
    compose: str | None = Field(description="Compose text for the stack, or null.")
    portainer: PortainerDeployResult | None = Field(description="Portainer outcome, or null.")
