"""Typed public inputs, workflow state, and production planning artifacts."""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields so malformed provider output cannot pass silently."""

    model_config = ConfigDict(extra="forbid")


class JobStatus(str, Enum):
    """Externally visible asynchronous job states."""

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    RENDERING = "rendering"
    QA = "qa"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED_CONFLICT = "blocked_conflict"


class SourceMaterials(StrictModel):
    """Required and optional source documents for one workflow run."""

    financial_reports: list[str] = Field(min_length=1)
    earnings_transcript: str | None = None
    supplementary_files: list[str] = Field(default_factory=list)
    disclaimer: str = Field(min_length=1)


class ContentRequirements(StrictModel):
    """Content goal used only when narration is generated."""

    topic: str = "Quarterly earnings overview"
    objective: str = "Explain the principal results and outlook"
    required_metrics: list[str] = Field(default_factory=list)
    target_duration_seconds: int = Field(default=90, ge=15, le=600)


class Transcript(StrictModel):
    """Transcript mode and language with cross-field validation."""

    mode: Literal["generate", "pre-written"] = "generate"
    text: str | None = None
    allow_editing: bool = True
    language: str = "zh-CN"

    @model_validator(mode="after")
    def require_pre_written_text(self) -> "Transcript":
        """Reject pre-written mode without actual text."""

        if self.mode == "pre-written" and not (self.text or "").strip():
            raise ValueError("pre-written mode requires transcript.text")
        return self


class CreativeDirection(StrictModel):
    """Creative defaults exposed to planners and the renderer."""

    audience: str = "retail investors"
    tone: str = "engaging and credible"
    visual_style: str = "cartoon"
    pacing: str = "medium-fast"
    chart_style: str = "clean-modern"
    motion_style: str = "playful"


class AudioConfig(StrictModel):
    """Voice provider and speaking configuration."""

    provider: str = "elevenlabs"
    voice_id: str | None = None
    speaking_rate: float = Field(default=.95, ge=.7, le=1.2)
    background_music: bool = False
    language: str = "zh-CN"


class CaptionConfig(StrictModel):
    """Burned-in subtitle configuration."""

    enabled: bool = True
    mode: Literal["burned_in"] = "burned_in"
    language: str = "zh-CN"
    position: Literal["bottom"] = "bottom"
    max_lines: int = Field(default=2, ge=1, le=3)
    highlight_keywords: bool = True


class BrandConfig(StrictModel):
    """Minimal deterministic renderer brand tokens."""

    logo: str | None = None
    primary_color: str = Field(default="#76B900", pattern=r"^#[0-9A-Fa-f]{6}$")
    font_family: str = "Inter"


class OutputConfig(StrictModel):
    """Supported local render output settings."""

    aspect_ratio: Literal["16:9", "9:16", "1:1"] = "16:9"
    resolution: Literal["1920x1080", "1080x1920", "1080x1080"] = "1920x1080"
    fps: Literal[24, 25, 30, 60] = 30
    format: Literal["mp4"] = "mp4"

    @model_validator(mode="after")
    def ratio_matches_resolution(self) -> "OutputConfig":
        """Reject contradictory canvas geometry."""

        expected = {"16:9": "1920x1080", "9:16": "1080x1920", "1:1": "1080x1080"}
        if expected[self.aspect_ratio] != self.resolution:
            raise ValueError("output.aspect_ratio does not match output.resolution")
        return self


class JobInput(StrictModel):
    """Validated public workflow request with documented defaults."""

    source_materials: SourceMaterials
    content_requirements: ContentRequirements | None = None
    transcript: Transcript = Field(default_factory=Transcript)
    creative_direction: CreativeDirection = Field(default_factory=CreativeDirection)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    captions: CaptionConfig = Field(default_factory=CaptionConfig)
    brand: BrandConfig = Field(default_factory=BrandConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @model_validator(mode="after")
    def generated_content_requires_requirements(self) -> "JobInput":
        """Require content intent only when the system authors the narration."""

        if self.transcript.mode == "generate" and self.content_requirements is None:
            raise ValueError("generate mode requires content_requirements")
        return self


class MoneyQuantity(StrictModel):
    kind: Literal["money"]
    amount: float
    currency: Literal["USD"]


class MoneyPerShareQuantity(StrictModel):
    kind: Literal["money_per_share"]
    amount: float
    currency: Literal["USD"]


class PercentageQuantity(StrictModel):
    kind: Literal["percentage"]
    value: float
    semantics: Literal["level", "change"]


class PercentagePointsQuantity(StrictModel):
    kind: Literal["percentage_points"]
    value: float


class CountQuantity(StrictModel):
    kind: Literal["count"]
    value: float
    subject: str


class RatioQuantity(StrictModel):
    kind: Literal["ratio"]
    value: float


CanonicalQuantity = Annotated[Union[MoneyQuantity, MoneyPerShareQuantity, PercentageQuantity, PercentagePointsQuantity, CountQuantity, RatioQuantity], Field(discriminator="kind")]


class CanonicalFact(StrictModel):
    """One source-addressable financial fact in normalized base units."""

    id: str
    metric: str
    value: float
    unit: str
    scale: str
    currency: str | None
    basis: str
    fiscal_period: str
    period_end: str | None
    comparison: dict[str, float] = Field(default_factory=dict)
    source: str
    source_locator: str
    confidence: float = Field(ge=0, le=1)
    derived_from: list[str] = Field(default_factory=list)
    formula: str | None = None
    quantity: CanonicalQuantity | None = None
    reported: dict[str, Any] | None = None


class CanonicalFacts(StrictModel):
    """Versioned canonical financial-facts document."""

    schema_version: str
    entity: str
    ticker: str | None = None
    report: dict[str, Any]
    facts: list[CanonicalFact] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_fact_ids(self) -> "CanonicalFacts":
        """Ensure every fact reference resolves unambiguously."""

        ids = [fact.id for fact in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("canonical fact IDs must be unique")
        return self


class Scene(StrictModel):
    """A renderer-supported scene boundary."""

    id: str
    kind: Literal["content", "disclaimer"] = "content"
    visual_kind: Literal["chart", "metric_cards", "broll", "disclaimer"] = "broll"
    purpose: str
    duration_seconds: float = Field(gt=0, le=120)
    title: str | None = None
    chart: str | None = None
    asset_subject: str | None = None
    subject_id: str | None = None
    visual_prompt: str | None = None
    transition: Literal["cut", "fade", "slide", "wipe", "zoom"] = "cut"


class ScenePlan(StrictModel):
    """Ordered scene plan with stable retry identifiers."""

    scenes: list[Scene] = Field(min_length=1, max_length=12)


class NarrationSegment(StrictModel):
    """Display and spoken forms linked to canonical facts."""

    scene_id: str
    display_text: str
    spoken_text: str
    fact_ids: list[str] = Field(default_factory=list)


class Narration(StrictModel):
    """Narration artifact consumed by TTS and subtitles."""

    language: str
    segments: list[NarrationSegment] = Field(min_length=1)
    source: str = "generated"
    editing_applied: bool | None = None


class Chart(StrictModel):
    """Finite chart template selected by the planner."""

    id: str
    type: Literal["bar", "horizontal_bar", "donut", "range", "line", "price_line", "waterfall", "gauge", "metric_cards", "table"]
    title: str | None = None
    labels: list[str] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)
    formatted_values: list[str] = Field(default_factory=list)
    units: list[str] = Field(default_factory=list)
    unit: str | None = None
    midpoint: float | None = None
    low: float | None = None
    high: float | None = None
    fact_ids: list[str] = Field(default_factory=list)
    key_levels: list[dict[str, Any]] = Field(default_factory=list)
    animation: Literal["grow", "sweep", "pan-and-highlight", "draw-and-highlight"]


class ChartSpec(StrictModel):
    """Renderer-supported charts and reserved caption geometry."""

    reserved_regions: dict[str, dict[str, float]] = Field(default_factory=dict)
    charts: list[Chart] = Field(default_factory=list)


class PlanningBundle(StrictModel):
    """Strict production planner boundary validated before any downstream call."""

    canonical_facts: CanonicalFacts
    financial_analysis: dict[str, Any]
    story_plan: dict[str, Any]
    scene_plan: ScenePlan
    narration: Narration
    chart_spec: ChartSpec


# OpenAI Structured Outputs contract. The model chooses facts and references;
# application code owns unit conversion and every displayed numeric string.
class ProviderMoneyQuantity(StrictModel):
    kind: Literal["money"]
    amount: float
    currency: Literal["USD"]
    magnitude: Literal["ones", "thousands", "millions", "billions", "trillions"]


class ProviderMoneyPerShareQuantity(StrictModel):
    kind: Literal["money_per_share"]
    amount: float
    currency: Literal["USD"]
    magnitude: Literal["ones", "thousands", "millions", "billions", "trillions"]


class ProviderPercentageQuantity(StrictModel):
    kind: Literal["percentage"]
    value: float
    semantics: Literal["level", "change"]


class ProviderPercentagePointsQuantity(StrictModel):
    kind: Literal["percentage_points"]
    value: float


class ProviderCountQuantity(StrictModel):
    kind: Literal["count"]
    value: float
    subject: str
    magnitude: Literal["ones", "thousands", "millions", "billions", "trillions"]


class ProviderRatioQuantity(StrictModel):
    kind: Literal["ratio"]
    value: float


ProviderQuantity = Union[ProviderMoneyQuantity, ProviderMoneyPerShareQuantity, ProviderPercentageQuantity, ProviderPercentagePointsQuantity, ProviderCountQuantity, ProviderRatioQuantity]


class ProviderReport(StrictModel):
    title: str
    document_type: str
    fiscal_period: str
    period_end: str | None


class ProviderCanonicalFact(StrictModel):
    id: str
    metric: str
    quantity: ProviderQuantity
    reported_value: float
    reported_unit_text: str
    basis: str
    fiscal_period: str
    period_end: str | None
    source: str
    source_locator: str
    confidence: float = Field(ge=0, le=1)
    derived_from: list[str]
    formula: str | None


class ProviderCanonicalFacts(StrictModel):
    schema_version: str
    entity: str
    ticker: str | None
    report: ProviderReport
    facts: list[ProviderCanonicalFact] = Field(min_length=1)


class ProviderInsight(StrictModel):
    title: str
    summary: str
    fact_ids: list[str]


class ProviderFinancialAnalysis(StrictModel):
    summary: str
    insights: list[ProviderInsight]


class ProviderStoryBeat(StrictModel):
    purpose: str
    summary: str
    fact_ids: list[str]


class ProviderStoryPlan(StrictModel):
    title: str
    thesis: str
    beats: list[ProviderStoryBeat]


class ProviderScene(StrictModel):
    id: str
    kind: Literal["content", "disclaimer"]
    visual_kind: Literal["chart", "metric_cards", "broll", "disclaimer"]
    purpose: str
    duration_seconds: float = Field(gt=0, le=120)
    title: str | None
    chart: str | None
    asset_subject: str | None
    subject_id: str | None
    visual_prompt: str | None
    transition: Literal["cut", "fade", "slide", "wipe", "zoom"]


class ProviderTextPart(StrictModel):
    type: Literal["text"]
    value: str = Field(pattern=r"^[^0-9０-９$¥€£%]*$")


class ProviderFactPart(StrictModel):
    type: Literal["fact"]
    fact_id: str
    precision: int = Field(ge=0, le=4)
    compact: bool


ProviderNarrationPart = Union[ProviderTextPart, ProviderFactPart]


class ProviderNarrationSegment(StrictModel):
    scene_id: str
    parts: list[ProviderNarrationPart] = Field(min_length=1)


class ProviderNarration(StrictModel):
    language: str
    segments: list[ProviderNarrationSegment] = Field(min_length=1)
    source: str
    editing_applied: bool | None


class ProviderChartPoint(StrictModel):
    label: str
    fact_id: str
    role: Literal["value", "low", "midpoint", "high"]


class ProviderChart(StrictModel):
    id: str
    type: Literal["bar", "horizontal_bar", "donut", "range", "line", "price_line", "waterfall", "gauge", "metric_cards", "table"]
    title: str | None
    series: list[ProviderChartPoint] = Field(min_length=1)
    precision: int = Field(ge=0, le=4)
    compact: bool
    animation: Literal["grow", "sweep", "pan-and-highlight", "draw-and-highlight"]


class ProviderCaptionRegion(StrictModel):
    x: float
    y: float
    width: float
    height: float


class ProviderChartSpec(StrictModel):
    caption_region: ProviderCaptionRegion
    charts: list[ProviderChart]


class ProviderPlanningBundle(StrictModel):
    """OpenAI-facing contract that is valid with Structured Outputs strict mode."""

    canonical_facts: ProviderCanonicalFacts
    financial_analysis: ProviderFinancialAnalysis
    story_plan: ProviderStoryPlan
    scene_plan: list[ProviderScene] = Field(min_length=1, max_length=12)
    narration: ProviderNarration
    chart_spec: ProviderChartSpec

    def to_domain(self) -> PlanningBundle:
        """Normalize facts and compile all numeric references into domain artifacts."""
        from .facts import compile_provider_bundle
        return compile_provider_bundle(self)


class JobRecord(StrictModel):
    """Persisted job state returned to clients."""

    id: str
    status: JobStatus
    current_node: str | None = None
    progress: float = Field(default=0, ge=0, le=1)
    retries: int = 0
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    nodes: dict[str, str] = Field(default_factory=dict)


class ReviewRequest(StrictModel):
    """Reviewer decision and optional note."""

    note: str = ""


class RetryRequest(StrictModel):
    """Scene retry reason and supported scene overrides."""

    reason: str = "manual retry"
    overrides: dict[str, Any] = Field(default_factory=dict)
