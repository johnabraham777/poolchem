"""
Pool Chemistry Calculator — REST API
=====================================
FastAPI wrapper for the pool water chemistry calculator. Exposes analysis,
chart generation, and ideal ranges as HTTP endpoints.

Run with: python api.py  or  uvicorn api:app --host 0.0.0.0 --port 8000
Docs at: http://localhost:8000/docs
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Optional

# Configure logging from env (e.g. LOG_LEVEL=DEBUG) for deployment visibility
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless servers

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, model_validator

from pool_chemistry_calculator import (
    PoolProfile,
    PoolType,
    SurfaceType,
    TestResults,
    AnalysisReport,
    Recommendation,
    analyse_pool,
    get_default_ranges,
    plot_test_results,
)


# =============================================================================
# SECTION 1: PYDANTIC MODELS (INPUT/OUTPUT SCHEMAS)
# =============================================================================


class PoolProfileInput(BaseModel):
    """Pool physical and configuration properties (API input)."""

    volume_litres: float = Field(
        ...,
        gt=0,
        le=10_000_000,
        description="Pool water volume in litres (L).",
    )
    pool_type: str = Field(
        ...,
        description="Sanitisation method: 'traditional' or 'saltwater'.",
    )
    surface: str = Field(
        ...,
        description="Interior surface: 'plaster', 'vinyl', or 'fibreglass'.",
    )
    temperature_c: float = Field(
        default=28.0,
        ge=0,
        le=50,
        description="Water temperature in degrees Celsius (°C).",
    )

    @model_validator(mode="after")
    def validate_pool_type_and_surface(self) -> "PoolProfileInput":
        if self.pool_type not in ("traditional", "saltwater"):
            raise ValueError("pool_type must be 'traditional' or 'saltwater'")
        if self.surface not in ("plaster", "vinyl", "fibreglass"):
            raise ValueError("surface must be 'plaster', 'vinyl', or 'fibreglass'")
        return self


class TestResultsInput(BaseModel):
    """Chemical test readings from the user's pool test kit (API input).
    All concentrations in ppm except pH (dimensionless). At least one field required.
    """

    free_chlorine: Optional[float] = Field(default=None, ge=0, description="Free chlorine in ppm.")
    total_chlorine: Optional[float] = Field(default=None, ge=0, description="Total chlorine in ppm.")
    combined_chlorine: Optional[float] = Field(default=None, ge=0, description="Combined chlorine in ppm.")
    ph: Optional[float] = Field(default=None, ge=0.0, le=14.0, description="pH (dimensionless).")
    total_alkalinity: Optional[float] = Field(default=None, ge=0, description="Total alkalinity in ppm.")
    calcium_hardness: Optional[float] = Field(default=None, ge=0, description="Calcium hardness in ppm.")
    cyanuric_acid: Optional[float] = Field(default=None, ge=0, description="Cyanuric acid in ppm.")
    salt: Optional[float] = Field(default=None, ge=0, description="Salt in ppm.")
    total_dissolved_solids: Optional[float] = Field(default=None, ge=0, description="Total dissolved solids in ppm.")
    iron: Optional[float] = Field(default=None, ge=0, description="Iron (Fe) in ppm.")
    copper: Optional[float] = Field(default=None, ge=0, description="Copper (Cu) in ppm.")

    @model_validator(mode="after")
    def at_least_one_field(self) -> "TestResultsInput":
        provided = [
            self.free_chlorine,
            self.total_chlorine,
            self.combined_chlorine,
            self.ph,
            self.total_alkalinity,
            self.calcium_hardness,
            self.cyanuric_acid,
            self.salt,
            self.total_dissolved_solids,
            self.iron,
            self.copper,
        ]
        if all(v is None for v in provided):
            raise ValueError("At least one test result field must be provided.")
        return self

    @model_validator(mode="after")
    def total_chlorine_gte_free_chlorine(self) -> "TestResultsInput":
        if (
            self.free_chlorine is not None
            and self.total_chlorine is not None
            and self.total_chlorine < self.free_chlorine
        ):
            raise ValueError(
                "Total chlorine must be greater than or equal to free chlorine. "
                "Combined chlorine is computed as total chlorine minus free chlorine."
            )
        return self


class RecommendationOutput(BaseModel):
    """A single chemical adjustment recommendation (API output)."""

    parameter: str
    parameter_name: str
    current_value: float
    current_unit: str
    target_value: float
    ideal_range: dict[str, float]  # {"min": float, "max": float}
    chemical_name: str
    chemical_formula: str
    dose_amount: float
    dose_unit: str
    direction: str
    explanation: str
    secondary_effects: list[str]
    priority: int
    warnings: list[str]


class AnalysisReportOutput(BaseModel):
    """Complete analysis output (API response)."""

    pool: PoolProfileInput
    test_results: TestResultsInput
    recommendations: list[RecommendationOutput]
    lsi: Optional[float] = None
    lsi_interpretation: str = ""
    fc_cya_analysis: str = ""
    hocl_percent: Optional[float] = None
    corrected_alkalinity_ppm: Optional[float] = None
    summary: str = ""


class AnalyseRequest(BaseModel):
    """Request body for POST /analyse and POST /analyse/chart."""

    pool: PoolProfileInput
    test_results: TestResultsInput


# =============================================================================
# SECTION 2: CONVERSION LAYER
# =============================================================================


def pydantic_to_pool_profile(input_model: PoolProfileInput) -> PoolProfile:
    """
    Convert API input to the calculator's PoolProfile dataclass.

    Parameters
    ----------
    input_model : PoolProfileInput
        Pydantic model from request body.

    Returns
    -------
    PoolProfile
        Dataclass instance for analyse_pool().
    """
    pool_type = PoolType.TRADITIONAL if input_model.pool_type == "traditional" else PoolType.SALTWATER
    surface_map = {
        "plaster": SurfaceType.PLASTER,
        "vinyl": SurfaceType.VINYL,
        "fibreglass": SurfaceType.FIBREGLASS,
    }
    surface = surface_map[input_model.surface]
    return PoolProfile(
        volume_litres=input_model.volume_litres,
        pool_type=pool_type,
        surface=surface,
        temperature_c=input_model.temperature_c,
    )


def pydantic_to_test_results(input_model: TestResultsInput) -> TestResults:
    """
    Convert API input to the calculator's TestResults dataclass.

    Parameters
    ----------
    input_model : TestResultsInput
        Pydantic model from request body.

    Returns
    -------
    TestResults
        Dataclass instance for analyse_pool().
    """
    return TestResults(
        free_chlorine=input_model.free_chlorine,
        total_chlorine=input_model.total_chlorine,
        combined_chlorine=input_model.combined_chlorine,
        ph=input_model.ph,
        total_alkalinity=input_model.total_alkalinity,
        calcium_hardness=input_model.calcium_hardness,
        cyanuric_acid=input_model.cyanuric_acid,
        salt=input_model.salt,
        total_dissolved_solids=input_model.total_dissolved_solids,
        iron=input_model.iron,
        copper=input_model.copper,
    )


def recommendation_to_output(rec: Recommendation) -> RecommendationOutput:
    """Convert a single Recommendation dataclass to Pydantic output (ideal_range as dict)."""
    return RecommendationOutput(
        parameter=rec.parameter,
        parameter_name=rec.parameter_name,
        current_value=rec.current_value,
        current_unit=rec.current_unit,
        target_value=rec.target_value,
        ideal_range={"min": rec.ideal_range[0], "max": rec.ideal_range[1]},
        chemical_name=rec.chemical_name,
        chemical_formula=rec.chemical_formula,
        dose_amount=rec.dose_amount,
        dose_unit=rec.dose_unit,
        direction=rec.direction,
        explanation=rec.explanation,
        secondary_effects=rec.secondary_effects,
        priority=rec.priority,
        warnings=rec.warnings,
    )


def pool_profile_to_input(profile: PoolProfile) -> PoolProfileInput:
    """Convert PoolProfile back to API input shape for response embedding."""
    return PoolProfileInput(
        volume_litres=profile.volume_litres,
        pool_type=profile.pool_type.value,
        surface=profile.surface.value,
        temperature_c=profile.temperature_c,
    )


def test_results_to_input(test: TestResults) -> TestResultsInput:
    """Convert TestResults back to API input shape for response embedding."""
    return TestResultsInput(
        free_chlorine=test.free_chlorine,
        total_chlorine=test.total_chlorine,
        combined_chlorine=test.combined_chlorine,
        ph=test.ph,
        total_alkalinity=test.total_alkalinity,
        calcium_hardness=test.calcium_hardness,
        cyanuric_acid=test.cyanuric_acid,
        salt=test.salt,
        total_dissolved_solids=test.total_dissolved_solids,
        iron=test.iron,
        copper=test.copper,
    )


def analysis_report_to_output(report: AnalysisReport) -> AnalysisReportOutput:
    """
    Convert AnalysisReport from the calculator to API output model.

    Parameters
    ----------
    report : AnalysisReport
        Result of analyse_pool().

    Returns
    -------
    AnalysisReportOutput
        Pydantic model for JSON response.
    """
    return AnalysisReportOutput(
        pool=pool_profile_to_input(report.pool),
        test_results=test_results_to_input(report.test_results),
        recommendations=[recommendation_to_output(rec) for rec in report.recommendations],
        lsi=report.lsi,
        lsi_interpretation=report.lsi_interpretation,
        fc_cya_analysis=report.fc_cya_analysis,
        hocl_percent=report.hocl_percent,
        corrected_alkalinity_ppm=report.corrected_alkalinity_ppm,
        summary=report.summary,
    )


# =============================================================================
# SECTION 3: FASTAPI APP & CORS
# =============================================================================

app = FastAPI(
    title="Pool Chemistry Calculator API",
    description=(
        "REST API for pool water chemistry analysis. Submit pool profile and test results "
        "to receive prioritised chemical adjustment recommendations, LSI, FC/CYA analysis, "
        "and optional PNG chart of results vs ideal ranges."
    ),
    version="1.0.0",
)


@app.on_event("startup")
def startup() -> None:
    log.info("Pool Chemistry Calculator API started")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# =============================================================================
# SECTION 4: ERROR HANDLING
# =============================================================================


@app.exception_handler(ValueError)
async def value_error_handler(request: Any, exc: ValueError) -> Any:
    """Return 422 for validation errors (e.g. total_chlorine < free_chlorine, invalid enum)."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "ValueError"},
    )


@app.exception_handler(TypeError)
async def type_error_handler(request: Any, exc: TypeError) -> Any:
    """Return 422 for type-related errors."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "TypeError"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Any, exc: Exception) -> Any:
    """Return 500 for unexpected internal errors."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred.", "type": type(exc).__name__},
    )


# =============================================================================
# SECTION 5: ENDPOINTS
# =============================================================================


@app.get(
    "/health",
    summary="Health check",
    description="Simple health check endpoint. Returns status ok.",
    tags=["System"],
)
async def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


@app.get(
    "/ranges",
    summary="Get ideal ranges",
    description=(
        "Return the default ideal parameter ranges (min, max, target) for a given "
        "pool type and surface. Useful for frontend range indicators before submitting analysis."
    ),
    tags=["Reference"],
)
async def ranges(
    pool_type: str = Query(..., description="'traditional' or 'saltwater'"),
    surface: str = Query(..., description="'plaster', 'vinyl', or 'fibreglass'"),
) -> dict[str, dict[str, float]]:
    """
    Return ideal ranges as JSON: each parameter maps to {"min": float, "max": float, "target": float}.
    """
    if pool_type not in ("traditional", "saltwater"):
        raise HTTPException(
            status_code=422,
            detail="pool_type must be 'traditional' or 'saltwater'",
        )
    if surface not in ("plaster", "vinyl", "fibreglass"):
        raise HTTPException(
            status_code=422,
            detail="surface must be 'plaster', 'vinyl', or 'fibreglass'",
        )
    pool_type_enum = PoolType.TRADITIONAL if pool_type == "traditional" else PoolType.SALTWATER
    surface_map = {
        "plaster": SurfaceType.PLASTER,
        "vinyl": SurfaceType.VINYL,
        "fibreglass": SurfaceType.FIBREGLASS,
    }
    surface_enum = surface_map[surface]
    raw_ranges = get_default_ranges(pool_type_enum, surface_enum)
    return {
        key: {"min": t[0], "max": t[1], "target": t[2]}
        for key, t in raw_ranges.items()
    }


@app.post(
    "/analyse",
    response_model=AnalysisReportOutput,
    summary="Analyse pool chemistry",
    description=(
        "Submit pool profile and test results. Returns full analysis: recommendations, "
        "Langelier Saturation Index, FC/CYA analysis, HOCl percentage, and summary."
    ),
    tags=["Analysis"],
)
async def analyse(body: AnalyseRequest) -> AnalysisReportOutput:
    """
    Run the pool chemistry analysis and return the full report as JSON.
    """
    pool = pydantic_to_pool_profile(body.pool)
    test = pydantic_to_test_results(body.test_results)
    report = analyse_pool(pool, test)
    return analysis_report_to_output(report)


@app.post(
    "/analyse/chart",
    summary="Generate analysis chart",
    description=(
        "Same input as /analyse. Generates a matplotlib PNG chart of test results "
        "vs ideal ranges and returns it as image/png."
    ),
    tags=["Analysis"],
)
async def analyse_chart(body: AnalyseRequest) -> StreamingResponse:
    """
    Generate the test-results-vs-ranges chart as PNG. Uses a temporary buffer (no permanent file).
    """
    pool = pydantic_to_pool_profile(body.pool)
    test = pydantic_to_test_results(body.test_results)
    fig = plot_test_results(pool, test, save_path=None, figsize=(14, 9))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buffer.seek(0)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return StreamingResponse(buffer, media_type="image/png")


# =============================================================================
# SECTION 6: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
