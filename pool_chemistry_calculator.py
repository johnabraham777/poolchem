"""
Pool Chemistry Calculator — Core Engine
=========================================
A stoichiometry-based pool water chemistry calculator that analyses test results
and recommends chemical adjustments to reach optimal ranges.

Supports both traditional chlorine pools and saltwater chlorinator (SWG) pools.
All units are metric:
    - Concentrations in parts per million (ppm), equivalent to mg/L
    - Volumes in litres (L)
    - Masses in grams (g) or kilograms (kg)
    - Liquid doses in millilitres (mL)
    - Temperature in degrees Celsius (°C)
    - pH is dimensionless

Chemistry references:
    - PHTA (Pool & Hot Tub Alliance) fact sheets
    - Journal of the Swimming Pool and Spa Industry (JSPSI) dosage tables
    - Trouble Free Pool (TFP) FC/CYA relationship model
    - Langelier Saturation Index (LSI) per ASTM D3739

Author: Pool Chemistry Calculator Project
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# =============================================================================
# SECTION 1: CONSTANTS & CONFIGURATION
# =============================================================================

class PoolType(Enum):
    """Pool sanitisation method."""
    TRADITIONAL = "traditional"   # Manual chlorine dosing (liquid, cal-hypo, trichlor)
    SALTWATER = "saltwater"       # Salt water chlorine generator (SWG)


class SurfaceType(Enum):
    """Pool interior surface — affects calcium hardness targets."""
    PLASTER = "plaster"           # Concrete/plaster/pebble finish
    VINYL = "vinyl"               # Vinyl liner
    FIBREGLASS = "fibreglass"     # Fibreglass shell


# ---------------------------------------------------------------------------
# Molecular weights in grams per mole (g/mol)
# ---------------------------------------------------------------------------
MW = {
    "NaCl": 58.44,          # g/mol — Sodium chloride
    "NaHCO3": 84.01,        # g/mol — Sodium bicarbonate
    "NaOCl": 74.44,         # g/mol — Sodium hypochlorite
    "Ca(ClO)2": 142.98,     # g/mol — Calcium hypochlorite
    "HCl": 36.46,           # g/mol — Hydrochloric acid
    "CaCO3": 100.09,        # g/mol — Calcium carbonate (alkalinity reference)
    "Cl2": 70.91,           # g/mol — Molecular chlorine (available chlorine reference)
    "Ca": 40.08,            # g/mol — Calcium atom
    "CaCl2": 110.98,        # g/mol — Calcium chloride
}

# ---------------------------------------------------------------------------
# Chemical product concentrations (as-sold, dimensionless fractions)
# ---------------------------------------------------------------------------
CHEMICAL_DEFAULTS = {
    "NaCl_purity": 0.998,                  # fraction — Pool-grade salt, ≥ 99.8% pure
    "NaHCO3_purity": 1.0,                  # fraction — Bicarb soda, essentially pure
    "NaOCl_concentration": 0.125,           # fraction — 12.5% available chlorine
    "NaOCl_density_kg_per_L": 1.16,         # kg/L — Density of 12.5% NaOCl solution
    "CaClO2_available_chlorine": 0.67,      # fraction — 67% available chlorine
    "HCl_concentration": 0.3145,            # fraction — 31.45% HCl (pool-grade muriatic acid)
    "HCl_density_kg_per_L": 1.16,           # kg/L — Density of 31.45% HCl solution
}

# ---------------------------------------------------------------------------
# Parameter metadata: used for display, plotting, and range lookup
# ---------------------------------------------------------------------------
PARAMETER_KEYS = [
    "free_chlorine",
    "combined_chlorine",
    "ph",
    "total_alkalinity",
    "calcium_hardness",
    "cyanuric_acid",
    "salt",
    "total_dissolved_solids",
    "iron",
    "copper",
]

PARAMETER_DISPLAY = {
    "free_chlorine":          ("Free Chlorine",          "ppm"),
    "combined_chlorine":      ("Combined Chlorine",      "ppm"),
    "ph":                     ("pH",                     ""),      # dimensionless
    "total_alkalinity":       ("Total Alkalinity",       "ppm"),
    "calcium_hardness":       ("Calcium Hardness",       "ppm"),
    "cyanuric_acid":          ("Cyanuric Acid",          "ppm"),
    "salt":                   ("Salt",                   "ppm"),
    "total_dissolved_solids": ("Total Dissolved Solids",  "ppm"),
    "iron":                   ("Iron (Fe)",              "ppm"),
    "copper":                 ("Copper (Cu)",            "ppm"),
}


def get_default_ranges(pool_type: PoolType, surface: SurfaceType) -> dict:
    """
    Return the recommended parameter ranges for a given pool configuration.

    Each entry is a tuple: (minimum, maximum, target_midpoint)
    All concentration values are in ppm.  pH is dimensionless.

    Free chlorine targets are CYA-dependent and computed dynamically by
    fc_target_from_cya(), so the range here is a fallback for CYA = 0.
    """
    # --- Calcium hardness varies by surface (ppm) ---
    if surface == SurfaceType.PLASTER:
        calcium_hardness_range = (250.0, 450.0, 350.0)   # ppm
    elif surface == SurfaceType.VINYL:
        calcium_hardness_range = (75.0, 300.0, 175.0)    # ppm
    else:  # fibreglass
        calcium_hardness_range = (75.0, 300.0, 175.0)    # ppm

    # --- Base ranges (traditional chlorine pool) ---
    ranges = {
        "free_chlorine":          (1.0,    5.0,    3.0),      # ppm
        "combined_chlorine":      (0.0,    0.5,    0.0),      # ppm
        "ph":                     (7.2,    7.8,    7.5),      # dimensionless
        "total_alkalinity":       (80.0,   120.0,  100.0),    # ppm
        "calcium_hardness":       calcium_hardness_range,      # ppm
        "cyanuric_acid":          (30.0,   60.0,   40.0),     # ppm
        "salt":                   (0.0,    2000.0, 0.0),      # ppm — non-SWG: salt optional
        "total_dissolved_solids": (0.0,    2000.0, 1000.0),   # ppm
        "iron":                   (0.0,    0.2,    0.0),      # ppm
        "copper":                 (0.0,    0.2,    0.0),      # ppm
    }

    # --- Saltwater pool overrides ---
    if pool_type == PoolType.SALTWATER:
        ranges["total_alkalinity"] = (60.0,  80.0,  70.0)    # ppm
        ranges["cyanuric_acid"]    = (60.0,  90.0,  70.0)    # ppm
        ranges["salt"]             = (2700.0, 3400.0, 3200.0) # ppm
        ranges["total_dissolved_solids"] = (3000.0, 6000.0, 4500.0)  # ppm — higher due to salt

    return ranges


# =============================================================================
# SECTION 2: FREE CHLORINE / CYANURIC ACID RELATIONSHIP
# =============================================================================

def fc_target_from_cya(
    cyanuric_acid_ppm: float,   # ppm
    pool_type: PoolType,
) -> tuple[float, float, float]:
    """
    Calculate the minimum, target, and SLAM (superchlorination) free chlorine
    levels based on cyanuric acid concentration, using the TFP model.

    The minimum free chlorine is 7.5% of CYA for traditional pools, 5% for SWG.
    The SLAM level is 40% of CYA.

    Parameters
    ----------
    cyanuric_acid_ppm : float
        Cyanuric acid concentration in ppm.
    pool_type : PoolType
        TRADITIONAL or SALTWATER.

    Returns
    -------
    (min_fc_ppm, target_fc_ppm, slam_fc_ppm) : tuple of floats, all in ppm.
    """
    if cyanuric_acid_ppm <= 0:
        return (1.0, 3.0, 10.0)   # ppm — unstabilised pool defaults

    if pool_type == PoolType.SALTWATER:
        min_fc = cyanuric_acid_ppm * 0.05       # ppm
    else:
        min_fc = cyanuric_acid_ppm * 0.075      # ppm

    # Target: ~10–12% of CYA, at least 1 ppm above minimum
    target_fc = cyanuric_acid_ppm * (0.10 if pool_type == PoolType.SALTWATER else 0.12)  # ppm
    target_fc = max(target_fc, min_fc + 1.0)    # ppm

    slam_fc = cyanuric_acid_ppm * 0.40          # ppm

    # Apply practical floors
    min_fc = max(min_fc, 1.0)       # ppm
    target_fc = max(target_fc, 2.0) # ppm
    slam_fc = max(slam_fc, 10.0)    # ppm

    return (round(min_fc, 1), round(target_fc, 1), round(slam_fc, 1))


def percent_hocl(
    ph: float,                       # dimensionless
    cyanuric_acid_ppm: float = 0.0,  # ppm
) -> float:
    """
    Estimate the percentage of measured free chlorine that exists as
    active hypochlorous acid (HOCl).

    Without CYA, governed by: HOCl ⇌ H⁺ + OCl⁻, pKa ≈ 7.54
    With CYA, the vast majority of free chlorine is bound in an inactive
    chlorinated isocyanurate reservoir.

    Returns
    -------
    float   Percentage of free chlorine that is active HOCl (0–100), dimensionless.
    """
    pka = 7.54  # dimensionless — HOCl dissociation constant at 25 °C
    fraction_hocl_no_cya = 1.0 / (1.0 + 10 ** (ph - pka))  # dimensionless

    if cyanuric_acid_ppm <= 0:
        return round(fraction_hocl_no_cya * 100, 2)  # percent

    # Empirical CYA binding model derived from O'Brien et al. (1974)
    K = 0.035 * (1.0 + 10 ** (ph - pka))  # L/mg — pH-dependent binding constant
    fraction_with_cya = fraction_hocl_no_cya / (1.0 + K * cyanuric_acid_ppm)  # dimensionless

    return round(fraction_with_cya * 100, 2)  # percent


# =============================================================================
# SECTION 3: LANGELIER SATURATION INDEX (LSI)
# =============================================================================

def _temperature_factor(temp_c: float) -> float:
    """
    LSI temperature factor from standard lookup table, linearly interpolated.

    Parameters
    ----------
    temp_c : float   Water temperature in °C.

    Returns
    -------
    float   Temperature factor (dimensionless).
    """
    # Table: (temperature in °C, factor — dimensionless)
    table = [
        (0.0, 0.0), (4.4, 0.1), (7.8, 0.2), (11.7, 0.3),
        (15.6, 0.4), (18.9, 0.5), (24.4, 0.6), (28.9, 0.7),
        (34.4, 0.8), (40.6, 0.9), (52.8, 1.0),
    ]
    if temp_c <= table[0][0]:
        return table[0][1]
    if temp_c >= table[-1][0]:
        return table[-1][1]
    for i in range(len(table) - 1):
        t0, f0 = table[i]
        t1, f1 = table[i + 1]
        if t0 <= temp_c <= t1:
            return f0 + (f1 - f0) * (temp_c - t0) / (t1 - t0)
    return 0.6  # fallback


def corrected_alkalinity(
    total_alkalinity_ppm: float,   # ppm
    cyanuric_acid_ppm: float,      # ppm
    ph: float,                     # dimensionless
) -> float:
    """
    Compute carbonate alkalinity by subtracting the CYA contribution from
    the measured total alkalinity reading.

    The correction factor varies with pH:
        pH 7.0 → 0.22,  pH 7.5 → 0.33,  pH 8.0 → 0.38

    Returns
    -------
    float   Corrected (carbonate) alkalinity in ppm.
    """
    if ph <= 7.0:
        factor = 0.22   # dimensionless
    elif ph >= 8.0:
        factor = 0.38   # dimensionless
    else:
        factor = 0.22 + (0.38 - 0.22) * (ph - 7.0) / (8.0 - 7.0)  # dimensionless

    corrected = total_alkalinity_ppm - (cyanuric_acid_ppm * factor)  # ppm
    return max(corrected, 1.0)  # ppm — floor at 1 to avoid log(0)


def calculate_lsi(
    ph: float,                      # dimensionless
    temp_c: float,                  # °C
    calcium_hardness_ppm: float,    # ppm
    total_alkalinity_ppm: float,    # ppm
    cyanuric_acid_ppm: float = 0.0, # ppm
    tds_ppm: float = 1000.0,       # ppm
) -> float:
    """
    Calculate the Langelier Saturation Index.

    LSI = pH + TF + CF + AF − TDSF

    where:
        TF   = temperature factor (dimensionless, from lookup table)
        CF   = log₁₀(calcium hardness in ppm)
        AF   = log₁₀(corrected alkalinity in ppm)
        TDSF = TDS constant (12.1 for < 1000 ppm TDS, 12.2 for saltwater)

    Returns
    -------
    float   LSI value (dimensionless).  Ideal range is −0.3 to +0.3.
    """
    tf = _temperature_factor(temp_c)                                          # dimensionless
    cf = math.log10(max(calcium_hardness_ppm, 1.0))                           # dimensionless
    corr_alk = corrected_alkalinity(total_alkalinity_ppm, cyanuric_acid_ppm, ph)  # ppm
    af = math.log10(max(corr_alk, 1.0))                                       # dimensionless
    tdsf = 12.2 if tds_ppm > 2000 else 12.1                                  # dimensionless

    lsi = ph + tf + cf + af - tdsf  # dimensionless
    return round(lsi, 2)


def interpret_lsi(lsi: float) -> str:
    """Human-readable interpretation of an LSI value."""
    if lsi < -0.5:
        return "SEVERELY CORROSIVE — water will aggressively etch plaster, corrode metals, and dissolve grout."
    elif lsi < -0.3:
        return "CORROSIVE — water is under-saturated and may damage surfaces and equipment over time."
    elif lsi <= 0.3:
        return "BALANCED — water is in the ideal range. No scaling or corrosion tendency."
    elif lsi <= 0.5:
        return "SLIGHTLY SCALING — water is mildly over-saturated. Light calcium deposits may form."
    else:
        return "SCALING — water will deposit calcium carbonate on surfaces, heat exchangers, and salt cells."


# =============================================================================
# SECTION 4: DOSAGE CALCULATIONS
# =============================================================================
# Core conversion: 1 ppm in 1 litre = 1 mg.  So 1 ppm in 10,000 L = 10 g.
# All dosage functions return grams (g) or millilitres (mL) of PRODUCT.
# =============================================================================

def dose_salt(
    pool_volume_litres: float,   # L
    ppm_increase: float,         # ppm
    purity: float = CHEMICAL_DEFAULTS["NaCl_purity"],  # fraction
) -> float:
    """
    Grams of pool-grade NaCl to raise salt by the desired amount.

    Stoichiometry: NaCl dissolves completely.
    1 ppm in 10,000 L = 10 g pure NaCl.

    Returns
    -------
    float   Mass of NaCl product in grams (g).
    """
    grams_pure = ppm_increase * pool_volume_litres / 1000.0  # g
    return round(grams_pure / purity, 1)  # g


def dose_sodium_bicarbonate(
    pool_volume_litres: float,   # L
    ta_increase_ppm: float,      # ppm
    purity: float = CHEMICAL_DEFAULTS["NaHCO3_purity"],  # fraction
) -> float:
    """
    Grams of sodium bicarbonate (NaHCO₃) to raise total alkalinity.

    Stoichiometry:
        Alkalinity is measured as CaCO₃ equivalents.
        CaCO₃ is diprotic → 2 equivalents per mole.
        NaHCO₃ provides 1 equivalent per mole.
        ∴ 2 mol NaHCO₃ per 1 mol CaCO₃

    Secondary effects: raises pH by ~0.1–0.2 per 10 ppm total alkalinity increase.

    Returns
    -------
    float   Mass of NaHCO₃ product in grams (g).
    """
    grams_caco3_eq = ta_increase_ppm * pool_volume_litres / 1000.0   # g
    moles_caco3 = grams_caco3_eq / MW["CaCO3"]                       # mol
    moles_nahco3 = moles_caco3 * 2.0                                  # mol
    grams_nahco3 = moles_nahco3 * MW["NaHCO3"]                        # g
    return round(grams_nahco3 / purity, 1)  # g


def dose_sodium_hypochlorite(
    pool_volume_litres: float,   # L
    fc_increase_ppm: float,      # ppm
    concentration: float = CHEMICAL_DEFAULTS["NaOCl_concentration"],        # fraction
    density: float = CHEMICAL_DEFAULTS["NaOCl_density_kg_per_L"],           # kg/L
) -> float:
    """
    Millilitres of liquid chlorine (NaOCl solution) to raise free chlorine.

    Stoichiometry:
        12.5% NaOCl solution, density 1.16 kg/L.
        Available Cl₂ per litre = 0.125 × 1.16 × 1000 = 145 g/L.
        For +1 ppm in 10,000 L: need 10 g Cl₂ → 10/145 = 69 mL.

    Secondary effects: net pH change ≈ 0 over a full chlorine consumption cycle.

    Returns
    -------
    float   Volume of NaOCl solution in millilitres (mL).
    """
    grams_cl2_needed = fc_increase_ppm * pool_volume_litres / 1000.0  # g
    available_cl2_per_ml = concentration * density   # g/mL (since density is kg/L = g/mL)
    ml_needed = grams_cl2_needed / available_cl2_per_ml  # mL
    return round(ml_needed, 1)  # mL


def dose_calcium_hypochlorite(
    pool_volume_litres: float,   # L
    fc_increase_ppm: float,      # ppm
    available_chlorine: float = CHEMICAL_DEFAULTS["CaClO2_available_chlorine"],  # fraction
) -> tuple[float, float]:
    """
    Grams of calcium hypochlorite to raise free chlorine.

    Stoichiometry:
        67% available chlorine product.
        For +1 ppm in 10,000 L: need 10 g Cl₂ → 10/0.67 = 14.9 g.

    Secondary effects:
        Raises calcium hardness by ~0.8 ppm per 1 ppm free chlorine added.
        Raises pH temporarily (solution pH ~11–12).

    Returns
    -------
    (grams_product, calcium_hardness_increase_ppm) : tuple
        grams_product : float — mass of cal-hypo in grams (g).
        calcium_hardness_increase_ppm : float — estimated calcium hardness increase in ppm.
    """
    grams_cl2_needed = fc_increase_ppm * pool_volume_litres / 1000.0  # g
    grams_product = grams_cl2_needed / available_chlorine              # g
    calcium_hardness_increase = fc_increase_ppm * 0.8                  # ppm
    return (round(grams_product, 1), round(calcium_hardness_increase, 1))


def dose_muriatic_acid(
    pool_volume_litres: float,   # L
    ta_decrease_ppm: float,      # ppm
    concentration: float = CHEMICAL_DEFAULTS["HCl_concentration"],    # fraction
    density: float = CHEMICAL_DEFAULTS["HCl_density_kg_per_L"],       # kg/L
) -> float:
    """
    Millilitres of muriatic acid (31.45% HCl) to reduce total alkalinity.

    Stoichiometry:
        HCO₃⁻ + H⁺ → H₂O + CO₂↑
        2 mol HCl per 1 mol CaCO₃ equivalent.

        For −10 ppm in 10,000 L:
        100 g CaCO₃ eq → 0.999 mol → × 2 × 36.46 = 72.85 g pure HCl
        / 0.3145 = 231.6 g solution / 1.16 g/mL = 200 mL.

    Secondary effects: lowers pH significantly (non-linear).

    Returns
    -------
    float   Volume of 31.45% muriatic acid in millilitres (mL).
    """
    grams_caco3_eq = ta_decrease_ppm * pool_volume_litres / 1000.0   # g
    moles_caco3 = grams_caco3_eq / MW["CaCO3"]                       # mol
    moles_hcl = moles_caco3 * 2.0                                     # mol
    grams_hcl_pure = moles_hcl * MW["HCl"]                            # g
    grams_solution = grams_hcl_pure / concentration                    # g
    ml_solution = grams_solution / density                             # mL (density kg/L = g/mL)
    return round(ml_solution, 1)  # mL


def dose_muriatic_acid_for_ph(
    pool_volume_litres: float,       # L
    ph_current: float,               # dimensionless
    ph_target: float,                # dimensionless
    total_alkalinity_ppm: float,     # ppm
    cyanuric_acid_ppm: float = 0.0,  # ppm
) -> float:
    """
    Estimate millilitres of muriatic acid to lower pH.

    Uses an empirical model calibrated against JSPSI dosage tables:
        For each 0.1 pH unit decrease in a 10,000 L pool:
            base_dose ≈ 40 mL per 100 ppm effective buffer of 31.45% HCl.

    Returns
    -------
    float   Volume of 31.45% muriatic acid in millilitres (mL).
    """
    if ph_target >= ph_current:
        return 0.0  # mL — no acid needed

    ph_drop = ph_current - ph_target                  # dimensionless
    ph_steps = ph_drop / 0.1                           # count of 0.1 pH steps

    # Effective buffer capacity including CYA contribution (ppm)
    effective_buffer_ppm = total_alkalinity_ppm + (cyanuric_acid_ppm * 0.33)  # ppm

    # Base dose: ~40 mL per 0.1 pH per 100 ppm buffer per 10,000 L
    base_dose_per_step = 40.0 * (effective_buffer_ppm / 100.0) * (pool_volume_litres / 10000.0)  # mL

    total_ml = ph_steps * base_dose_per_step  # mL
    return round(total_ml, 1)  # mL


# =============================================================================
# SECTION 5: DATA STRUCTURES
# =============================================================================

@dataclass
class PoolProfile:
    """Physical and configuration properties of the pool."""
    volume_litres: float                        # L — pool water volume
    pool_type: PoolType = PoolType.TRADITIONAL  # sanitisation method
    surface: SurfaceType = SurfaceType.PLASTER  # interior surface type
    temperature_c: float = 28.0                 # °C — water temperature


@dataclass
class TestResults:
    """
    Chemical test readings from the user's pool test kit.
    All concentrations in ppm (mg/L) except pH which is dimensionless.
    Set a value to None if not tested.
    """
    free_chlorine: Optional[float] = None           # ppm
    total_chlorine: Optional[float] = None          # ppm
    combined_chlorine: Optional[float] = None       # ppm (can be derived: total − free)
    ph: Optional[float] = None                      # dimensionless
    total_alkalinity: Optional[float] = None        # ppm
    calcium_hardness: Optional[float] = None        # ppm
    cyanuric_acid: Optional[float] = None           # ppm
    salt: Optional[float] = None                    # ppm
    total_dissolved_solids: Optional[float] = None  # ppm
    iron: Optional[float] = None                    # ppm
    copper: Optional[float] = None                  # ppm


@dataclass
class Recommendation:
    """A single chemical adjustment recommendation."""
    parameter: str              # internal key (e.g. "free_chlorine")
    parameter_name: str         # human-readable (e.g. "Free Chlorine")
    current_value: float        # current measured value (ppm or dimensionless)
    current_unit: str           # unit string (e.g. "ppm" or "")
    target_value: float         # target value
    ideal_range: tuple          # (min, max)
    chemical_name: str          # chemical to add
    chemical_formula: str       # formula
    dose_amount: float          # amount to add
    dose_unit: str              # "g" or "mL" or ""
    direction: str              # "increase" or "decrease"
    explanation: str            # detailed explanation
    secondary_effects: list[str] = field(default_factory=list)
    priority: int = 5           # 1 = highest, 10 = lowest
    warnings: list[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    """Complete analysis output."""
    pool: PoolProfile
    test_results: TestResults
    recommendations: list[Recommendation]
    lsi: Optional[float] = None                # dimensionless
    lsi_interpretation: str = ""
    fc_cya_analysis: str = ""
    hocl_percent: Optional[float] = None       # percent (0–100)
    corrected_alkalinity_ppm: Optional[float] = None   # ppm
    summary: str = ""


# =============================================================================
# SECTION 6: RECOMMENDATION ENGINE
# =============================================================================

def _derive_combined_chlorine(test: TestResults) -> Optional[float]:
    """Derive combined chlorine (ppm) from free and total chlorine if not directly measured."""
    if test.combined_chlorine is not None:
        return test.combined_chlorine  # ppm
    if test.free_chlorine is not None and test.total_chlorine is not None:
        return max(test.total_chlorine - test.free_chlorine, 0.0)  # ppm
    return None


def _get_test_value(test: TestResults, key: str) -> Optional[float]:
    """Retrieve a test result value by parameter key."""
    return getattr(test, key, None)


def analyse_pool(pool: PoolProfile, test: TestResults) -> AnalysisReport:
    """
    Main entry point: analyse test results and generate prioritised recommendations.

    Processing priority:
        1. Combined chlorine — chloramine alert, needs immediate action
        2. pH — affects everything else
        3. Total alkalinity — pH buffer
        4. Free chlorine — sanitation (CYA-dependent)
        5. Calcium hardness — structural protection
        6. Cyanuric acid — UV protection
        7. Salt — SWG operation
        8. Total dissolved solids — water freshness
        9. Iron — staining risk
       10. Copper — staining risk
    """
    ranges = get_default_ranges(pool.pool_type, pool.surface)
    recommendations: list[Recommendation] = []
    vol = pool.volume_litres  # L

    # Derive combined chlorine if possible
    combined_chlorine_ppm = _derive_combined_chlorine(test)

    # Current CYA for FC/CYA calculations
    cyanuric_acid_ppm = test.cyanuric_acid if test.cyanuric_acid is not None else 0.0  # ppm

    # CYA-dependent free chlorine targets (all in ppm)
    fc_min, fc_target, fc_slam = fc_target_from_cya(cyanuric_acid_ppm, pool.pool_type)
    fc_range_max = min(fc_slam * 0.5, fc_target + 4.0)  # ppm — reasonable daily max

    # =====================================================================
    # CHECK 1: COMBINED CHLORINE (ppm)
    # =====================================================================
    if combined_chlorine_ppm is not None:
        cc_min, cc_max, cc_target = ranges["combined_chlorine"]  # ppm
        if combined_chlorine_ppm > cc_max:
            fc_current = test.free_chlorine if test.free_chlorine is not None else 0.0  # ppm
            fc_boost_needed = fc_slam - fc_current  # ppm

            if fc_boost_needed > 0:
                naocl_ml = dose_sodium_hypochlorite(vol, fc_boost_needed)  # mL
                recommendations.append(Recommendation(
                    parameter="combined_chlorine",
                    parameter_name="Combined Chlorine",
                    current_value=round(combined_chlorine_ppm, 1),
                    current_unit="ppm",
                    target_value=0.0,
                    ideal_range=(cc_min, cc_max),
                    chemical_name="Sodium Hypochlorite (liquid chlorine) — SLAM Process",
                    chemical_formula="NaOCl 12.5%",
                    dose_amount=naocl_ml,
                    dose_unit="mL",
                    direction="increase free chlorine to SLAM level",
                    explanation=(
                        f"Combined chlorine of {combined_chlorine_ppm:.1f} ppm indicates chloramines "
                        f"are present (acceptable limit is {cc_max:.1f} ppm). Chloramines cause the "
                        f"'chlorine smell' and eye/skin irritation. The SLAM (Shock Level And Maintain) "
                        f"process requires raising free chlorine to {fc_slam:.0f} ppm "
                        f"(40% of cyanuric acid = {cyanuric_acid_ppm:.0f} ppm) and maintaining it "
                        f"there until combined chlorine drops to 0 ppm. "
                        f"Add {naocl_ml:,.0f} mL of 12.5% liquid chlorine to reach SLAM level. "
                        f"Re-test and re-dose every few hours to maintain the SLAM free chlorine level."
                    ),
                    secondary_effects=[
                        "Temporary pH increase until chlorine is consumed",
                        "Pool may be cloudy during SLAM — this is normal",
                    ],
                    priority=1,
                    warnings=[
                        "Do not swim until free chlorine drops below 10 ppm",
                        "Run pump 24/7 during SLAM process",
                        "Brush pool surfaces daily during SLAM",
                    ],
                ))

    # =====================================================================
    # CHECK 2: pH (dimensionless)
    # =====================================================================
    if test.ph is not None:
        ph_min, ph_max, ph_target = ranges["ph"]  # dimensionless
        ta_val = test.total_alkalinity if test.total_alkalinity is not None else 100.0  # ppm

        if test.ph < ph_min:
            recommendations.append(Recommendation(
                parameter="ph",
                parameter_name="pH",
                current_value=test.ph,
                current_unit="",
                target_value=ph_target,
                ideal_range=(ph_min, ph_max),
                chemical_name="Aeration (run water features, fountains, point returns upward)",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="increase",
                explanation=(
                    f"pH of {test.ph:.2f} is below the minimum of {ph_min:.1f}. "
                    f"Low pH causes corrosion of metal fittings, etching of plaster surfaces, "
                    f"and skin/eye irritation. The best way to raise pH without affecting total "
                    f"alkalinity is aeration — running water features, pointing return jets upward, "
                    f"or using a spa spillover. This drives off dissolved CO₂, which naturally "
                    f"raises pH. If total alkalinity is also low, sodium bicarbonate will raise both."
                ),
                secondary_effects=[
                    "Aeration raises pH only — no effect on total alkalinity, calcium hardness, or free chlorine",
                ],
                priority=2,
            ))

        elif test.ph > ph_max:
            acid_ml = dose_muriatic_acid_for_ph(vol, test.ph, ph_target, ta_val, cyanuric_acid_ppm)  # mL
            # Estimate the total alkalinity drop from this acid dose
            ta_drop_from_acid = 0.0  # ppm
            dose_for_10_ta = dose_muriatic_acid(vol, 10.0)  # mL per 10 ppm total alkalinity
            if dose_for_10_ta > 0:
                ta_drop_from_acid = (acid_ml / dose_for_10_ta) * 10.0  # ppm

            recommendations.append(Recommendation(
                parameter="ph",
                parameter_name="pH",
                current_value=test.ph,
                current_unit="",
                target_value=ph_target,
                ideal_range=(ph_min, ph_max),
                chemical_name="Hydrochloric Acid (muriatic acid)",
                chemical_formula="HCl 31.45%",
                dose_amount=acid_ml,
                dose_unit="mL",
                direction="decrease",
                explanation=(
                    f"pH of {test.ph:.2f} is above the maximum of {ph_max:.1f}. "
                    f"High pH reduces chlorine effectiveness, promotes calcium scaling, and can "
                    f"cause cloudy water. Add {acid_ml:,.0f} mL of 31.45% muriatic acid with the "
                    f"pump running. Pour slowly into the deep end, away from skimmers. "
                    f"Wait 30 minutes and re-test. The dose is estimated based on your "
                    f"total alkalinity of {ta_val:.0f} ppm — actual results may vary."
                ),
                secondary_effects=[
                    f"Will also lower total alkalinity by approximately {ta_drop_from_acid:.0f} ppm",
                    "Does not affect free chlorine, cyanuric acid, or salt levels",
                ],
                priority=2,
                warnings=[
                    "Always add acid to water, never water to acid",
                    "Wear gloves and eye protection when handling muriatic acid",
                    f"Add no more than {500 * vol / 10000:.0f} mL at a time; re-test after 30 min",
                ],
            ))

    # =====================================================================
    # CHECK 3: TOTAL ALKALINITY (ppm)
    # =====================================================================
    if test.total_alkalinity is not None:
        ta_min, ta_max, ta_target = ranges["total_alkalinity"]  # ppm

        if test.total_alkalinity < ta_min:
            ta_increase = ta_target - test.total_alkalinity  # ppm
            bicarb_g = dose_sodium_bicarbonate(vol, ta_increase)  # g
            recommendations.append(Recommendation(
                parameter="total_alkalinity",
                parameter_name="Total Alkalinity",
                current_value=test.total_alkalinity,
                current_unit="ppm",
                target_value=ta_target,
                ideal_range=(ta_min, ta_max),
                chemical_name="Sodium Bicarbonate (bicarb soda / baking soda)",
                chemical_formula="NaHCO₃",
                dose_amount=bicarb_g,
                dose_unit="g",
                direction="increase",
                explanation=(
                    f"Total alkalinity of {test.total_alkalinity:.0f} ppm is below the minimum of "
                    f"{ta_min:.0f} ppm. Low alkalinity means the water has little buffering "
                    f"capacity, so pH will swing erratically. "
                    f"Add {bicarb_g:,.0f} g of sodium bicarbonate to raise total alkalinity to "
                    f"{ta_target:.0f} ppm. Dissolve in a bucket of pool water first, then pour "
                    f"around the perimeter with the pump running. Wait 1 hour and re-test."
                ),
                secondary_effects=[
                    f"Will raise pH by approximately {ta_increase * 0.015:.1f} units",
                    "No effect on calcium hardness, cyanuric acid, or salt",
                ],
                priority=3,
            ))

        elif test.total_alkalinity > ta_max:
            ta_decrease = test.total_alkalinity - ta_target  # ppm
            acid_ml = dose_muriatic_acid(vol, ta_decrease)   # mL
            recommendations.append(Recommendation(
                parameter="total_alkalinity",
                parameter_name="Total Alkalinity",
                current_value=test.total_alkalinity,
                current_unit="ppm",
                target_value=ta_target,
                ideal_range=(ta_min, ta_max),
                chemical_name="Hydrochloric Acid (muriatic acid) + Aeration cycle",
                chemical_formula="HCl 31.45%",
                dose_amount=acid_ml,
                dose_unit="mL",
                direction="decrease",
                explanation=(
                    f"Total alkalinity of {test.total_alkalinity:.0f} ppm is above the maximum of "
                    f"{ta_max:.0f} ppm. High total alkalinity causes persistent pH rise. "
                    f"Use the acid-then-aerate technique: "
                    f"1) Add acid ({acid_ml:,.0f} mL total, in stages) — this lowers both "
                    f"total alkalinity and pH. "
                    f"2) Aerate to bring pH back up without raising total alkalinity. "
                    f"3) Repeat until total alkalinity reaches {ta_target:.0f} ppm."
                ),
                secondary_effects=[
                    "Acid will lower pH — aeration step is needed to restore pH",
                    "No effect on calcium hardness, cyanuric acid, free chlorine, or salt",
                ],
                priority=3,
                warnings=[
                    "Work in cycles — do not add all acid at once for large doses",
                    "Monitor pH closely during this process",
                ],
            ))

    # =====================================================================
    # CHECK 4: FREE CHLORINE (ppm)
    # =====================================================================
    if test.free_chlorine is not None:
        if test.free_chlorine < fc_min:
            fc_increase = fc_target - test.free_chlorine  # ppm
            calcium_hardness_val = test.calcium_hardness if test.calcium_hardness is not None else 300.0  # ppm
            calcium_hardness_min = ranges["calcium_hardness"][0]  # ppm

            if calcium_hardness_val < calcium_hardness_min:
                # Calcium is low — use calcium hypochlorite to address both
                cal_hypo_g, calcium_hardness_increase = dose_calcium_hypochlorite(vol, fc_increase)
                recommendations.append(Recommendation(
                    parameter="free_chlorine",
                    parameter_name="Free Chlorine",
                    current_value=test.free_chlorine,
                    current_unit="ppm",
                    target_value=fc_target,
                    ideal_range=(fc_min, fc_range_max),
                    chemical_name="Calcium Hypochlorite (cal-hypo)",
                    chemical_formula="Ca(ClO)₂ 67%",
                    dose_amount=cal_hypo_g,
                    dose_unit="g",
                    direction="increase",
                    explanation=(
                        f"Free chlorine of {test.free_chlorine:.1f} ppm is below the recommended "
                        f"minimum of {fc_min:.1f} ppm (based on cyanuric acid of {cyanuric_acid_ppm:.0f} ppm). "
                        f"Calcium hardness ({calcium_hardness_val:.0f} ppm) is also below target, so "
                        f"calcium hypochlorite is recommended — it raises both free chlorine and calcium "
                        f"hardness simultaneously. "
                        f"Add {cal_hypo_g:,.0f} g of 67% cal-hypo. Pre-dissolve in a bucket of "
                        f"pool water before adding."
                    ),
                    secondary_effects=[
                        f"Raises calcium hardness by approximately {calcium_hardness_increase:.1f} ppm",
                        "Temporarily raises pH (solution pH ~11–12)",
                        "Does not add cyanuric acid or salt",
                    ],
                    priority=4,
                ))
            else:
                # Calcium is fine — use sodium hypochlorite
                naocl_ml = dose_sodium_hypochlorite(vol, fc_increase)  # mL
                recommendations.append(Recommendation(
                    parameter="free_chlorine",
                    parameter_name="Free Chlorine",
                    current_value=test.free_chlorine,
                    current_unit="ppm",
                    target_value=fc_target,
                    ideal_range=(fc_min, fc_range_max),
                    chemical_name="Sodium Hypochlorite (liquid chlorine)",
                    chemical_formula="NaOCl 12.5%",
                    dose_amount=naocl_ml,
                    dose_unit="mL",
                    direction="increase",
                    explanation=(
                        f"Free chlorine of {test.free_chlorine:.1f} ppm is below the recommended "
                        f"minimum of {fc_min:.1f} ppm (based on cyanuric acid of {cyanuric_acid_ppm:.0f} ppm). "
                        f"Add {naocl_ml:,.0f} mL of 12.5% liquid chlorine. Pour with the pump running, "
                        f"ideally in the evening to minimise UV burn-off."
                    ),
                    secondary_effects=[
                        "Minimal net pH effect over a full chlorine consumption cycle",
                        "Does not add calcium, cyanuric acid, or salt",
                    ],
                    priority=4,
                ))

        elif test.free_chlorine > fc_range_max + 2.0:
            recommendations.append(Recommendation(
                parameter="free_chlorine",
                parameter_name="Free Chlorine",
                current_value=test.free_chlorine,
                current_unit="ppm",
                target_value=fc_target,
                ideal_range=(fc_min, fc_range_max),
                chemical_name="No chemical needed — wait for natural consumption",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="decrease",
                explanation=(
                    f"Free chlorine of {test.free_chlorine:.1f} ppm is above the target range "
                    f"({fc_min:.1f}–{fc_range_max:.1f} ppm). Let UV sunlight and natural chlorine "
                    f"demand consume the excess — typically 24–72 hours. "
                    f"Avoid swimming if free chlorine is above 10 ppm."
                ),
                priority=8,
            ))

    # =====================================================================
    # CHECK 5: CALCIUM HARDNESS (ppm)
    # =====================================================================
    if test.calcium_hardness is not None:
        ch_min, ch_max, ch_target = ranges["calcium_hardness"]  # ppm

        if test.calcium_hardness < ch_min:
            ch_increase = ch_target - test.calcium_hardness  # ppm
            fc_equivalent = ch_increase / 0.8  # ppm of free chlorine from cal-hypo
            cal_hypo_g, _ = dose_calcium_hypochlorite(vol, fc_equivalent)

            recommendations.append(Recommendation(
                parameter="calcium_hardness",
                parameter_name="Calcium Hardness",
                current_value=test.calcium_hardness,
                current_unit="ppm",
                target_value=ch_target,
                ideal_range=(ch_min, ch_max),
                chemical_name="Calcium Hypochlorite (or Calcium Chloride if available)",
                chemical_formula="Ca(ClO)₂ 67% / CaCl₂",
                dose_amount=cal_hypo_g,
                dose_unit="g (cal-hypo)",
                direction="increase",
                explanation=(
                    f"Calcium hardness of {test.calcium_hardness:.0f} ppm is below the minimum of "
                    f"{ch_min:.0f} ppm for your {pool.surface.value} pool. Low calcium causes water "
                    f"to dissolve calcium from surfaces. "
                    f"Calcium hypochlorite adds both chlorine and calcium, but would also raise "
                    f"free chlorine by ~{fc_equivalent:.0f} ppm. For a targeted calcium-only "
                    f"increase, use calcium chloride (CaCl₂) flakes instead."
                ),
                secondary_effects=[
                    f"Cal-hypo will raise free chlorine by approximately {fc_equivalent:.0f} ppm",
                    "Cal-hypo will temporarily raise pH",
                    "Consider timing with low free chlorine levels",
                ],
                priority=5,
                warnings=[
                    "If free chlorine is already at target, use CaCl₂ instead of cal-hypo",
                ],
            ))

        elif test.calcium_hardness > ch_max:
            recommendations.append(Recommendation(
                parameter="calcium_hardness",
                parameter_name="Calcium Hardness",
                current_value=test.calcium_hardness,
                current_unit="ppm",
                target_value=ch_target,
                ideal_range=(ch_min, ch_max),
                chemical_name="Partial drain and refill (dilution)",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="decrease",
                explanation=(
                    f"Calcium hardness of {test.calcium_hardness:.0f} ppm is above the maximum of "
                    f"{ch_max:.0f} ppm. No chemical removes dissolved calcium — only dilution works. "
                    f"Keep pH in the lower range (7.2–7.4) to reduce scaling tendency until "
                    f"a partial drain can be performed."
                ),
                priority=6,
            ))

    # =====================================================================
    # CHECK 6: CYANURIC ACID (ppm)
    # =====================================================================
    if test.cyanuric_acid is not None:
        cya_min, cya_max, cya_target = ranges["cyanuric_acid"]  # ppm

        if test.cyanuric_acid < cya_min:
            cya_dose_g = (cya_target - test.cyanuric_acid) * pool.volume_litres / 1000.0  # g
            recommendations.append(Recommendation(
                parameter="cyanuric_acid",
                parameter_name="Cyanuric Acid (Stabiliser)",
                current_value=test.cyanuric_acid,
                current_unit="ppm",
                target_value=cya_target,
                ideal_range=(cya_min, cya_max),
                chemical_name="Cyanuric Acid (stabiliser / conditioner)",
                chemical_formula="C₃H₃N₃O₃",
                dose_amount=round(cya_dose_g, 0),
                dose_unit="g",
                direction="increase",
                explanation=(
                    f"Cyanuric acid of {test.cyanuric_acid:.0f} ppm is below the minimum of "
                    f"{cya_min:.0f} ppm. Without adequate stabiliser, sunlight destroys free "
                    f"chlorine within 1–2 hours. "
                    f"Add {cya_dose_g:,.0f} g of cyanuric acid. Place it in a mesh bag in the "
                    f"skimmer basket — it dissolves slowly over 48–72 hours."
                ),
                secondary_effects=[
                    "Raising cyanuric acid reduces the active chlorine percentage — you may need "
                    "to raise free chlorine to maintain sanitation",
                    "Contributes to the total alkalinity test reading (corrected alkalinity will differ)",
                ],
                priority=6,
                warnings=[
                    "Cyanuric acid cannot be removed chemically — only by dilution (drain and refill)",
                    "Add conservatively and re-test after 72 hours",
                ],
            ))

        elif test.cyanuric_acid > cya_max:
            drain_fraction = (1 - cya_target / test.cyanuric_acid) * 100  # percent
            recommendations.append(Recommendation(
                parameter="cyanuric_acid",
                parameter_name="Cyanuric Acid (Stabiliser)",
                current_value=test.cyanuric_acid,
                current_unit="ppm",
                target_value=cya_target,
                ideal_range=(cya_min, cya_max),
                chemical_name="Partial drain and refill (dilution)",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="decrease",
                explanation=(
                    f"Cyanuric acid of {test.cyanuric_acid:.0f} ppm is above the maximum of "
                    f"{cya_max:.0f} ppm. Excess cyanuric acid severely reduces chlorine effectiveness. "
                    f"The only solution is to replace approximately {drain_fraction:.0f}% of pool water."
                ),
                priority=5,
                warnings=[
                    "High cyanuric acid is the leading cause of algae in pools with 'normal' free chlorine readings",
                    "After dilution, re-test ALL parameters and rebalance",
                ],
            ))

    # =====================================================================
    # CHECK 7: SALT (ppm) — SWG pools only
    # =====================================================================
    if pool.pool_type == PoolType.SALTWATER and test.salt is not None:
        salt_min, salt_max, salt_target = ranges["salt"]  # ppm

        if test.salt < salt_min:
            salt_increase = salt_target - test.salt  # ppm
            salt_g = dose_salt(vol, salt_increase)   # g
            recommendations.append(Recommendation(
                parameter="salt",
                parameter_name="Salt",
                current_value=test.salt,
                current_unit="ppm",
                target_value=salt_target,
                ideal_range=(salt_min, salt_max),
                chemical_name="Sodium Chloride (pool-grade salt)",
                chemical_formula="NaCl",
                dose_amount=salt_g,
                dose_unit="g",
                direction="increase",
                explanation=(
                    f"Salt of {test.salt:.0f} ppm is below the minimum of {salt_min:.0f} ppm. "
                    f"Your salt chlorinator needs {salt_min:.0f}–{salt_max:.0f} ppm to operate. "
                    f"Add {salt_g / 1000:.1f} kg of pool-grade NaCl. Broadcast across the surface "
                    f"with the pump running. Re-test after 24 hours of circulation. "
                    f"Salt is not consumed during chlorine generation — only lost via dilution."
                ),
                secondary_effects=[
                    "No effect on pH, total alkalinity, calcium hardness, cyanuric acid, or free chlorine",
                    "Increases total dissolved solids",
                ],
                priority=7,
            ))

        elif test.salt > salt_max:
            drain_pct = (1 - salt_target / test.salt) * 100  # percent
            recommendations.append(Recommendation(
                parameter="salt",
                parameter_name="Salt",
                current_value=test.salt,
                current_unit="ppm",
                target_value=salt_target,
                ideal_range=(salt_min, salt_max),
                chemical_name="Partial drain and refill (dilution)",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="decrease",
                explanation=(
                    f"Salt of {test.salt:.0f} ppm is above the maximum of {salt_max:.0f} ppm. "
                    f"Excess salt can damage the chlorinator cell. "
                    f"Replace approximately {drain_pct:.0f}% of pool water to reach {salt_target:.0f} ppm."
                ),
                priority=7,
            ))

    # =====================================================================
    # CHECK 8: TOTAL DISSOLVED SOLIDS (ppm)
    # =====================================================================
    if test.total_dissolved_solids is not None:
        tds_min, tds_max, tds_target = ranges["total_dissolved_solids"]  # ppm

        if test.total_dissolved_solids > tds_max:
            drain_pct = (1 - tds_target / test.total_dissolved_solids) * 100  # percent
            recommendations.append(Recommendation(
                parameter="total_dissolved_solids",
                parameter_name="Total Dissolved Solids",
                current_value=test.total_dissolved_solids,
                current_unit="ppm",
                target_value=tds_target,
                ideal_range=(tds_min, tds_max),
                chemical_name="Partial drain and refill (dilution)",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="decrease",
                explanation=(
                    f"Total dissolved solids of {test.total_dissolved_solids:.0f} ppm is above "
                    f"the maximum of {tds_max:.0f} ppm. High total dissolved solids can reduce "
                    f"chlorine efficiency, cause cloudy water, increase corrosion, and give water "
                    f"a salty or metallic taste. Total dissolved solids accumulate over time as "
                    f"chemicals are added and water evaporates. "
                    f"The only remedy is dilution — replace approximately {drain_pct:.0f}% of pool water."
                ),
                priority=8,
            ))

    # =====================================================================
    # CHECK 9: IRON (ppm)
    # =====================================================================
    if test.iron is not None:
        iron_min, iron_max, iron_target = ranges["iron"]  # ppm

        if test.iron > iron_max:
            recommendations.append(Recommendation(
                parameter="iron",
                parameter_name="Iron (Fe)",
                current_value=test.iron,
                current_unit="ppm",
                target_value=iron_target,
                ideal_range=(iron_min, iron_max),
                chemical_name="Metal sequestrant + source elimination",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="decrease",
                explanation=(
                    f"Iron of {test.iron:.2f} ppm is above the maximum of {iron_max:.2f} ppm. "
                    f"Dissolved iron causes brown/orange staining on pool surfaces, discolouration "
                    f"of water (especially after chlorination, which oxidises Fe²⁺ to Fe³⁺), "
                    f"and clogging of filter media. "
                    f"Immediate action: add a metal sequestrant (follow product directions for "
                    f"dosage based on pool volume of {vol:,.0f} L). This binds dissolved iron "
                    f"and prevents staining but does not remove it. "
                    f"Long-term: identify the source — bore/well water is the most common cause. "
                    f"Consider a pre-filter or hose-end filter when topping up. "
                    f"Do NOT shock the pool while iron is elevated — chlorine oxidises iron and "
                    f"causes immediate staining."
                ),
                secondary_effects=[
                    "Sequestrants are consumed over time and need regular re-dosing",
                    "Iron can interfere with the combined chlorine (DPD) test — readings may be falsely high",
                ],
                priority=3,
                warnings=[
                    "Do NOT superchlorinate (SLAM) while iron is elevated — this will cause staining",
                    "Keep pH at 7.2–7.4 to minimise iron oxidation",
                    "Run the filter continuously to catch any precipitated iron particles",
                ],
            ))

    # =====================================================================
    # CHECK 10: COPPER (ppm)
    # =====================================================================
    if test.copper is not None:
        cu_min, cu_max, cu_target = ranges["copper"]  # ppm

        if test.copper > cu_max:
            recommendations.append(Recommendation(
                parameter="copper",
                parameter_name="Copper (Cu)",
                current_value=test.copper,
                current_unit="ppm",
                target_value=cu_target,
                ideal_range=(cu_min, cu_max),
                chemical_name="Metal sequestrant + source elimination",
                chemical_formula="—",
                dose_amount=0,
                dose_unit="",
                direction="decrease",
                explanation=(
                    f"Copper of {test.copper:.2f} ppm is above the maximum of {cu_max:.2f} ppm. "
                    f"Dissolved copper causes blue/green staining on pool surfaces and can turn "
                    f"blonde hair green. Common sources include copper-based algaecides, "
                    f"corroding copper heat exchangers, and bore/well water. "
                    f"Immediate action: add a metal sequestrant. "
                    f"Long-term: stop using copper-based algaecides, check heat exchanger "
                    f"condition, and ensure LSI is balanced (corrosive water dissolves copper "
                    f"fittings). If copper is from fill water, use a pre-filter."
                ),
                secondary_effects=[
                    "Copper stains are difficult to remove once set — prevention is critical",
                    "Sequestrants bind copper but do not remove it from the water",
                ],
                priority=3,
                warnings=[
                    "Maintain pH above 7.2 — acidic water accelerates copper corrosion",
                    "Check and balance LSI to prevent equipment corrosion",
                    "Copper can interfere with DPD chlorine test readings",
                ],
            ))

    # =====================================================================
    # COMPUTE LSI (dimensionless)
    # =====================================================================
    lsi_val = None             # dimensionless
    lsi_text = ""
    corr_alk_ppm = None        # ppm
    if test.ph is not None and test.calcium_hardness is not None and test.total_alkalinity is not None:
        tds_estimate = 1000.0  # ppm
        if test.total_dissolved_solids is not None:
            tds_estimate = test.total_dissolved_solids
        elif pool.pool_type == PoolType.SALTWATER and test.salt is not None:
            tds_estimate = test.salt + 500  # ppm — rough estimate

        corr_alk_ppm = corrected_alkalinity(test.total_alkalinity, cyanuric_acid_ppm, test.ph)
        lsi_val = calculate_lsi(
            ph=test.ph,
            temp_c=pool.temperature_c,
            calcium_hardness_ppm=test.calcium_hardness,
            total_alkalinity_ppm=test.total_alkalinity,
            cyanuric_acid_ppm=cyanuric_acid_ppm,
            tds_ppm=tds_estimate,
        )
        lsi_text = interpret_lsi(lsi_val)

    # =====================================================================
    # FREE CHLORINE / CYA ANALYSIS
    # =====================================================================
    fc_cya_text = ""
    hocl_pct = None  # percent
    if test.free_chlorine is not None:
        ph_val = test.ph if test.ph is not None else 7.5  # dimensionless
        hocl_pct = percent_hocl(ph_val, cyanuric_acid_ppm)  # percent

        if cyanuric_acid_ppm > 0:
            effective_hocl = test.free_chlorine * hocl_pct / 100  # ppm
            fc_cya_text = (
                f"At cyanuric acid {cyanuric_acid_ppm:.0f} ppm and pH {ph_val:.1f}, approximately "
                f"{hocl_pct:.2f}% of your measured free chlorine ({test.free_chlorine:.1f} ppm) is "
                f"active HOCl. That is an effective HOCl concentration of {effective_hocl:.3f} ppm. "
                f"Recommended minimum free chlorine at this cyanuric acid level: {fc_min:.1f} ppm. "
                f"Target free chlorine: {fc_target:.1f} ppm. SLAM level: {fc_slam:.0f} ppm."
            )
        else:
            fc_cya_text = (
                f"No cyanuric acid detected — your pool is unstabilised. At pH {ph_val:.1f}, "
                f"{hocl_pct:.1f}% of free chlorine is active HOCl, which is high but chlorine will "
                f"be destroyed rapidly by UV sunlight (typically 90% loss within 2 hours)."
            )

    # =====================================================================
    # SORT AND BUILD SUMMARY
    # =====================================================================
    recommendations.sort(key=lambda r: r.priority)

    if not recommendations:
        summary = (
            "All tested parameters are within the recommended ranges. "
            "No chemical adjustments needed at this time. Continue regular testing."
        )
    else:
        param_names = [r.parameter_name for r in recommendations]
        summary = (
            f"Found {len(recommendations)} parameter(s) outside ideal range: "
            f"{', '.join(param_names)}. "
            f"Recommendations are listed in priority order — address them from top to bottom."
        )

    return AnalysisReport(
        pool=pool,
        test_results=test,
        recommendations=recommendations,
        lsi=lsi_val,
        lsi_interpretation=lsi_text,
        fc_cya_analysis=fc_cya_text,
        hocl_percent=hocl_pct,
        corrected_alkalinity_ppm=corr_alk_ppm,
        summary=summary,
    )


# =============================================================================
# SECTION 7: VISUALISATION — Horizontal bar chart of test results vs ranges
# =============================================================================

def plot_test_results(
    pool: PoolProfile,
    test: TestResults,
    save_path: Optional[str] = None,
    figsize: tuple = (14, 9),
) -> plt.Figure:
    """
    Plot each tested parameter as a horizontal bar showing the measured value
    relative to the desired range.

    The chart shows:
      - A shaded green band for the ideal range
      - A vertical green line at the target midpoint
      - A marker (dot) for the measured value, colour-coded:
            green  = within range
            amber  = slightly outside (within 20% of boundary)
            red    = significantly outside range
      - A text annotation showing the exact measured value with units

    Parameters that were not tested (None) are omitted.

    Parameters
    ----------
    pool : PoolProfile
    test : TestResults
    save_path : str, optional — if provided, save the figure to this path.
    figsize : tuple — figure dimensions in inches (width, height).

    Returns
    -------
    matplotlib.figure.Figure
    """
    ranges = get_default_ranges(pool.pool_type, pool.surface)

    # Override free chlorine range with CYA-dependent values
    cyanuric_acid_ppm = test.cyanuric_acid if test.cyanuric_acid is not None else 0.0
    fc_min, fc_target, fc_slam = fc_target_from_cya(cyanuric_acid_ppm, pool.pool_type)
    fc_range_max = min(fc_slam * 0.5, fc_target + 4.0)
    ranges["free_chlorine"] = (fc_min, fc_range_max, fc_target)

    # Derive combined chlorine
    combined_chlorine_ppm = _derive_combined_chlorine(test)

    # Build list of parameters to plot (only those with measured values)
    plot_data = []
    for key in PARAMETER_KEYS:
        display_name, unit = PARAMETER_DISPLAY[key]

        # Get the measured value
        if key == "combined_chlorine":
            value = combined_chlorine_ppm
        else:
            value = _get_test_value(test, key)

        if value is None:
            continue

        if key not in ranges:
            continue

        range_min, range_max, target = ranges[key]
        plot_data.append({
            "key": key,
            "name": display_name,
            "unit": unit,
            "value": value,
            "range_min": range_min,
            "range_max": range_max,
            "target": target,
        })

    if not plot_data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No test results to display",
                ha="center", va="center", fontsize=14, color="#666")
        ax.set_axis_off()
        return fig

    # Reverse so the first parameter appears at the top
    plot_data = list(reversed(plot_data))
    n = len(plot_data)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    y_positions = np.arange(n)
    bar_height = 0.45

    for i, d in enumerate(plot_data):
        y = y_positions[i]
        rmin = d["range_min"]
        rmax = d["range_max"]
        target = d["target"]
        value = d["value"]
        unit_str = f' {d["unit"]}' if d["unit"] else ""

        # Determine the axis extent for this row
        # Show at least 30% beyond the range on each side, and always include the value
        span = rmax - rmin if rmax > rmin else 1.0
        pad = span * 0.4
        x_lo = min(rmin - pad, value - span * 0.15)
        x_hi = max(rmax + pad, value + span * 0.15)

        # Draw the ideal range band (green)
        ax.barh(y, rmax - rmin, left=rmin, height=bar_height * 1.6,
                color="#d4edda", edgecolor="#28a745", linewidth=0.8, zorder=1)

        # Draw the full axis extent as a light grey line
        ax.plot([x_lo, x_hi], [y, y], color="#e0e0e0", linewidth=6, solid_capstyle="round", zorder=0)

        # Target midpoint line
        ax.plot([target, target], [y - bar_height * 0.9, y + bar_height * 0.9],
                color="#28a745", linewidth=2, linestyle="--", zorder=2, alpha=0.7)

        # Determine marker colour
        if rmin <= value <= rmax:
            colour = "#28a745"    # green — in range
            status = "✓"
        else:
            # How far outside?
            if value < rmin:
                deviation_fraction = (rmin - value) / span if span > 0 else 1.0
            else:
                deviation_fraction = (value - rmax) / span if span > 0 else 1.0

            if deviation_fraction <= 0.25:
                colour = "#ffc107"   # amber — slightly outside
                status = "⚠"
            else:
                colour = "#dc3545"   # red — significantly outside
                status = "✗"

        # Plot the measured value marker
        ax.plot(value, y, "o", color=colour, markersize=14, markeredgecolor="white",
                markeredgewidth=2, zorder=5)

        # Annotation: value with units
        # Position the text to the right of the marker, or left if near right edge
        text_x = value + span * 0.05
        ha = "left"
        if value > (rmax + rmin) / 2 + span * 0.3:
            text_x = value - span * 0.05
            ha = "right"

        # Format value: integers for large numbers, 1–2 decimals for small
        if abs(value) >= 10:
            val_str = f"{value:.0f}"
        elif abs(value) >= 1:
            val_str = f"{value:.1f}"
        else:
            val_str = f"{value:.2f}"

        ax.annotate(
            f"{status} {val_str}{unit_str}",
            xy=(value, y), xytext=(text_x, y + bar_height * 0.55),
            fontsize=10, fontweight="bold", color=colour,
            ha=ha, va="bottom", zorder=6,
        )

        # Set axis limits for each row (we'll use a common transform approach)
        # Since matplotlib doesn't support per-row x-axes, we normalise data
        # to a common scale below.

    # --- Common x-axis approach: normalise all values to 0–1 scale ---
    # This is complex, so instead we use the simplest approach: separate subplots

    # Actually, let's re-do this with subplots for proper per-parameter scaling
    plt.close(fig)

    fig, axes = plt.subplots(n, 1, figsize=figsize, gridspec_kw={"hspace": 0.5})
    fig.patch.set_facecolor("#fafafa")

    if n == 1:
        axes = [axes]

    for i, d in enumerate(plot_data):
        ax = axes[i]
        ax.set_facecolor("#fafafa")

        rmin = d["range_min"]
        rmax = d["range_max"]
        target = d["target"]
        value = d["value"]
        name = d["name"]
        unit_str = f' {d["unit"]}' if d["unit"] else ""

        span = rmax - rmin if rmax > rmin else max(abs(value), 1.0)
        pad = span * 0.5
        x_lo = min(rmin - pad, value - span * 0.2)
        x_hi = max(rmax + pad, value + span * 0.2)

        # Ensure we don't go negative for parameters that can't be negative
        if d["key"] not in ["ph"] and x_lo < 0:
            x_lo = 0

        # Background bar (full axis)
        ax.barh(0, x_hi - x_lo, left=x_lo, height=0.6,
                color="#f0f0f0", edgecolor="none", zorder=0)

        # Ideal range band
        ax.barh(0, rmax - rmin, left=rmin, height=0.6,
                color="#d4edda", edgecolor="#28a745", linewidth=1.2, zorder=1)

        # Target line
        ax.plot([target, target], [-0.35, 0.35],
                color="#28a745", linewidth=2, linestyle="--", zorder=2, alpha=0.8)

        # Range boundary labels
        ax.text(rmin, -0.42, f"{rmin:.4g}", ha="center", va="top",
                fontsize=7, color="#28a745", alpha=0.8)
        ax.text(rmax, -0.42, f"{rmax:.4g}", ha="center", va="top",
                fontsize=7, color="#28a745", alpha=0.8)

        # Marker colour
        if rmin <= value <= rmax:
            colour = "#28a745"
            status = "✓ IN RANGE"
        else:
            if value < rmin:
                deviation = (rmin - value) / span if span > 0 else 1.0
            else:
                deviation = (value - rmax) / span if span > 0 else 1.0

            if deviation <= 0.25:
                colour = "#fd7e14"
                status = "⚠ SLIGHTLY OUT"
            else:
                colour = "#dc3545"
                status = "✗ OUT OF RANGE"

        # Marker
        ax.plot(value, 0, "o", color=colour, markersize=16, markeredgecolor="white",
                markeredgewidth=2.5, zorder=5)

        # Value annotation
        if abs(value) >= 100:
            val_str = f"{value:,.0f}"
        elif abs(value) >= 10:
            val_str = f"{value:.0f}"
        elif abs(value) >= 1:
            val_str = f"{value:.1f}"
        else:
            val_str = f"{value:.2f}"

        ax.text(value, 0.45, f"{val_str}{unit_str}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=colour, zorder=6)

        # Parameter name label (left side)
        ax.set_ylabel(name, fontsize=10, fontweight="bold", rotation=0,
                      labelpad=140, ha="right", va="center")

        # Status badge (right side)
        ax.text(x_hi + (x_hi - x_lo) * 0.02, 0, status, ha="left", va="center",
                fontsize=8, fontweight="bold", color=colour, zorder=6)

        ax.set_xlim(x_lo, x_hi + (x_hi - x_lo) * 0.15)
        ax.set_ylim(-0.55, 0.65)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7, colors="#888")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#ccc")

    # Title
    pool_desc = (f"{pool.volume_litres:,.0f} L {pool.pool_type.value} "
                 f"{pool.surface.value} pool at {pool.temperature_c:.0f}°C")
    fig.suptitle(f"Pool Chemistry Test Results — {pool_desc}",
                 fontsize=14, fontweight="bold", y=0.98)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor="#d4edda", edgecolor="#28a745", label="Ideal range"),
        plt.Line2D([0], [0], color="#28a745", linewidth=2, linestyle="--", label="Target midpoint"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#28a745",
                   markersize=10, label="In range"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#fd7e14",
                   markersize=10, label="Slightly out"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#dc3545",
                   markersize=10, label="Out of range"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
              fontsize=9, frameon=True, edgecolor="#ccc", fancybox=True,
              bbox_to_anchor=(0.5, 0.01))

    fig.subplots_adjust(left=0.22, right=0.95, top=0.93, bottom=0.07, hspace=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


# =============================================================================
# SECTION 8: PRETTY PRINTING (CLI / Notebook output)
# =============================================================================

def format_report(report: AnalysisReport) -> str:
    """Format an AnalysisReport as a human-readable string with consistent units."""
    lines = []
    lines.append("=" * 76)
    lines.append("  POOL CHEMISTRY ANALYSIS REPORT")
    lines.append("=" * 76)

    # Pool info
    p = report.pool
    lines.append(f"\n  Pool: {p.volume_litres:,.0f} L  |  {p.pool_type.value}  |  "
                 f"{p.surface.value} surface  |  {p.temperature_c:.0f} °C")

    # Test results with units
    t = report.test_results
    cc = _derive_combined_chlorine(t)
    lines.append("\n  TEST RESULTS:")
    if t.free_chlorine is not None:
        lines.append(f"    Free Chlorine:           {t.free_chlorine:.1f} ppm")
    if t.total_chlorine is not None:
        lines.append(f"    Total Chlorine:          {t.total_chlorine:.1f} ppm")
    if cc is not None:
        lines.append(f"    Combined Chlorine:       {cc:.1f} ppm")
    if t.ph is not None:
        lines.append(f"    pH:                      {t.ph:.2f}")
    if t.total_alkalinity is not None:
        lines.append(f"    Total Alkalinity:        {t.total_alkalinity:.0f} ppm")
    if t.calcium_hardness is not None:
        lines.append(f"    Calcium Hardness:        {t.calcium_hardness:.0f} ppm")
    if t.cyanuric_acid is not None:
        lines.append(f"    Cyanuric Acid:           {t.cyanuric_acid:.0f} ppm")
    if t.salt is not None:
        lines.append(f"    Salt:                    {t.salt:.0f} ppm")
    if t.total_dissolved_solids is not None:
        lines.append(f"    Total Dissolved Solids:  {t.total_dissolved_solids:.0f} ppm")
    if t.iron is not None:
        lines.append(f"    Iron (Fe):               {t.iron:.2f} ppm")
    if t.copper is not None:
        lines.append(f"    Copper (Cu):             {t.copper:.2f} ppm")

    # LSI
    if report.lsi is not None:
        lines.append(f"\n  LANGELIER SATURATION INDEX (LSI): {report.lsi:+.2f} (dimensionless)")
        lines.append(f"    {report.lsi_interpretation}")
        if report.corrected_alkalinity_ppm is not None:
            lines.append(f"    Corrected alkalinity used for LSI: {report.corrected_alkalinity_ppm:.0f} ppm")

    # FC/CYA
    if report.fc_cya_analysis:
        lines.append(f"\n  FREE CHLORINE / CYANURIC ACID ANALYSIS:")
        lines.append(f"    {report.fc_cya_analysis}")

    # Summary
    lines.append(f"\n  SUMMARY: {report.summary}")

    # Recommendations
    if report.recommendations:
        lines.append(f"\n{'─' * 76}")
        lines.append("  RECOMMENDATIONS (in priority order):")
        lines.append(f"{'─' * 76}")

        for i, rec in enumerate(report.recommendations, 1):
            unit_str = f" {rec.current_unit}" if rec.current_unit else ""
            lines.append(f"\n  [{i}] {rec.parameter_name.upper()} — {rec.direction.upper()}")
            lines.append(
                f"      Current: {rec.current_value}{unit_str}  →  "
                f"Target: {rec.target_value}{unit_str}  "
                f"(Range: {rec.ideal_range[0]}–{rec.ideal_range[1]}{unit_str})"
            )
            lines.append(f"      Chemical: {rec.chemical_name}")
            if rec.dose_amount > 0:
                lines.append(f"      Dose: {rec.dose_amount:,.1f} {rec.dose_unit}")
            lines.append(f"\n      {rec.explanation}")

            if rec.secondary_effects:
                lines.append(f"\n      Side effects:")
                for se in rec.secondary_effects:
                    lines.append(f"        • {se}")

            if rec.warnings:
                lines.append(f"\n      Warnings:")
                for w in rec.warnings:
                    lines.append(f"        ⚠ {w}")

    lines.append(f"\n{'=' * 76}")
    return "\n".join(lines)


# =============================================================================
# SECTION 9: DEMO / CLI ENTRY POINT
# =============================================================================

def demo():
    """Run a demonstration with a sample pool, print report, and generate chart."""
    print("\n" + "=" * 76)
    print("  POOL CHEMISTRY CALCULATOR — DEMO RUN")
    print("=" * 76)

    # ------------------------------------------------------------------
    # Saltwater pool with several issues including metals
    # ------------------------------------------------------------------
    pool = PoolProfile(
        volume_litres=50_000,       # L
        pool_type=PoolType.SALTWATER,
        surface=SurfaceType.PLASTER,
        temperature_c=28,           # °C
    )

    test = TestResults(
        free_chlorine=2.0,          # ppm — low for SWG with CYA 80
        total_chlorine=2.8,         # ppm — TC > FC → combined chlorine = 0.8 ppm
        ph=7.9,                     # dimensionless — high
        total_alkalinity=120,       # ppm — high for SWG
        calcium_hardness=280,       # ppm — slightly low for plaster
        cyanuric_acid=80,           # ppm — upper end of range
        salt=2500,                  # ppm — low for SWG
        total_dissolved_solids=4200, # ppm — within SWG range
        iron=0.35,                  # ppm — elevated
        copper=0.05,                # ppm — acceptable
    )

    print("\n  Example: 50,000 L saltwater plaster pool at 28 °C")
    print("  Test: FC=2.0 ppm, TC=2.8 ppm, pH=7.90, TA=120 ppm, CH=280 ppm,")
    print("        CYA=80 ppm, Salt=2500 ppm, TDS=4200 ppm, Fe=0.35 ppm, Cu=0.05 ppm")
    print()

    report = analyse_pool(pool, test)
    print(format_report(report))

    # Generate the chart
    chart_path = "/home/claude/pool_chart.png"
    plot_test_results(pool, test, save_path=chart_path)
    print(f"\n  Chart saved to: {chart_path}")

    return report


if __name__ == "__main__":
    demo()
