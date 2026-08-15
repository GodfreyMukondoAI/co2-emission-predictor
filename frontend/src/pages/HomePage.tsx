import {
  Activity,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Database,
  Gauge,
  Info,
  Target,
  TrendingUp,
  XCircle,
} from "lucide-react";

import MetricCard from "../components/dashboard/MetricCard";
import ModelStatus from "../components/dashboard/ModelStatus";
import PredictionCard from "../components/prediction/PredictionCard";

import type { ModelInfoResponse } from "../types/prediction";

/* ==========================================================================
   TYPES
========================================================================== */

interface HomePageProps {
  apiOnline: boolean;
  modelInfo: ModelInfoResponse | null;
  modelLoading: boolean;
  onNavigate: (page: string) => void;
}

interface FeatureStatistics {
  min: number | null;
  max: number | null;
  mean?: number | null;
  median?: number | null;
  std?: number | null;
}

interface DatasetInfo {
  records: number | null;
  feature_count: number | null;

  features: string[];

  target: string | null;

  feature_statistics: Record<
    string,
    FeatureStatistics
  >;

  target_statistics: FeatureStatistics | null;

  /*
   * Legacy flattened fields are kept for compatibility
   * with older backend responses.
   */
  engine_size_min?: number | null;
  engine_size_max?: number | null;

  fuel_consumption_mpg_min?: number | null;
  fuel_consumption_mpg_max?: number | null;

  co2_min?: number | null;
  co2_max?: number | null;
}

interface ModelMetrics {
  r2_score?: number | null;
  mae?: number | null;
  rmse?: number | null;
}

interface NormalizedModelInfo {
  name: string;
  algorithm: string;
  version: string;
  target: string;
  targetUnit: string;
  features: string[];
  metrics: ModelMetrics | null;
  dataset: DatasetInfo | null;
}

/* ==========================================================================
   APPLICATION CONFIGURATION
========================================================================== */

const NAVIGATION = {
  prediction: "prediction",
  model: "model",
} as const;

const UI = {
  notAvailable: "—",

  model: {
    fallbackName: "Machine Learning Model",
    fallbackAlgorithm: "Machine Learning",
    fallbackTarget: "Prediction",
    fallbackUnit: "",
    fallbackVersion: "—",
  },

  status: {
    online: "Operational",
    offline: "Unavailable",
  },

  featureLabels: {
    ENGINESIZE: "Engine Size",
    FUELCONSUMPTION_COMB_MPG:
      "Combined Fuel Consumption",
    CO2EMISSIONS: "CO₂ Emissions",
  },

  units: {
    ENGINESIZE: "L",
    FUELCONSUMPTION_COMB_MPG: "MPG",
    CO2EMISSIONS: "g/km",
  },
} as const;

/* ==========================================================================
   HELPERS
========================================================================== */

/**
 * Safely formats numeric values.
 */
function formatNumber(
  value: number | null | undefined,
  maximumFractionDigits = 2,
): string {
  if (
    value === null ||
    value === undefined ||
    !Number.isFinite(value)
  ) {
    return UI.notAvailable;
  }

  return new Intl.NumberFormat(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits,
  }).format(value);
}

/**
 * Safely formats a numerical range.
 */
function formatRange(
  minimum: number | null | undefined,
  maximum: number | null | undefined,
  unit?: string | null,
): string {
  if (
    minimum === null ||
    minimum === undefined ||
    !Number.isFinite(minimum) ||
    maximum === null ||
    maximum === undefined ||
    !Number.isFinite(maximum)
  ) {
    return UI.notAvailable;
  }

  const range = `${formatNumber(
    minimum,
    1,
  )} – ${formatNumber(
    maximum,
    1,
  )}`;

  const normalizedUnit =
    typeof unit === "string" &&
    unit.trim().length > 0
      ? unit.trim()
      : "";

  return normalizedUnit
    ? `${range} ${normalizedUnit}`
    : range;
}

/**
 * Safely normalizes API text.
 */
function normalizeText(
  value: unknown,
  fallback: string,
): string {
  if (
    typeof value !== "string" ||
    value.trim().length === 0
  ) {
    return fallback;
  }

  return value.trim();
}

/**
 * Returns only valid feature names.
 */
function normalizeFeatures(
  value: unknown,
): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .filter(
      (feature): feature is string =>
        typeof feature === "string" &&
        feature.trim().length > 0,
    )
    .map((feature) => feature.trim());
}

/**
 * Safely converts an unknown value to a finite number.
 */
function normalizeNumber(
  value: unknown,
): number | null {
  return typeof value === "number" &&
    Number.isFinite(value)
    ? value
    : null;
}

/**
 * Safely extracts metrics.
 */
function normalizeMetrics(
  value: unknown,
): ModelMetrics | null {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    return null;
  }

  const metrics =
    value as Record<string, unknown>;

  const normalized: ModelMetrics = {
    r2_score: normalizeNumber(
      metrics.r2_score,
    ),

    mae: normalizeNumber(
      metrics.mae,
    ),

    rmse: normalizeNumber(
      metrics.rmse,
    ),
  };

  const hasMetrics =
    normalized.r2_score !== null ||
    normalized.mae !== null ||
    normalized.rmse !== null;

  return hasMetrics
    ? normalized
    : null;
}

/**
 * Safely normalizes feature statistics.
 */
function normalizeFeatureStatistics(
  value: unknown,
): FeatureStatistics {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    return {
      min: null,
      max: null,
      mean: null,
      median: null,
      std: null,
    };
  }

  const statistics =
    value as Record<string, unknown>;

  return {
    min: normalizeNumber(statistics.min),
    max: normalizeNumber(statistics.max),
    mean: normalizeNumber(statistics.mean),
    median: normalizeNumber(statistics.median),
    std: normalizeNumber(statistics.std),
  };
}

/**
 * Safely extracts feature statistics.
 */
function normalizeFeatureStatisticsMap(
  value: unknown,
): Record<string, FeatureStatistics> {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    return {};
  }

  const source =
    value as Record<string, unknown>;

  const result: Record<
    string,
    FeatureStatistics
  > = {};

  Object.entries(source).forEach(
    ([featureName, statistics]) => {
      result[featureName] =
        normalizeFeatureStatistics(
          statistics,
        );
    },
  );

  return result;
}

/**
 * Safely extracts dataset information.
 *
 * Supports the current backend structure:
 *
 * dataset.feature_statistics.ENGINESIZE
 * dataset.feature_statistics.FUELCONSUMPTION_COMB_MPG
 * dataset.target_statistics
 *
 * It also supports the previous flattened structure
 * so the frontend remains backward compatible.
 */
function normalizeDataset(
  value: unknown,
): DatasetInfo | null {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  ) {
    return null;
  }

  const dataset =
    value as Record<string, unknown>;

  const featureStatistics =
    normalizeFeatureStatisticsMap(
      dataset.feature_statistics,
    );

  const rawTargetStatistics =
    dataset.target_statistics;

  const targetStatistics =
    rawTargetStatistics !== undefined
      ? normalizeFeatureStatistics(
          rawTargetStatistics,
        )
      : null;

  return {
    records: normalizeNumber(
      dataset.records,
    ),

    feature_count: normalizeNumber(
      dataset.feature_count,
    ),

    features: normalizeFeatures(
      dataset.features,
    ),

    target:
      typeof dataset.target === "string"
        ? dataset.target
        : null,

    feature_statistics:
      featureStatistics,

    target_statistics:
      targetStatistics,

    /*
     * Backward-compatible flattened fields.
     */
    engine_size_min:
      normalizeNumber(
        dataset.engine_size_min,
      ),

    engine_size_max:
      normalizeNumber(
        dataset.engine_size_max,
      ),

    fuel_consumption_mpg_min:
      normalizeNumber(
        dataset.fuel_consumption_mpg_min,
      ),

    fuel_consumption_mpg_max:
      normalizeNumber(
        dataset.fuel_consumption_mpg_max,
      ),

    co2_min:
      normalizeNumber(
        dataset.co2_min,
      ),

    co2_max:
      normalizeNumber(
        dataset.co2_max,
      ),
  };
}

/**
 * Normalizes the complete backend model response.
 */
function normalizeModelInfo(
  modelInfo: ModelInfoResponse | null,
): NormalizedModelInfo {
  if (!modelInfo) {
    return {
      name: UI.model.fallbackName,
      algorithm:
        UI.model.fallbackAlgorithm,
      version: UI.model.fallbackVersion,
      target: UI.model.fallbackTarget,
      targetUnit: UI.model.fallbackUnit,
      features: [],
      metrics: null,
      dataset: null,
    };
  }

  return {
    name: normalizeText(
      modelInfo.model_name,
      UI.model.fallbackName,
    ),

    algorithm: normalizeText(
      modelInfo.algorithm,
      UI.model.fallbackAlgorithm,
    ),

    version: normalizeText(
      modelInfo.version,
      UI.model.fallbackVersion,
    ),

    target: normalizeText(
      modelInfo.target,
      UI.model.fallbackTarget,
    ),

    targetUnit: normalizeText(
      modelInfo.target_unit,
      UI.model.fallbackUnit,
    ),

    features: normalizeFeatures(
      modelInfo.features,
    ),

    metrics: normalizeMetrics(
      modelInfo.metrics,
    ),

    dataset: normalizeDataset(
      modelInfo.dataset,
    ),
  };
}

/**
 * Determines whether at least one metric is available.
 */
function hasAvailableMetrics(
  metrics: ModelMetrics | null,
): boolean {
  if (!metrics) {
    return false;
  }

  return (
    Number.isFinite(metrics.r2_score) ||
    Number.isFinite(metrics.mae) ||
    Number.isFinite(metrics.rmse)
  );
}

/**
 * Adds a unit to a formatted value.
 */
function withUnit(
  value: string,
  unit?: string | null,
): string {
  if (
    !unit ||
    unit.trim().length === 0 ||
    value === UI.notAvailable
  ) {
    return value;
  }

  return `${value} ${unit.trim()}`;
}

/**
 * Returns a friendly display name for a feature.
 *
 * Backend feature names remain the source of truth.
 * This only controls presentation.
 */
function getFeatureLabel(
  featureName: string,
): string {
  const knownLabel =
    UI.featureLabels[
      featureName as keyof typeof UI.featureLabels
    ];

  if (knownLabel) {
    return knownLabel;
  }

  return featureName
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(/\b\w/g, (character) =>
      character.toUpperCase(),
    );
}

/**
 * Returns the appropriate unit for a known feature.
 */
function getFeatureUnit(
  featureName: string,
): string {
  const unit =
    UI.units[
      featureName as keyof typeof UI.units
    ];

  return unit ?? "";
}

/**
 * Finds the statistics for a specific feature.
 */
function getFeatureStatistics(
  dataset: DatasetInfo | null,
  featureName: string,
): FeatureStatistics | null {
  if (!dataset) {
    return null;
  }

  const statistics =
    dataset.feature_statistics[
      featureName
    ];

  if (statistics) {
    return statistics;
  }

  /*
   * Backward compatibility for older API response.
   */
  if (
    featureName === "ENGINESIZE"
  ) {
    return {
      min:
        dataset.engine_size_min ??
        null,

      max:
        dataset.engine_size_max ??
        null,
    };
  }

  if (
    featureName ===
    "FUELCONSUMPTION_COMB_MPG"
  ) {
    return {
      min:
        dataset.fuel_consumption_mpg_min ??
        null,

      max:
        dataset.fuel_consumption_mpg_max ??
        null,
    };
  }

  return null;
}

/**
 * Returns the target statistics.
 */
function getTargetStatistics(
  dataset: DatasetInfo | null,
): FeatureStatistics | null {
  if (!dataset) {
    return null;
  }

  if (dataset.target_statistics) {
    return dataset.target_statistics;
  }

  /*
   * Backward compatibility.
   */
  return {
    min: dataset.co2_min ?? null,
    max: dataset.co2_max ?? null,
  };
}

/* ==========================================================================
   COMPONENT
========================================================================== */

function HomePage({
  apiOnline,
  modelInfo,
  modelLoading,
  onNavigate,
}: HomePageProps) {
  /* ------------------------------------------------------------------------
     NORMALIZED BACKEND DATA
  ------------------------------------------------------------------------ */

  const model =
    normalizeModelInfo(modelInfo);

  const {
    name: modelName,
    algorithm,
    version,
    target,
    targetUnit,
    features,
    metrics,
    dataset,
  } = model;

  const featureCount =
    features.length;

  const metricsAvailable =
    hasAvailableMetrics(metrics);

  /* ------------------------------------------------------------------------
     NAVIGATION
  ------------------------------------------------------------------------ */

  const handlePredictionNavigation =
    () => {
      onNavigate(
        NAVIGATION.prediction,
      );
    };

  const handleModelNavigation = () => {
    onNavigate(
      NAVIGATION.model,
    );
  };

  /* ------------------------------------------------------------------------
     DATASET INFORMATION
  ------------------------------------------------------------------------ */

  const datasetRecords =
    dataset?.records ?? null;

  const hasDataset =
    dataset !== null;

  /*
   * Get the actual feature statistics
   * generated by the backend.
   */
  const engineStatistics =
    getFeatureStatistics(
      dataset,
      "ENGINESIZE",
    );

  const fuelStatistics =
    getFeatureStatistics(
      dataset,
      "FUELCONSUMPTION_COMB_MPG",
    );

  const targetStatistics =
    getTargetStatistics(
      dataset,
    );

  const hasEngineRange =
    engineStatistics?.min !== null &&
    engineStatistics?.min !== undefined &&
    engineStatistics?.max !== null &&
    engineStatistics?.max !== undefined;

  const hasFuelRange =
    fuelStatistics?.min !== null &&
    fuelStatistics?.min !== undefined &&
    fuelStatistics?.max !== null &&
    fuelStatistics?.max !== undefined;

  const hasTargetRange =
    targetStatistics?.min !== null &&
    targetStatistics?.min !== undefined &&
    targetStatistics?.max !== null &&
    targetStatistics?.max !== undefined;

  const r2Available =
    metrics?.r2_score !== null &&
    metrics?.r2_score !== undefined &&
    Number.isFinite(
      metrics.r2_score,
    );

  /* ------------------------------------------------------------------------
     RENDER
  ------------------------------------------------------------------------ */

  return (
    <div className="space-y-8 pb-10">
      {/* ====================================================================
          HEADER
      ===================================================================== */}

      <section
        aria-labelledby="dashboard-title"
        className="relative overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm"
      >
        <div
          aria-hidden="true"
          className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-emerald-100/60 blur-3xl"
        />

        <div
          aria-hidden="true"
          className="absolute -bottom-24 left-1/3 h-56 w-56 rounded-full bg-sky-100/40 blur-3xl"
        />

        <div className="relative p-6 sm:p-8 lg:p-10">
          <div className="flex flex-col gap-8 lg:flex-row lg:items-end lg:justify-between">
            <div className="max-w-3xl">
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-emerald-700">
                <Activity
                  size={14}
                  aria-hidden="true"
                />

                {modelName}
              </div>

              <h1
                id="dashboard-title"
                className="mt-5 break-words text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl lg:text-5xl"
              >
                {modelName}
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                {algorithm} is deployed to
                estimate{" "}
                <span className="font-semibold text-slate-800">
                  {target}
                </span>{" "}
                using{" "}
                <span className="font-semibold text-slate-800">
                  {featureCount}
                </span>{" "}
                available model{" "}
                {featureCount === 1
                  ? "feature"
                  : "features"}.
              </p>

              <div className="mt-6 flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={
                    handlePredictionNavigation
                  }
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-emerald-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2"
                >
                  Run Prediction

                  <ArrowRight
                    size={17}
                    aria-hidden="true"
                  />
                </button>

                <button
                  type="button"
                  onClick={
                    handleModelNavigation
                  }
                  className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-300 bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2"
                >
                  View Model

                  <TrendingUp
                    size={17}
                    aria-hidden="true"
                  />
                </button>
              </div>
            </div>

            {/* API STATUS */}

            <div
              className={[
                "flex shrink-0 items-center gap-3 rounded-2xl border px-4 py-3",
                apiOnline
                  ? "border-emerald-200 bg-emerald-50"
                  : "border-red-200 bg-red-50",
              ].join(" ")}
              role="status"
              aria-live="polite"
            >
              <div
                className={[
                  "flex h-10 w-10 items-center justify-center rounded-xl",
                  apiOnline
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-red-100 text-red-700",
                ].join(" ")}
              >
                {apiOnline ? (
                  <CheckCircle2
                    size={20}
                    aria-hidden="true"
                  />
                ) : (
                  <XCircle
                    size={20}
                    aria-hidden="true"
                  />
                )}
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  API Status
                </p>

                <p
                  className={[
                    "mt-0.5 text-sm font-bold",
                    apiOnline
                      ? "text-emerald-700"
                      : "text-red-700",
                  ].join(" ")}
                >
                  {apiOnline
                    ? UI.status.online
                    : UI.status.offline}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ====================================================================
          PERFORMANCE
      ===================================================================== */}

      <section
        aria-labelledby="performance-heading"
      >
        <div className="mb-4 flex items-end justify-between gap-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-emerald-600">
              Performance
            </p>

            <h2
              id="performance-heading"
              className="mt-1 text-xl font-bold text-slate-950"
            >
              Evaluation metrics
            </h2>
          </div>

          <button
            type="button"
            onClick={
              handleModelNavigation
            }
            className="hidden items-center gap-1.5 text-sm font-semibold text-emerald-700 transition hover:text-emerald-800 sm:inline-flex"
          >
            View details

            <ArrowRight
              size={15}
              aria-hidden="true"
            />
          </button>
        </div>

        {modelLoading ? (
          <div
            className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"
            aria-busy="true"
            aria-label="Loading model performance"
          >
            {Array.from({
              length: 4,
            }).map((_, index) => (
              <div
                key={`metric-skeleton-${index}`}
                className="h-32 animate-pulse rounded-2xl border border-slate-200 bg-white"
                aria-hidden="true"
              />
            ))}
          </div>
        ) : modelInfo &&
          metricsAvailable ? (
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="R² Score"
              value={formatNumber(
                metrics?.r2_score,
                4,
              )}
              description="Model explanatory power"
              icon={Target}
            />

            <MetricCard
              label="MAE"
              value={withUnit(
                formatNumber(
                  metrics?.mae,
                  2,
                ),
                targetUnit,
              )}
              description="Mean absolute prediction error"
              icon={Gauge}
            />

            <MetricCard
              label="RMSE"
              value={withUnit(
                formatNumber(
                  metrics?.rmse,
                  2,
                ),
                targetUnit,
              )}
              description="Root mean squared prediction error"
              icon={BarChart3}
            />

            <MetricCard
              label="Dataset"
              value={formatNumber(
                datasetRecords,
                0,
              )}
              description="Records used for model development"
              icon={Database}
            />
          </div>
        ) : (
          <div
            className="rounded-2xl border border-amber-200 bg-amber-50 p-5"
            role="status"
          >
            <p className="text-sm font-semibold text-amber-800">
              Model evaluation data is
              currently unavailable.
            </p>

            <p className="mt-1 text-xs leading-5 text-amber-700">
              The model metadata service
              did not provide usable
              evaluation metrics.
            </p>
          </div>
        )}
      </section>

      {/* ====================================================================
          PREDICTION + MODEL STATUS
      ===================================================================== */}

      <section
        aria-label="Prediction and model overview"
        className="grid gap-6 xl:grid-cols-[1.45fr_1fr]"
      >
        <PredictionCard
          onClick={
            handlePredictionNavigation
          }
        />

        <ModelStatus
          online={apiOnline}
        />
      </section>

      {/* ====================================================================
          MODEL INFORMATION
      ===================================================================== */}

      {modelInfo && (
        <section
          aria-labelledby="model-insight-heading"
          className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7"
        >
          <div className="flex flex-col gap-5 sm:flex-row sm:items-start">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-sky-50 text-sky-700">
              <Info
                size={21}
                aria-hidden="true"
              />
            </div>

            <div className="min-w-0">
              <h2
                id="model-insight-heading"
                className="text-base font-bold text-slate-950"
              >
                Model overview
              </h2>

              <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-600">
                The deployed{" "}
                <strong>
                  {algorithm}
                </strong>{" "}
                model uses{" "}
                <strong>
                  {featureCount}
                </strong>{" "}
                input{" "}
                {featureCount === 1
                  ? "feature"
                  : "features"}{" "}
                to estimate{" "}
                <strong>
                  {target}
                </strong>
                {r2Available ? (
                  <>
                    . The model achieved an
                    R² score of{" "}
                    <strong>
                      {formatNumber(
                        metrics?.r2_score,
                        4,
                      )}
                    </strong>{" "}
                    on the available
                    evaluation data.
                  </>
                ) : (
                  "."
                )}
              </p>

              <div className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Model
                  </p>

                  <p className="mt-1 break-words text-sm font-bold text-slate-900">
                    {modelName}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Algorithm
                  </p>

                  <p className="mt-1 break-words text-sm font-bold text-slate-900">
                    {algorithm}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Input Features
                  </p>

                  <p className="mt-1 text-sm font-bold text-slate-900">
                    {formatNumber(
                      featureCount,
                      0,
                    )}
                  </p>
                </div>

                <div className="rounded-xl bg-slate-50 p-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Prediction Unit
                  </p>

                  <p className="mt-1 break-words text-sm font-bold text-slate-900">
                    {targetUnit ||
                      UI.notAvailable}
                  </p>
                </div>
              </div>

              {/* VERSION */}

              <div className="mt-3 rounded-xl bg-slate-50 p-4">
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Model Version
                </p>

                <p className="mt-1 text-sm font-bold text-slate-900">
                  {version}
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {/* ====================================================================
          DATASET COVERAGE
      ===================================================================== */}

      {modelInfo && hasDataset && (
        <section
          aria-labelledby="dataset-heading"
          className="space-y-6"
        >
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-emerald-600">
              Dataset coverage
            </p>

            <h2
              id="dataset-heading"
              className="mt-1 text-xl font-bold text-slate-950"
            >
              Training data ranges
            </h2>

            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">
              These ranges are calculated directly
              from the current cleaned dataset
              returned by the machine-learning
              backend.
            </p>
          </div>

          {/* Dataset statistics */}

          <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
            {/* RECORDS */}

            <article className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700">
                  <Database
                    size={19}
                    aria-hidden="true"
                  />
                </div>

                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Dataset
                  </p>

                  <p className="mt-1 break-words text-lg font-bold text-slate-950">
                    {formatNumber(
                      datasetRecords,
                      0,
                    )}{" "}
                    records
                  </p>
                </div>
              </div>
            </article>

            {/* ENGINE SIZE */}

            <article className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-50 text-violet-700">
                  <Gauge
                    size={19}
                    aria-hidden="true"
                  />
                </div>

                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Engine Size
                  </p>

                  <p className="mt-1 break-words text-lg font-bold text-slate-950">
                    {hasEngineRange
                      ? formatRange(
                          engineStatistics?.min,
                          engineStatistics?.max,
                          getFeatureUnit(
                            "ENGINESIZE",
                          ),
                        )
                      : UI.notAvailable}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    {getFeatureLabel(
                      "ENGINESIZE",
                    )}
                  </p>
                </div>
              </div>
            </article>

            {/* FUEL CONSUMPTION */}

            <article className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-50 text-sky-700">
                  <Gauge
                    size={19}
                    aria-hidden="true"
                  />
                </div>

                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Fuel Consumption
                  </p>

                  <p className="mt-1 break-words text-lg font-bold text-slate-950">
                    {hasFuelRange
                      ? formatRange(
                          fuelStatistics?.min,
                          fuelStatistics?.max,
                          getFeatureUnit(
                            "FUELCONSUMPTION_COMB_MPG",
                          ),
                        )
                      : UI.notAvailable}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Combined fuel consumption
                  </p>
                </div>
              </div>
            </article>

            {/* TARGET */}

            <article className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-50 text-amber-700">
                  <BarChart3
                    size={19}
                    aria-hidden="true"
                  />
                </div>

                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Target Range
                  </p>

                  <p className="mt-1 break-words text-lg font-bold text-slate-950">
                    {hasTargetRange
                      ? formatRange(
                          targetStatistics?.min,
                          targetStatistics?.max,
                          targetUnit,
                        )
                      : UI.notAvailable}
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    {target}
                  </p>
                </div>
              </div>
            </article>
          </div>

          {/* FEATURE COVERAGE */}

          {features.length > 0 && (
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-100 text-slate-700">
                  <Target
                    size={19}
                    aria-hidden="true"
                  />
                </div>

                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Model Inputs
                  </p>

                  <p className="mt-1 text-sm font-bold text-slate-950">
                    {featureCount}{" "}
                    {featureCount === 1
                      ? "feature"
                      : "features"}{" "}
                    used by the model
                  </p>
                </div>
              </div>

              <div className="mt-5 flex flex-wrap gap-3">
                {features.map(
                  (feature) => {
                    const statistics =
                      getFeatureStatistics(
                        dataset,
                        feature,
                      );

                    return (
                      <div
                        key={feature}
                        className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3"
                      >
                        <p className="text-xs font-semibold text-slate-500">
                          {getFeatureLabel(
                            feature,
                          )}
                        </p>

                        <p className="mt-1 text-sm font-bold text-slate-900">
                          {statistics
                            ? formatRange(
                                statistics.min,
                                statistics.max,
                                getFeatureUnit(
                                  feature,
                                ),
                              )
                            : UI.notAvailable}
                        </p>
                      </div>
                    );
                  },
                )}
              </div>
            </div>
          )}
        </section>
      )}

      {/* ====================================================================
          DISCLAIMER
      ===================================================================== */}

      <footer className="border-t border-slate-200 pt-6">
        <p className="mx-auto max-w-4xl text-center text-xs leading-5 text-slate-500">
          Model predictions are estimates generated
          from machine-learning inference and should
          be interpreted according to the model
          documentation and evaluation results.
        </p>
      </footer>
    </div>
  );
}

export default HomePage;