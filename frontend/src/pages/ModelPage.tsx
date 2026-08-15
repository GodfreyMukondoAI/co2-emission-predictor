import {
  BarChart3,
  CheckCircle2,
  Code2,
  Cpu,
  Database,
  GitBranch,
  Info,
  Target,
  type LucideIcon,
} from "lucide-react";

import type { ModelInfoResponse } from "../types/prediction";

/* ==========================================================================
   TYPES
   ========================================================================== */

interface ModelPageProps {
  modelInfo: ModelInfoResponse | null;
  modelLoading: boolean;
}

/* ==========================================================================
   HELPERS
   ========================================================================== */

function isFiniteNumber(
  value: unknown,
): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value)
  );
}

function formatNumber(
  value: unknown,
  decimals = 4,
): string {
  if (!isFiniteNumber(value)) {
    return "—";
  }

  return value.toFixed(decimals);
}

function formatInteger(
  value: unknown,
): string {
  if (!isFiniteNumber(value)) {
    return "—";
  }

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);
}

function normalizeText(
  value: unknown,
): string {
  if (
    typeof value !== "string" ||
    value.trim().length === 0
  ) {
    return "—";
  }

  return value.trim();
}

function normalizeFeatures(
  value: unknown,
): string[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.filter(
    (feature): feature is string =>
      typeof feature === "string" &&
      feature.trim().length > 0,
  );
}

/* ==========================================================================
   LOADING SKELETON
   ========================================================================== */

function ModelPageSkeleton() {
  return (
    <div
      className="space-y-8 pb-10"
      aria-busy="true"
      aria-live="polite"
      aria-label="Loading model information"
    >
      {/* Header */}
      <section
        className="relative overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm"
        aria-hidden="true"
      >
        <div className="p-6 sm:p-8 lg:p-10">
          <div className="h-7 w-48 animate-pulse rounded-full bg-slate-200" />

          <div className="mt-5 h-10 w-96 max-w-full animate-pulse rounded-lg bg-slate-200" />

          <div className="mt-4 h-5 w-full max-w-2xl animate-pulse rounded bg-slate-100" />

          <div className="mt-2 h-5 w-3/4 max-w-xl animate-pulse rounded bg-slate-100" />
        </div>
      </section>

      {/* Overview */}
      <section
        className="grid gap-6 lg:grid-cols-2"
        aria-hidden="true"
      >
        {[1, 2].map((item) => (
          <div
            key={item}
            className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
          >
            <div className="h-10 w-10 animate-pulse rounded-xl bg-slate-200" />

            <div className="mt-5 h-6 w-32 animate-pulse rounded bg-slate-200" />

            <div className="mt-5 h-8 w-64 max-w-full animate-pulse rounded bg-slate-200" />

            <div className="mt-4 h-16 w-full animate-pulse rounded bg-slate-100" />
          </div>
        ))}
      </section>

      {/* Features */}
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
        aria-hidden="true"
      >
        <div className="h-10 w-10 animate-pulse rounded-xl bg-slate-200" />

        <div className="mt-5 h-6 w-40 animate-pulse rounded bg-slate-200" />

        <div className="mt-6 grid gap-4 lg:grid-cols-2">
          <div className="h-36 animate-pulse rounded-xl bg-slate-100" />

          <div className="h-36 animate-pulse rounded-xl bg-slate-100" />
        </div>
      </section>

      {/* Evaluation */}
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
        aria-hidden="true"
      >
        <div className="h-10 w-10 animate-pulse rounded-xl bg-slate-200" />

        <div className="mt-5 h-6 w-40 animate-pulse rounded bg-slate-200" />

        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((item) => (
            <div
              key={item}
              className="h-28 animate-pulse rounded-xl bg-slate-100"
            />
          ))}
        </div>
      </section>

      {/* Dataset */}
      <section
        className="space-y-4"
        aria-hidden="true"
      >
        <div className="h-6 w-44 animate-pulse rounded bg-slate-200" />

        <div className="h-4 w-80 max-w-full animate-pulse rounded bg-slate-100" />

        <div className="grid gap-6 md:grid-cols-3">
          {[1, 2, 3].map((item) => (
            <div
              key={item}
              className="h-28 animate-pulse rounded-2xl bg-white shadow-sm"
            />
          ))}
        </div>
      </section>

      {/* Configuration */}
      <section
        className="rounded-2xl border border-slate-800 bg-slate-950 p-6"
        aria-hidden="true"
      >
        <div className="h-10 w-10 animate-pulse rounded-xl bg-white/10" />

        <div className="mt-5 h-6 w-40 animate-pulse rounded bg-white/10" />

        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5, 6].map((item) => (
            <div
              key={item}
              className="h-24 animate-pulse rounded-xl bg-white/5"
            />
          ))}
        </div>
      </section>
    </div>
  );
}

/* ==========================================================================
   UNAVAILABLE STATE
   ========================================================================== */

function ModelUnavailable() {
  return (
    <div className="flex min-h-[420px] items-center justify-center py-10">
      <section
        className="w-full max-w-xl rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm"
        role="alert"
        aria-labelledby="model-unavailable-title"
      >
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-50 text-amber-600">
          <Info
            size={24}
            aria-hidden="true"
          />
        </div>

        <h1
          id="model-unavailable-title"
          className="mt-5 text-xl font-bold text-slate-950"
        >
          Model information unavailable
        </h1>

        <p className="mt-3 text-sm leading-6 text-slate-500">
          The application could not retrieve model metadata
          from the machine-learning backend.
        </p>

        <div className="mt-5 rounded-xl border border-amber-200 bg-amber-50 p-4 text-left">
          <p className="text-xs font-semibold uppercase tracking-wider text-amber-700">
            Troubleshooting
          </p>

          <p className="mt-2 text-sm leading-6 text-amber-800">
            Make sure the FastAPI server is running, the
            <code className="mx-1 rounded bg-amber-100 px-1 py-0.5 font-mono text-xs">
              /api/model
            </code>
            endpoint is available, and the frontend API URL
            points to the correct backend.
          </p>
        </div>
      </section>
    </div>
  );
}

/* ==========================================================================
   MAIN COMPONENT
   ========================================================================== */

function ModelPage({
  modelInfo,
  modelLoading,
}: ModelPageProps) {
  if (modelLoading) {
    return <ModelPageSkeleton />;
  }

  if (!modelInfo) {
    return <ModelUnavailable />;
  }

  /*
   * All model information below comes from the FastAPI backend.
   *
   * Nothing related to:
   * - model name
   * - algorithm
   * - version
   * - features
   * - metrics
   * - dataset statistics
   *
   * is hardcoded into the UI.
   */

  const modelName = normalizeText(
    modelInfo.model_name,
  );

  const algorithm = normalizeText(
    modelInfo.algorithm,
  );

  const version = normalizeText(
    modelInfo.version,
  );

  const target = normalizeText(
    modelInfo.target,
  );

  const targetUnit = normalizeText(
    modelInfo.target_unit,
  );

  const features = normalizeFeatures(
    modelInfo.features,
  );

  const metrics = modelInfo.metrics ?? null;

  const dataset = modelInfo.dataset ?? null;

  const hasMetrics =
    metrics !== null &&
    (
      isFiniteNumber(metrics.r2_score) ||
      isFiniteNumber(metrics.mae) ||
      isFiniteNumber(metrics.rmse)
    );

  const hasDataset =
    dataset !== null;

  const backendFeatureCount =
    dataset?.feature_count;

  const featureCount =
    isFiniteNumber(backendFeatureCount)
      ? backendFeatureCount
      : features.length;

  return (
    <div className="space-y-8 pb-10">

      {/* ==================================================================
          HEADER
      ================================================================== */}

      <section
        aria-labelledby="model-page-title"
        className="relative overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-sm"
      >
        <div
          className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-emerald-100/60 blur-3xl"
          aria-hidden="true"
        />

        <div
          className="absolute -bottom-24 left-1/3 h-56 w-56 rounded-full bg-sky-100/40 blur-3xl"
          aria-hidden="true"
        />

        <div className="relative p-6 sm:p-8 lg:p-10">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">

            <div className="max-w-3xl">

              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-emerald-700">
                <Cpu
                  size={14}
                  aria-hidden="true"
                />

                Machine Learning Model
              </div>

              <h1
                id="model-page-title"
                className="mt-5 break-words text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl"
              >
                {modelName}
              </h1>

              <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-600 sm:text-base">
                Technical information retrieved directly
                from the deployed machine-learning model,
                including its algorithm, input features,
                evaluation metrics, dataset statistics,
                and prediction target.
              </p>
            </div>

            <div className="flex shrink-0 items-center gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 px-4 py-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100 text-emerald-700">
                <CheckCircle2
                  size={20}
                  aria-hidden="true"
                />
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Model Version
                </p>

                <p className="mt-0.5 break-all text-sm font-bold text-emerald-700">
                  {version}
                </p>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* ==================================================================
          MODEL OVERVIEW
      ================================================================== */}

      <section
        aria-labelledby="model-overview-heading"
        className="grid gap-6 lg:grid-cols-2"
      >

        {/* Algorithm */}

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

          <div className="flex items-center gap-3">

            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-700">
              <Cpu
                size={20}
                aria-hidden="true"
              />
            </div>

            <div>
              <h2
                id="model-overview-heading"
                className="font-bold text-slate-950"
              >
                Algorithm
              </h2>

              <p className="text-sm text-slate-500">
                Machine-learning algorithm
              </p>
            </div>

          </div>

          <p className="mt-6 break-words text-2xl font-bold text-slate-950">
            {algorithm}
          </p>

          <p className="mt-3 text-sm leading-6 text-slate-500">
            The algorithm shown here is supplied by the
            deployed machine-learning model metadata.
          </p>

        </div>

        {/* Model metadata */}

        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

          <div className="flex items-center gap-3">

            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-50 text-violet-700">
              <GitBranch
                size={20}
                aria-hidden="true"
              />
            </div>

            <div>
              <h2 className="font-bold text-slate-950">
                Model Metadata
              </h2>

              <p className="text-sm text-slate-500">
                Backend-provided information
              </p>
            </div>

          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-2">

            <MetadataItem
              label="Model"
              value={modelName}
            />

            <MetadataItem
              label="Version"
              value={version}
            />

            <MetadataItem
              label="Target"
              value={target}
            />

            <MetadataItem
              label="Target Unit"
              value={targetUnit}
            />

          </div>
        </div>

      </section>

      {/* ==================================================================
          MODEL FEATURES
      ================================================================== */}

      <section
        aria-labelledby="features-heading"
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7"
      >

        <div className="flex items-center gap-3">

          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sky-50 text-sky-700">
            <Target
              size={20}
              aria-hidden="true"
            />
          </div>

          <div>
            <h2
              id="features-heading"
              className="font-bold text-slate-950"
            >
              Model Features
            </h2>

            <p className="text-sm text-slate-500">
              Input variables used to generate predictions
            </p>
          </div>

        </div>

        <div className="mt-6 grid gap-4 lg:grid-cols-[1fr_auto_1fr]">

          {/* Input features */}

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">

            <div className="flex items-center justify-between gap-4">

              <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Input Features
              </p>

              <span className="rounded-full bg-white px-2.5 py-1 text-xs font-semibold text-slate-500">
                {formatInteger(featureCount)}
              </span>

            </div>

            <div className="mt-4 space-y-3">

              {features.length > 0 ? (
                features.map((feature, index) => (
                  <div
                    key={`${feature}-${index}`}
                    className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-4 py-3"
                  >
                    <div
                      className="h-2 w-2 shrink-0 rounded-full bg-emerald-500"
                      aria-hidden="true"
                    />

                    <code className="break-all text-xs font-semibold text-slate-700">
                      {feature}
                    </code>
                  </div>
                ))
              ) : (
                <div className="rounded-lg border border-dashed border-slate-300 bg-white p-4">
                  <p className="text-sm text-slate-500">
                    No input feature metadata was returned
                    by the backend.
                  </p>
                </div>
              )}

            </div>
          </div>

          {/* Connector */}

          <div
            className="hidden items-center justify-center lg:flex"
            aria-hidden="true"
          >
            <div className="rounded-full border border-slate-200 bg-white p-3 text-slate-400 shadow-sm">
              <Code2 size={18} />
            </div>
          </div>

          {/* Prediction target */}

          <div className="rounded-xl border border-slate-200 bg-slate-50 p-5">

            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Prediction Target
            </p>

            <div className="mt-4 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-4">

              <code className="break-all text-sm font-bold text-emerald-700">
                {target}
              </code>

              <p className="mt-2 break-words text-xs text-emerald-700/80">
                Output unit: {targetUnit}
              </p>

            </div>

          </div>

        </div>
      </section>

      {/* ==================================================================
          MODEL EVALUATION
      ================================================================== */}

      <section
        aria-labelledby="evaluation-heading"
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm sm:p-7"
      >

        <div className="flex items-center gap-3">

          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-50 text-amber-700">
            <BarChart3
              size={20}
              aria-hidden="true"
            />
          </div>

          <div>
            <h2
              id="evaluation-heading"
              className="font-bold text-slate-950"
            >
              Model Evaluation
            </h2>

            <p className="text-sm text-slate-500">
              Performance metrics returned by the model API
            </p>
          </div>

        </div>

        {hasMetrics && metrics ? (
          <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

            <Metric
              label="R² Score"
              value={formatNumber(
                metrics.r2_score,
                4,
              )}
              description="Explained variance"
            />

            <Metric
              label="MAE"
              value={
                isFiniteNumber(metrics.mae)
                  ? `${formatNumber(metrics.mae, 2)} ${targetUnit}`
                  : "—"
              }
              description="Mean absolute error"
            />

            <Metric
              label="RMSE"
              value={
                isFiniteNumber(metrics.rmse)
                  ? `${formatNumber(metrics.rmse, 2)} ${targetUnit}`
                  : "—"
              }
              description="Root mean squared error"
            />

          </div>
        ) : (
          <div
            className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-5"
            role="status"
          >
            <p className="text-sm font-semibold text-amber-800">
              Model evaluation data is unavailable.
            </p>

            <p className="mt-1 text-xs leading-5 text-amber-700">
              The backend did not return valid evaluation
              metrics for this model.
            </p>
          </div>
        )}

      </section>

      {/* ==================================================================
          DATASET INFORMATION
      ================================================================== */}

      <section
        aria-labelledby="dataset-heading"
        className="space-y-4"
      >

        <div>
          <h2
            id="dataset-heading"
            className="text-lg font-bold text-slate-950"
          >
            Dataset Information
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Dataset characteristics supplied by the model API.
          </p>
        </div>

        {hasDataset && dataset ? (
          <>

            <div className="grid gap-6 md:grid-cols-3">

              <DataCard
                icon={Database}
                label="Dataset Records"
                value={formatInteger(
                  dataset.records,
                )}
              />

              <DataCard
                icon={Target}
                label="Feature Count"
                value={formatInteger(
                  dataset.feature_count,
                )}
              />

              <DataCard
                icon={GitBranch}
                label="Model Features"
                value={formatInteger(
                  featureCount,
                )}
              />

            </div>

            <div className="grid gap-6 lg:grid-cols-3">

              <RangeCard
                title="Engine Size"
                min={dataset.engine_size_min}
                max={dataset.engine_size_max}
                unit="L"
              />

              <RangeCard
                title="Fuel Consumption"
                min={dataset.fuel_consumption_mpg_min}
                max={dataset.fuel_consumption_mpg_max}
                unit="MPG"
              />

              <RangeCard
                title="CO₂ Emissions"
                min={dataset.co2_min}
                max={dataset.co2_max}
                unit={targetUnit}
              />

            </div>

          </>
        ) : (
          <div
            className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
            role="status"
          >

            <div className="flex items-start gap-4">

              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-slate-100 text-slate-600">
                <Database
                  size={19}
                  aria-hidden="true"
                />
              </div>

              <div>

                <h3 className="font-bold text-slate-950">
                  Dataset information unavailable
                </h3>

                <p className="mt-1 text-sm leading-6 text-slate-500">
                  Dataset statistics were not included
                  in the model metadata returned by the
                  API.
                </p>

              </div>

            </div>

          </div>
        )}

      </section>

      {/* ==================================================================
          MODEL CONFIGURATION
      ================================================================== */}

      <section
        aria-labelledby="configuration-heading"
        className="rounded-2xl border border-slate-800 bg-slate-950 p-6 text-white shadow-sm sm:p-7"
      >

        <div className="flex items-center gap-3">

          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10">
            <Code2
              size={20}
              aria-hidden="true"
            />
          </div>

          <div>
            <h2
              id="configuration-heading"
              className="font-bold"
            >
              Model Configuration
            </h2>

            <p className="text-sm text-slate-400">
              Runtime configuration supplied by the backend
            </p>
          </div>

        </div>

        <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">

          <ConfigurationItem
            label="Model"
            value={modelName}
          />

          <ConfigurationItem
            label="Algorithm"
            value={algorithm}
          />

          <ConfigurationItem
            label="Model Version"
            value={version}
          />

          <ConfigurationItem
            label="Input Features"
            value={formatInteger(featureCount)}
          />

          <ConfigurationItem
            label="Prediction Target"
            value={target}
            highlight
          />

          <ConfigurationItem
            label="Prediction Unit"
            value={targetUnit}
            highlight
          />

        </div>
      </section>

      {/* ==================================================================
          DISCLAIMER
      ================================================================== */}

      <footer className="border-t border-slate-200 pt-6">

        <p className="mx-auto max-w-4xl text-center text-xs leading-5 text-slate-500">
          Model evaluation metrics describe performance
          on the evaluation data returned by the backend.
          Predictions are estimates generated by a
          machine-learning model and should not be interpreted
          as official certification, regulatory measurements,
          or laboratory results.
        </p>

      </footer>

    </div>
  );
}

/* ==========================================================================
   METRIC COMPONENT
   ========================================================================== */

interface MetricProps {
  label: string;
  value: string;
  description: string;
}

function Metric({
  label,
  value,
  description,
}: MetricProps) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-5 transition-shadow hover:shadow-sm">

      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </p>

      <p className="mt-2 break-words text-xl font-bold text-slate-950">
        {value}
      </p>

      <p className="mt-1 text-xs text-slate-500">
        {description}
      </p>

    </div>
  );
}

/* ==========================================================================
   DATA CARD
   ========================================================================== */

interface DataCardProps {
  icon: LucideIcon;
  label: string;
  value: string;
}

function DataCard({
  icon: Icon,
  label,
  value,
}: DataCardProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm transition-shadow hover:shadow-md">

      <div className="flex items-center gap-3">

        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-slate-50 text-slate-700">
          <Icon
            size={19}
            aria-hidden="true"
          />
        </div>

        <div className="min-w-0">

          <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            {label}
          </p>

          <p className="mt-1 break-words text-lg font-bold text-slate-950">
            {value}
          </p>

        </div>

      </div>

    </div>
  );
}

/* ==========================================================================
   RANGE CARD
   ========================================================================== */

interface RangeCardProps {
  title: string;
  min: number | null | undefined;
  max: number | null | undefined;
  unit: string;
}

function RangeCard({
  title,
  min,
  max,
  unit,
}: RangeCardProps) {
  const hasValidRange =
    isFiniteNumber(min) &&
    isFiniteNumber(max);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

      <div className="flex items-center justify-between gap-4">

        <h3 className="font-bold text-slate-950">
          {title}
        </h3>

        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-500">
          Range
        </span>

      </div>

      {hasValidRange ? (
        <div className="mt-5 flex items-end justify-between gap-4">

          <div>

            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Minimum
            </p>

            <p className="mt-1 text-xl font-bold text-slate-950">
              {formatNumber(min, 1)}
            </p>

          </div>

          <div
            className="pb-1 text-slate-300"
            aria-hidden="true"
          >
            →
          </div>

          <div className="text-right">

            <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Maximum
            </p>

            <p className="mt-1 text-xl font-bold text-slate-950">
              {formatNumber(max, 1)}
            </p>

          </div>

        </div>
      ) : (
        <p className="mt-5 text-sm text-slate-500">
          Range information unavailable.
        </p>
      )}

      <p className="mt-4 break-words text-xs text-slate-400">
        Unit: {normalizeText(unit)}
      </p>

    </div>
  );
}

/* ==========================================================================
   CONFIGURATION ITEM
   ========================================================================== */

interface ConfigurationItemProps {
  label: string;
  value: string;
  highlight?: boolean;
}

function ConfigurationItem({
  label,
  value,
  highlight = false,
}: ConfigurationItemProps) {
  return (
    <div className="rounded-xl bg-white/5 p-5">

      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </p>

      <p
        className={[
          "mt-2 break-words text-sm font-bold",
          highlight
            ? "text-emerald-300"
            : "text-white",
        ].join(" ")}
      >
        {value}
      </p>

    </div>
  );
}

/* ==========================================================================
   METADATA ITEM
   ========================================================================== */

interface MetadataItemProps {
  label: string;
  value: string;
}

function MetadataItem({
  label,
  value,
}: MetadataItemProps) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">

      <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
        {label}
      </p>

      <p className="mt-1 break-words text-sm font-bold text-slate-900">
        {value}
      </p>

    </div>
  );
}

/* ==========================================================================
   EXPORT
   ========================================================================== */

export default ModelPage;