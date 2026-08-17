/**
 * ============================================================================
 * CO₂ EMISSION PREDICTOR
 * ============================================================================
 *
 * DatasetPage
 *
 * Production-ready dataset information page.
 *
 * Architecture:
 *
 * FastAPI is the single source of truth for:
 * - Dataset metadata
 * - Dataset statistics
 * - Feature definitions
 * - Model information
 * - Data quality information
 *
 * Frontend responsibilities:
 * - Request dataset metadata
 * - Validate the runtime API response
 * - Safely format values
 * - Handle loading/error/refresh states
 * - Cancel stale requests
 * - Enforce request timeout
 * - Never trust unvalidated API data
 *
 * Environment variables:
 *
 * VITE_API_URL
 *   Development:
 *     http://127.0.0.1:8000
 *
 *   Production:
 *     https://co2-emission-predictor-ys3q.onrender.com
 *
 * VITE_DATASET_METADATA_ENDPOINT
 *   Optional.
 *   Defaults:
 *     /dataset/metadata
 *
 * VITE_API_TIMEOUT
 *   Optional.
 *   Defaults:
 *     30000
 *
 * ============================================================================
 */

import {
  AlertCircle,
  BarChart3,
  CheckCircle2,
  Database,
  Fuel,
  Gauge,
  Info,
  RefreshCw,
  Rows3,
  Target,
  type LucideIcon,
} from "lucide-react";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/* ============================================================================
   TYPES
============================================================================ */

type DatasetFeatureRole = "Feature" | "Target";

interface DatasetFeature {
  column: string;
  role: DatasetFeatureRole;
  unit: string;
  description: string;
}

interface DatasetStatisticResponse {
  label: string;
  value: string;
  description: string;
  type?: string | null;
}

interface DatasetStatistic {
  label: string;
  value: string;
  description: string;
  icon: LucideIcon;
}

interface DatasetModel {
  name: string;
  description: string;
  features: string[];
  target: string;
  targetUnit: string;
}

interface DatasetNumericStatistics {
  min: number;
  max: number;
  mean: number;
  median: number;
  std: number;
}

interface DatasetTargetStatistics
  extends DatasetNumericStatistics {
  column: string;
}

interface DatasetMetadataResponse {
  datasetName: string;
  title: string;
  description: string;

  recordCount: number;
  columnCount: number;
  columnNames: string[];

  missingValues: number;

  validRecordCount: number;
  invalidRecordCount: number;

  featureCount: number;
  targetCount: number;

  validationPercentage: number;

  features: DatasetFeature[];

  featureStatistics: Record<
    string,
    DatasetNumericStatistics
  >;

  targetStatistics: DatasetTargetStatistics;

  statistics: DatasetStatisticResponse[];

  model: DatasetModel;

  lastUpdated?: string | null;
}

interface ApiErrorResponse {
  detail?: unknown;
  message?: unknown;
  error?: unknown;
  code?: unknown;
}

/* ============================================================================
   ENVIRONMENT CONFIGURATION
============================================================================ */

const DEFAULT_API_URL =
  "http://127.0.0.1:8000";

const DEFAULT_DATASET_ENDPOINT =
  "/dataset/metadata";

const DEFAULT_TIMEOUT_MS = 30000;

const API_BASE_URL = normalizeBaseUrl(
  import.meta.env.VITE_API_URL ||
    DEFAULT_API_URL,
);

const DATASET_METADATA_PATH =
  normalizeEndpointPath(
    import.meta.env
      .VITE_DATASET_METADATA_ENDPOINT ||
      DEFAULT_DATASET_ENDPOINT,
  );

const REQUEST_TIMEOUT_MS =
  parseTimeout(
    import.meta.env.VITE_API_TIMEOUT ||
      String(DEFAULT_TIMEOUT_MS),
  );

const DATASET_METADATA_ENDPOINT =
  `${API_BASE_URL}${DATASET_METADATA_PATH}`;

/* ============================================================================
   ENVIRONMENT HELPERS
============================================================================ */

function normalizeBaseUrl(
  value: string,
): string {
  const normalized = value.trim();

  if (!normalized) {
    return DEFAULT_API_URL;
  }

  return normalized.replace(
    /\/+$/,
    "",
  );
}

function normalizeEndpointPath(
  value: string,
): string {
  const normalized = value.trim();

  if (!normalized) {
    return DEFAULT_DATASET_ENDPOINT;
  }

  return normalized.startsWith("/")
    ? normalized
    : `/${normalized}`;
}

function parseTimeout(
  value: string,
): number {
  const timeout = Number(value);

  if (
    !Number.isFinite(timeout) ||
    timeout <= 0
  ) {
    return DEFAULT_TIMEOUT_MS;
  }

  return Math.max(
    1000,
    Math.floor(timeout),
  );
}

/* ============================================================================
   DEVELOPMENT DIAGNOSTICS
============================================================================ */

if (import.meta.env.DEV) {
  console.debug(
    "[CO₂ Emission Predictor] Dataset API configuration",
    {
      endpoint:
        DATASET_METADATA_ENDPOINT,
      timeout:
        REQUEST_TIMEOUT_MS,
      environment:
        import.meta.env.MODE,
    },
  );
}

/* ============================================================================
   RUNTIME TYPE HELPERS
============================================================================ */

function isObject(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function isFiniteNumber(
  value: unknown,
): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value)
  );
}

function isNonEmptyString(
  value: unknown,
): value is string {
  return (
    typeof value === "string" &&
    value.trim().length > 0
  );
}

function isStringArray(
  value: unknown,
): value is string[] {
  return (
    Array.isArray(value) &&
    value.every(isNonEmptyString)
  );
}

function isNonNegativeInteger(
  value: unknown,
): value is number {
  return (
    isFiniteNumber(value) &&
    Number.isInteger(value) &&
    value >= 0
  );
}

function isPercentage(
  value: unknown,
): value is number {
  return (
    isFiniteNumber(value) &&
    value >= 0 &&
    value <= 100
  );
}

function hasUniqueStrings(
  values: string[],
): boolean {
  return (
    new Set(values).size ===
    values.length
  );
}

/* ============================================================================
   FEATURE VALIDATION
============================================================================ */

function isDatasetFeature(
  value: unknown,
): value is DatasetFeature {
  if (!isObject(value)) {
    return false;
  }

  return (
    isNonEmptyString(value.column) &&
    (
      value.role === "Feature" ||
      value.role === "Target"
    ) &&
    typeof value.unit === "string" &&
    typeof value.description ===
      "string"
  );
}

/* ============================================================================
   STATISTIC VALIDATION
============================================================================ */

function isDatasetStatisticResponse(
  value: unknown,
): value is DatasetStatisticResponse {
  if (!isObject(value)) {
    return false;
  }

  return (
    isNonEmptyString(value.label) &&
    isNonEmptyString(value.value) &&
    isNonEmptyString(
      value.description,
    ) &&
    (
      value.type === undefined ||
      value.type === null ||
      isNonEmptyString(value.type)
    )
  );
}

/* ============================================================================
   NUMERIC STATISTICS VALIDATION
============================================================================ */

function isDatasetNumericStatistics(
  value: unknown,
): value is DatasetNumericStatistics {
  if (!isObject(value)) {
    return false;
  }

  const min = value.min;
  const max = value.max;
  const mean = value.mean;
  const median = value.median;
  const std = value.std;

  return (
    isFiniteNumber(min) &&
    isFiniteNumber(max) &&
    isFiniteNumber(mean) &&
    isFiniteNumber(median) &&
    isFiniteNumber(std) &&
    min <= max &&
    std >= 0
  );
}

/* ============================================================================
   FEATURE STATISTICS VALIDATION
============================================================================ */

function isFeatureStatistics(
  value: unknown,
): value is Record<
  string,
  DatasetNumericStatistics
> {
  if (!isObject(value)) {
    return false;
  }

  const entries =
    Object.entries(value);

  return entries.every(
    ([column, statistics]) =>
      isNonEmptyString(column) &&
      isDatasetNumericStatistics(
        statistics,
      ),
  );
}

/* ============================================================================
   TARGET STATISTICS VALIDATION
============================================================================ */

function isTargetStatistics(
  value: unknown,
): value is DatasetTargetStatistics {
  if (!isObject(value)) {
    return false;
  }

  return (
    isNonEmptyString(value.column) &&
    isDatasetNumericStatistics(value)
  );
}

/* ============================================================================
   MODEL VALIDATION
============================================================================ */

function isDatasetModel(
  value: unknown,
): value is DatasetModel {
  if (!isObject(value)) {
    return false;
  }

  return (
    isNonEmptyString(value.name) &&
    isNonEmptyString(
      value.description,
    ) &&
    isStringArray(value.features) &&
    value.features.length > 0 &&
    hasUniqueStrings(value.features) &&
    isNonEmptyString(value.target) &&
    isNonEmptyString(value.targetUnit)
  );
}

/* ============================================================================
   FULL DATASET VALIDATION
============================================================================ */

function isDatasetMetadataResponse(
  value: unknown,
): value is DatasetMetadataResponse {
  if (!isObject(value)) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Basic metadata
  -------------------------------------------------------------------------- */

  if (
    !isNonEmptyString(
      value.datasetName,
    ) ||
    !isNonEmptyString(value.title) ||
    typeof value.description !==
      "string"
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Numeric counts
  -------------------------------------------------------------------------- */

  const recordCount =
    value.recordCount;

  const columnCount =
    value.columnCount;

  const missingValues =
    value.missingValues;

  const validRecordCount =
    value.validRecordCount;

  const invalidRecordCount =
    value.invalidRecordCount;

  const featureCount =
    value.featureCount;

  const targetCount =
    value.targetCount;

  if (
    !isNonNegativeInteger(
      recordCount,
    ) ||
    !isNonNegativeInteger(
      columnCount,
    ) ||
    !isNonNegativeInteger(
      missingValues,
    ) ||
    !isNonNegativeInteger(
      validRecordCount,
    ) ||
    !isNonNegativeInteger(
      invalidRecordCount,
    ) ||
    !isNonNegativeInteger(
      featureCount,
    ) ||
    !isNonNegativeInteger(
      targetCount,
    )
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Dataset must contain records
  -------------------------------------------------------------------------- */

  if (recordCount <= 0) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Validation percentage
  -------------------------------------------------------------------------- */

  const validationPercentage =
    value.validationPercentage;

  if (
    !isPercentage(
      validationPercentage,
    )
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Column names
  -------------------------------------------------------------------------- */

  const columnNames =
    value.columnNames;

  if (
    !isStringArray(columnNames) ||
    !hasUniqueStrings(columnNames)
  ) {
    return false;
  }

  if (
    columnCount !==
    columnNames.length
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Features
  -------------------------------------------------------------------------- */

  const features =
    value.features;

  if (
    !Array.isArray(features) ||
    features.length === 0 ||
    !features.every(
      isDatasetFeature,
    )
  ) {
    return false;
  }

  const featureColumns =
    features
      .filter(
        (feature) =>
          feature.role === "Feature",
      )
      .map(
        (feature) =>
          feature.column,
      );

  const targetColumns =
    features
      .filter(
        (feature) =>
          feature.role === "Target",
      )
      .map(
        (feature) =>
          feature.column,
      );

  if (
    !hasUniqueStrings(
      features.map(
        (feature) =>
          feature.column,
      ),
    )
  ) {
    return false;
  }

  if (
    featureColumns.length !==
    featureCount
  ) {
    return false;
  }

  if (
    targetColumns.length !==
    targetCount
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Model
  -------------------------------------------------------------------------- */

  const model =
    value.model;

  if (!isDatasetModel(model)) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Model feature integrity
  -------------------------------------------------------------------------- */

  if (
    model.features.length !==
    featureCount
  ) {
    return false;
  }

  if (
    !model.features.every(
      (feature) =>
        featureColumns.includes(
          feature,
        ),
    )
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Target integrity
  -------------------------------------------------------------------------- */

  if (
    !columnNames.includes(
      model.target,
    )
  ) {
    return false;
  }

  if (
    targetCount !== 1
  ) {
    return false;
  }

  if (
    targetColumns.length !== 1
  ) {
    return false;
  }

  if (
    targetColumns[0] !==
    model.target
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Statistics
  -------------------------------------------------------------------------- */

  const statistics =
    value.statistics;

  if (
    !Array.isArray(statistics) ||
    !statistics.every(
      isDatasetStatisticResponse,
    )
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Feature statistics
  -------------------------------------------------------------------------- */

  const featureStatistics =
    value.featureStatistics;

  if (
    !isFeatureStatistics(
      featureStatistics,
    )
  ) {
    return false;
  }

  /*
   * Every model feature must have
   * corresponding numeric statistics.
   */
  for (
    const feature of model.features
  ) {
    if (
      !Object.prototype.hasOwnProperty.call(
        featureStatistics,
        feature,
      )
    ) {
      return false;
    }
  }

  /* --------------------------------------------------------------------------
     Target statistics
  -------------------------------------------------------------------------- */

  const targetStatistics =
    value.targetStatistics;

  if (
    !isTargetStatistics(
      targetStatistics,
    )
  ) {
    return false;
  }

  if (
    targetStatistics.column !==
    model.target
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Missing values integrity
  -------------------------------------------------------------------------- */

  if (
    missingValues >
    recordCount * columnCount
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Record integrity
  -------------------------------------------------------------------------- */

  if (
    validRecordCount +
      invalidRecordCount !==
    recordCount
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Validation percentage integrity
  -------------------------------------------------------------------------- */

  const calculatedPercentage =
    (
      validRecordCount /
      recordCount
    ) *
    100;

  if (
    Math.abs(
      calculatedPercentage -
        validationPercentage,
    ) > 0.2
  ) {
    return false;
  }

  /* --------------------------------------------------------------------------
     Last updated
  -------------------------------------------------------------------------- */

  if (
    value.lastUpdated !==
      undefined &&
    value.lastUpdated !== null &&
    !isNonEmptyString(
      value.lastUpdated,
    )
  ) {
    return false;
  }

  return true;
}

/* ============================================================================
   ICON RESOLUTION
============================================================================ */

function getStatisticIcon(
  type?: string | null,
): LucideIcon {
  switch (
    type?.trim().toLowerCase()
  ) {
    case "records":
      return Rows3;

    case "engine-size":
    case "engine_size":
      return Gauge;

    case "fuel-consumption":
    case "fuel_consumption":
      return Fuel;

    case "target":
      return Target;

    default:
      return Database;
  }
}

/* ============================================================================
   FORMATTERS
============================================================================ */

function formatNumber(
  value: number,
): string {
  if (!Number.isFinite(value)) {
    return "—";
  }

  return new Intl.NumberFormat(
    undefined,
    {
      maximumFractionDigits: 2,
    },
  ).format(value);
}

function formatPercentage(
  value: number,
): string {
  if (!Number.isFinite(value)) {
    return "—";
  }

  const safeValue =
    Math.min(
      Math.max(value, 0),
      100,
    );

  return `${safeValue.toFixed(1)}%`;
}

function formatDate(
  value?: string | null,
): string | null {
  if (!value) {
    return null;
  }

  const date = new Date(value);

  if (
    Number.isNaN(
      date.getTime(),
    )
  ) {
    return null;
  }

  try {
    return new Intl.DateTimeFormat(
      undefined,
      {
        dateStyle: "medium",
        timeStyle: "short",
      },
    ).format(date);
  } catch {
    return null;
  }
}

/* ============================================================================
   API ERROR HANDLING
============================================================================ */

function getApiErrorMessage(
  body: unknown,
  status: number,
): string {
  if (isObject(body)) {
    const errorBody =
      body as ApiErrorResponse;

    const candidates = [
      errorBody.detail,
      errorBody.message,
      errorBody.error,
    ];

    for (
      const candidate of candidates
    ) {
      if (
        isNonEmptyString(candidate)
      ) {
        return candidate;
      }
    }
  }

  switch (status) {
    case 400:
      return "The dataset metadata request was invalid.";

    case 401:
      return "Authentication is required to access dataset metadata.";

    case 403:
      return "You are not authorized to access dataset metadata.";

    case 404:
      return "The dataset metadata endpoint was not found.";

    case 408:
      return "The dataset metadata request timed out.";

    case 429:
      return "Too many requests were sent to the API. Please try again shortly.";

    case 500:
      return "The machine-learning API encountered an internal server error.";

    case 502:
    case 503:
    case 504:
      return "The machine-learning API is temporarily unavailable. Please try again shortly.";

    default:
      return `Dataset API request failed with HTTP ${status}.`;
  }
}

/* ============================================================================
   ERROR TYPES
============================================================================ */

class DatasetApiError
  extends Error {
  readonly status?: number;
  readonly code?: string;

  constructor(
    message: string,
    options?: {
      status?: number;
      code?: string;
    },
  ) {
    super(message);

    this.name =
      "DatasetApiError";

    this.status =
      options?.status;

    this.code =
      options?.code;
  }
}

class DatasetApiTimeoutError
  extends DatasetApiError {
  constructor() {
    super(
      "The dataset API request timed out. Please try again.",
      {
        code:
          "REQUEST_TIMEOUT",
      },
    );

    this.name =
      "DatasetApiTimeoutError";
  }
}

/* ============================================================================
   ABORT HELPERS
============================================================================ */

function isAbortError(
  error: unknown,
): boolean {
  return (
    error instanceof DOMException &&
    error.name === "AbortError"
  );
}

/* ============================================================================
   API CLIENT
============================================================================ */

async function fetchDatasetMetadata(
  signal: AbortSignal,
): Promise<DatasetMetadataResponse> {
  const timeoutController =
    new AbortController();

  let timedOut = false;

  const timeoutId =
    window.setTimeout(() => {
      timedOut = true;
      timeoutController.abort();
    }, REQUEST_TIMEOUT_MS);

  const abortHandler = () => {
    timeoutController.abort();
  };

  if (signal.aborted) {
    timeoutController.abort();
  } else {
    signal.addEventListener(
      "abort",
      abortHandler,
      {
        once: true,
      },
    );
  }

  try {
    const response =
      await fetch(
        DATASET_METADATA_ENDPOINT,
        {
          method: "GET",

          headers: {
            Accept:
              "application/json",
          },

          credentials: "omit",

          cache: "no-store",

          redirect: "error",

          signal:
            timeoutController.signal,
        },
      );

    const contentType =
      (
        response.headers.get(
          "content-type",
        ) ?? ""
      ).toLowerCase();

    let responseBody: unknown =
      null;

    if (
      contentType.includes(
        "application/json",
      )
    ) {
      try {
        responseBody =
          await response.json();
      } catch {
        throw new DatasetApiError(
          "The dataset API returned invalid JSON.",
          {
            status:
              response.status,
            code:
              "INVALID_JSON",
          },
        );
      }
    }

    if (!response.ok) {
      const responseCode =
        isObject(
          responseBody,
        ) &&
        isNonEmptyString(
          responseBody.code,
        )
          ? responseBody.code
          : undefined;

      throw new DatasetApiError(
        getApiErrorMessage(
          responseBody,
          response.status,
        ),
        {
          status:
            response.status,
          code:
            responseCode,
        },
      );
    }

    if (
      !contentType.includes(
        "application/json",
      )
    ) {
      throw new DatasetApiError(
        "The dataset API returned an unsupported response format.",
        {
          status:
            response.status,
          code:
            "INVALID_CONTENT_TYPE",
        },
      );
    }

    if (
      !isDatasetMetadataResponse(
        responseBody,
      )
    ) {
      if (import.meta.env.DEV) {
        console.error(
          "[DatasetPage] Invalid dataset metadata response:",
          responseBody,
        );
      }

      throw new DatasetApiError(
        "The dataset API returned invalid dataset metadata.",
        {
          status:
            response.status,
          code:
            "INVALID_RESPONSE",
        },
      );
    }

    return responseBody;
  } catch (error) {
    if (timedOut) {
      throw new DatasetApiTimeoutError();
    }

    if (isAbortError(error)) {
      throw error;
    }

    if (
      error instanceof
      DatasetApiError
    ) {
      throw error;
    }

    if (
      error instanceof TypeError
    ) {
      throw new DatasetApiError(
        "Unable to connect to the FastAPI machine-learning backend. Check the API URL, server availability and CORS configuration.",
        {
          code:
            "NETWORK_ERROR",
        },
      );
    }

    throw new DatasetApiError(
      "An unexpected error occurred while loading dataset metadata.",
      {
        code:
          "UNKNOWN_ERROR",
      },
    );
  } finally {
    window.clearTimeout(
      timeoutId,
    );

    signal.removeEventListener(
      "abort",
      abortHandler,
    );
  }
}

/* ============================================================================
   LOADING COMPONENT
============================================================================ */

function DatasetPageSkeleton() {
  return (
    <div
      className="space-y-8"
      aria-busy="true"
      aria-label="Loading dataset information"
    >
      <section>
        <div className="animate-pulse rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
          <div className="space-y-5">
            <div className="h-6 w-32 rounded-full bg-slate-200" />

            <div className="h-10 w-80 max-w-full rounded-lg bg-slate-200" />

            <div className="h-5 w-full max-w-3xl rounded bg-slate-100" />

            <div className="flex flex-wrap gap-3">
              <div className="h-9 w-36 rounded-full bg-slate-100" />
              <div className="h-9 w-36 rounded-full bg-slate-100" />
              <div className="h-9 w-36 rounded-full bg-slate-100" />
            </div>
          </div>
        </div>
      </section>

      <section>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({
            length: 4,
          }).map(
            (_, index) => (
              <div
                key={`statistic-skeleton-${index}`}
                className="animate-pulse rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"
              >
                <div className="h-4 w-24 rounded bg-slate-200" />

                <div className="mt-4 h-8 w-32 rounded bg-slate-200" />

                <div className="mt-4 h-10 w-full rounded bg-slate-100" />
              </div>
            ),
          )}
        </div>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
        <div className="animate-pulse space-y-5">
          <div className="h-6 w-48 rounded bg-slate-200" />

          <div className="h-4 w-72 rounded bg-slate-100" />

          <div className="h-40 rounded bg-slate-100" />
        </div>
      </section>
    </div>
  );
}

/* ============================================================================
   ERROR COMPONENT
============================================================================ */

interface DatasetErrorStateProps {
  message: string;
  refreshing: boolean;
  onRetry: () => void;
}

function DatasetErrorState({
  message,
  refreshing,
  onRetry,
}: DatasetErrorStateProps) {
  return (
    <div className="flex min-h-[500px] items-center justify-center px-4">
      <div
        role="alert"
        className="w-full max-w-lg rounded-2xl border border-red-200 bg-white p-8 text-center shadow-sm"
      >
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-red-50 text-red-600">
          <AlertCircle
            size={28}
            aria-hidden="true"
          />
        </div>

        <h1 className="mt-5 text-xl font-bold text-slate-950">
          Unable to load dataset
        </h1>

        <p className="mt-2 text-sm leading-6 text-slate-500">
          {message}
        </p>

        <button
          type="button"
          onClick={onRetry}
          disabled={refreshing}
          className="mt-6 inline-flex items-center gap-2 rounded-xl bg-slate-950 px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCw
            size={16}
            className={
              refreshing
                ? "animate-spin"
                : undefined
            }
            aria-hidden="true"
          />

          {refreshing
            ? "Retrying..."
            : "Try again"}
        </button>
      </div>
    </div>
  );
}

/* ============================================================================
   REFRESH ERROR
============================================================================ */

interface RefreshErrorProps {
  message: string;
  onRetry: () => void;
  refreshing: boolean;
}

function RefreshError({
  message,
  onRetry,
  refreshing,
}: RefreshErrorProps) {
  return (
    <div
      role="alert"
      className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3"
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <AlertCircle
            size={18}
            className="mt-0.5 shrink-0 text-amber-600"
            aria-hidden="true"
          />

          <p className="text-sm text-amber-800">
            {message}
          </p>
        </div>

        <button
          type="button"
          onClick={onRetry}
          disabled={refreshing}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-amber-300 bg-white px-3 py-2 text-xs font-semibold text-amber-800 transition hover:bg-amber-100 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
        >
          <RefreshCw
            size={14}
            className={
              refreshing
                ? "animate-spin"
                : undefined
            }
            aria-hidden="true"
          />

          Retry
        </button>
      </div>
    </div>
  );
}

/* ============================================================================
   MAIN COMPONENT
============================================================================ */

export default function DatasetPage() {
  const [
    dataset,
    setDataset,
  ] =
    useState<DatasetMetadataResponse | null>(
      null,
    );

  const [
    loading,
    setLoading,
  ] = useState(true);

  const [
    refreshing,
    setRefreshing,
  ] = useState(false);

  const [
    error,
    setError,
  ] = useState<string | null>(
    null,
  );

  const activeRequestRef =
    useRef<AbortController | null>(
      null,
    );

  /* ==========================================================================
     LOAD DATASET
  ========================================================================== */

  const loadDataset =
    useCallback(
      async (
        options?: {
          refresh?: boolean;
        },
      ): Promise<void> => {
        const refresh =
          options?.refresh ??
          false;

        /*
         * Cancel the previous request.
         */
        activeRequestRef.current?.abort();

        const controller =
          new AbortController();

        activeRequestRef.current =
          controller;

        if (refresh) {
          setRefreshing(true);
        } else {
          setLoading(true);
        }

        setError(null);

        try {
          const data =
            await fetchDatasetMetadata(
              controller.signal,
            );

          /*
           * Ignore stale requests.
           */
          if (
            controller.signal.aborted
          ) {
            return;
          }

          if (
            activeRequestRef.current !==
            controller
          ) {
            return;
          }

          setDataset(data);
        } catch (err) {
          /*
           * Aborted requests are expected
           * during refresh/unmount.
           */
          if (
            isAbortError(err) ||
            controller.signal.aborted
          ) {
            return;
          }

          if (
            activeRequestRef.current !==
            controller
          ) {
            return;
          }

          const message =
            err instanceof Error
              ? err.message
              : "Unable to load dataset information from the AI API.";

          if (import.meta.env.DEV) {
            console.error(
              "[DatasetPage] Dataset request failed:",
              err,
            );
          }

          setError(message);
        } finally {
          if (
            activeRequestRef.current ===
            controller
          ) {
            activeRequestRef.current =
              null;

            setLoading(false);
            setRefreshing(false);
          }
        }
      },
      [],
    );

  /* ==========================================================================
     INITIAL LOAD
  ========================================================================== */

  useEffect(() => {
    void loadDataset();

    return () => {
      activeRequestRef.current?.abort();

      activeRequestRef.current =
        null;
    };
  }, [loadDataset]);

  /* ==========================================================================
     STATISTICS
  ========================================================================== */

  const statistics =
    useMemo<
      DatasetStatistic[]
    >(() => {
      if (!dataset) {
        return [];
      }

      return dataset.statistics.map(
        (statistic) => ({
          label:
            statistic.label,
          value:
            statistic.value,
          description:
            statistic.description,
          icon:
            getStatisticIcon(
              statistic.type,
            ),
        }),
      );
    }, [dataset]);

  /* ==========================================================================
     VALIDATION PERCENTAGE
  ========================================================================== */

  const validationPercentage =
    useMemo(() => {
      if (!dataset) {
        return 0;
      }

      return Math.min(
        Math.max(
          dataset.validationPercentage,
          0,
        ),
        100,
      );
    }, [dataset]);

  /* ==========================================================================
     LAST UPDATED
  ========================================================================== */

  const formattedLastUpdated =
    useMemo(
      () =>
        formatDate(
          dataset?.lastUpdated,
        ),
      [dataset?.lastUpdated],
    );

  /* ==========================================================================
     REFRESH
  ========================================================================== */

  const handleRefresh =
    useCallback(() => {
      void loadDataset({
        refresh: true,
      });
    }, [loadDataset]);

  /* ==========================================================================
     INITIAL LOADING
  ========================================================================== */

  if (
    loading &&
    !dataset
  ) {
    return (
      <DatasetPageSkeleton />
    );
  }

  /* ==========================================================================
     INITIAL ERROR
  ========================================================================== */

  if (
    error &&
    !dataset
  ) {
    return (
      <DatasetErrorState
        message={error}
        refreshing={
          refreshing
        }
        onRetry={
          handleRefresh
        }
      />
    );
  }

  if (!dataset) {
    return (
      <DatasetErrorState
        message="Dataset information is currently unavailable."
        refreshing={
          refreshing
        }
        onRetry={
          handleRefresh
        }
      />
    );
  }

  /* ==========================================================================
     SAFE VALUES
  ========================================================================== */

  const hasFeatures =
    dataset.features.length >
    0;

  const hasStatistics =
    statistics.length >
    0;

  /* ==========================================================================
     PAGE
  ========================================================================== */

  return (
    <div className="space-y-8 pb-10">
      {/* ======================================================================
          ACCESSIBLE REFRESH STATUS
      ====================================================================== */}

      <div
        className="sr-only"
        aria-live="polite"
        aria-atomic="true"
      >
        {refreshing
          ? "Refreshing dataset information."
          : error
            ? `Dataset refresh failed. ${error}`
            : "Dataset information loaded."}
      </div>

      {/* ======================================================================
          REFRESH ERROR
      ====================================================================== */}

      {error && (
        <RefreshError
          message={error}
          onRetry={
            handleRefresh
          }
          refreshing={
            refreshing
          }
        />
      )}

      {/* ======================================================================
          HEADER
      ====================================================================== */}

      <header className="relative overflow-hidden rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-8">
        <div
          className="absolute -right-20 -top-20 h-48 w-48 rounded-full bg-emerald-100/60 blur-3xl"
          aria-hidden="true"
        />

        <div
          className="absolute -bottom-24 left-1/3 h-48 w-48 rounded-full bg-sky-100/40 blur-3xl"
          aria-hidden="true"
        />

        <div className="relative">
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0">
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1.5 text-xs font-bold uppercase tracking-wider text-emerald-700">
                <Database
                  size={14}
                  aria-hidden="true"
                />

                Dataset
              </div>

              <h1 className="mt-4 break-words text-3xl font-bold tracking-tight text-slate-950 sm:text-4xl">
                {dataset.title}
              </h1>

              <p className="mt-3 max-w-3xl text-sm leading-7 text-slate-500 sm:text-base">
                {dataset.description}
              </p>
            </div>

            <button
              type="button"
              onClick={
                handleRefresh
              }
              disabled={
                refreshing
              }
              aria-label={
                refreshing
                  ? "Refreshing dataset information"
                  : "Refresh dataset information"
              }
              className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-sm font-semibold text-slate-700 shadow-sm transition hover:border-emerald-200 hover:bg-emerald-50 hover:text-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60"
            >
              <RefreshCw
                size={16}
                className={
                  refreshing
                    ? "animate-spin"
                    : undefined
                }
                aria-hidden="true"
              />

              {refreshing
                ? "Refreshing..."
                : "Refresh"}
            </button>
          </div>

          <div className="mt-6 flex flex-wrap items-center gap-3">
            <div className="inline-flex items-center gap-2 rounded-full bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600">
              <CheckCircle2
                size={15}
                className="text-emerald-600"
                aria-hidden="true"
              />

              {formatNumber(
                dataset.validRecordCount,
              )}{" "}
              valid records
            </div>

            <div className="inline-flex items-center gap-2 rounded-full bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600">
              <BarChart3
                size={15}
                className="text-emerald-600"
                aria-hidden="true"
              />

              {formatNumber(
                dataset.featureCount,
              )}{" "}
              input{" "}
              {dataset.featureCount ===
              1
                ? "feature"
                : "features"}
            </div>

            <div className="inline-flex items-center gap-2 rounded-full bg-slate-50 px-3 py-2 text-xs font-medium text-slate-600">
              <Target
                size={15}
                className="text-emerald-600"
                aria-hidden="true"
              />

              {formatNumber(
                dataset.targetCount,
              )}{" "}
              prediction{" "}
              {dataset.targetCount ===
              1
                ? "target"
                : "targets"}
            </div>
          </div>
        </div>
      </header>

      {/* ======================================================================
          DATASET OVERVIEW
      ====================================================================== */}

      <section
        aria-labelledby="dataset-overview-heading"
      >
        <div className="mb-4">
          <h2
            id="dataset-overview-heading"
            className="text-lg font-bold text-slate-950"
          >
            Dataset Overview
          </h2>

          <p className="mt-1 text-sm text-slate-500">
            Summary of the data available to the machine-learning pipeline.
          </p>
        </div>

        {!hasStatistics ? (
          <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
            <Database
              size={32}
              className="mx-auto text-slate-300"
              aria-hidden="true"
            />

            <p className="mt-3 text-sm font-medium text-slate-600">
              No dataset statistics are currently available.
            </p>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {statistics.map(
              (
                statistic,
                index,
              ) => {
                const Icon =
                  statistic.icon;

                return (
                  <article
                    key={`${statistic.label}-${index}`}
                    className="group rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition duration-200 hover:-translate-y-0.5 hover:border-emerald-200 hover:shadow-md"
                  >
                    <div className="flex items-start justify-between gap-4">
                      <div className="min-w-0">
                        <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                          {
                            statistic.label
                          }
                        </p>

                        <p className="mt-3 break-words text-2xl font-bold tracking-tight text-slate-950">
                          {
                            statistic.value
                          }
                        </p>
                      </div>

                      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600 transition group-hover:bg-emerald-100">
                        <Icon
                          size={20}
                          aria-hidden="true"
                        />
                      </div>
                    </div>

                    <p className="mt-3 text-xs leading-5 text-slate-500">
                      {
                        statistic.description
                      }
                    </p>
                  </article>
                );
              },
            )}
          </div>
        )}
      </section>

      {/* ======================================================================
          DATASET INFORMATION
      ====================================================================== */}

      <section
        aria-label="Dataset information"
        className="grid gap-6 md:grid-cols-3"
      >
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-50 text-blue-600">
              <Rows3
                size={20}
                aria-hidden="true"
              />
            </div>

            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Records
              </p>

              <p className="mt-1 text-xl font-bold text-slate-950">
                {formatNumber(
                  dataset.recordCount,
                )}
              </p>
            </div>
          </div>
        </article>

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-50 text-violet-600">
              <Database
                size={20}
                aria-hidden="true"
              />
            </div>

            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Columns
              </p>

              <p className="mt-1 text-xl font-bold text-slate-950">
                {formatNumber(
                  dataset.columnCount,
                )}
              </p>
            </div>
          </div>
        </article>

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-50 text-amber-600">
              <Info
                size={20}
                aria-hidden="true"
              />
            </div>

            <div>
              <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
                Missing Values
              </p>

              <p className="mt-1 text-xl font-bold text-slate-950">
                {formatNumber(
                  dataset.missingValues,
                )}
              </p>
            </div>
          </div>
        </article>
      </section>

      {/* ======================================================================
          DATASET FEATURES
      ====================================================================== */}

      <section
        aria-labelledby="dataset-features-heading"
        className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm"
      >
        <div className="border-b border-slate-200 p-6 sm:p-7">
          <div className="flex items-start gap-4">
            <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
              <Database
                size={21}
                aria-hidden="true"
              />
            </div>

            <div className="min-w-0">
              <h2
                id="dataset-features-heading"
                className="font-bold text-slate-950"
              >
                Dataset Features
              </h2>

              <p className="mt-1 text-sm leading-6 text-slate-500">
                Variables used by{" "}
                <strong className="font-semibold text-slate-700">
                  {
                    dataset.model.name
                  }
                </strong>
                .
              </p>
            </div>
          </div>
        </div>

        {!hasFeatures ? (
          <div className="p-10 text-center">
            <Database
              size={32}
              className="mx-auto text-slate-300"
              aria-hidden="true"
            />

            <p className="mt-3 text-sm font-medium text-slate-600">
              No dataset features are currently available.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-[760px] w-full text-left text-sm">
              <caption className="sr-only">
                {
                  dataset.datasetName
                }{" "}
                dataset features
              </caption>

              <thead className="border-b border-slate-200 bg-slate-50">
                <tr>
                  <th
                    scope="col"
                    className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500"
                  >
                    Column
                  </th>

                  <th
                    scope="col"
                    className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500"
                  >
                    Role
                  </th>

                  <th
                    scope="col"
                    className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500"
                  >
                    Unit
                  </th>

                  <th
                    scope="col"
                    className="px-6 py-4 text-xs font-bold uppercase tracking-wider text-slate-500"
                  >
                    Description
                  </th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-100">
                {dataset.features.map(
                  (feature) => (
                    <tr
                      key={`${feature.role}-${feature.column}`}
                      className="transition hover:bg-slate-50/80"
                    >
                      <td className="px-6 py-5">
                        <code className="rounded-lg bg-slate-100 px-2.5 py-1.5 text-xs font-semibold text-slate-800">
                          {
                            feature.column
                          }
                        </code>
                      </td>

                      <td className="px-6 py-5">
                        <span
                          className={
                            feature.role ===
                            "Target"
                              ? "inline-flex items-center rounded-full bg-blue-50 px-2.5 py-1 text-xs font-bold text-blue-700"
                              : "inline-flex items-center rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-bold text-emerald-700"
                          }
                        >
                          {
                            feature.role
                          }
                        </span>
                      </td>

                      <td className="px-6 py-5 font-medium text-slate-700">
                        {feature.unit ||
                          "—"}
                      </td>

                      <td className="max-w-md px-6 py-5 leading-6 text-slate-500">
                        {
                          feature.description
                        }
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        )}

        <div className="border-t border-slate-200 bg-slate-50/70 px-6 py-4">
          <div className="flex items-start gap-3">
            <Info
              size={17}
              className="mt-0.5 shrink-0 text-slate-400"
              aria-hidden="true"
            />

            <p className="text-xs leading-5 text-slate-500">
              The dataset contains{" "}
              <strong className="font-semibold text-slate-700">
                {formatNumber(
                  dataset.validRecordCount,
                )}
              </strong>{" "}
              valid records across{" "}
              <strong className="font-semibold text-slate-700">
                {formatNumber(
                  dataset.featureCount,
                )}
              </strong>{" "}
              input{" "}
              {dataset.featureCount ===
              1
                ? "feature"
                : "features"}{" "}
              and{" "}
              <strong className="font-semibold text-slate-700">
                {formatNumber(
                  dataset.targetCount,
                )}
              </strong>{" "}
              target{" "}
              {dataset.targetCount ===
              1
                ? "variable."
                : "variables."}
            </p>
          </div>
        </div>
      </section>

      {/* ======================================================================
          MODEL INFORMATION
      ====================================================================== */}

      <section
        aria-labelledby="model-information-heading"
        className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      >
        <div className="flex items-start gap-4">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-slate-950 text-emerald-300">
            <BarChart3
              size={21}
              aria-hidden="true"
            />
          </div>

          <div className="min-w-0">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Machine Learning Model
            </p>

            <h2
              id="model-information-heading"
              className="mt-1 text-xl font-bold text-slate-950"
            >
              {
                dataset.model.name
              }
            </h2>

            <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-500">
              {
                dataset.model
                  .description
              }
            </p>
          </div>
        </div>

        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Input Features
            </p>

            <div className="mt-3 flex flex-wrap gap-2">
              {dataset.model.features.map(
                (feature) => (
                  <code
                    key={feature}
                    className="rounded-lg bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 shadow-sm ring-1 ring-slate-200"
                  >
                    {feature}
                  </code>
                ),
              )}
            </div>
          </div>

          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Prediction Target
            </p>

            <code className="mt-3 inline-block rounded-lg bg-white px-2.5 py-1.5 text-xs font-semibold text-slate-700 shadow-sm ring-1 ring-slate-200">
              {
                dataset.model.target
              }
            </code>
          </div>

          <div className="rounded-xl bg-slate-50 p-4">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Target Unit
            </p>

            <p className="mt-3 text-lg font-bold text-slate-950">
              {
                dataset.model
                  .targetUnit
              }
            </p>
          </div>
        </div>
      </section>

      {/* ======================================================================
          DATA QUALITY
      ====================================================================== */}

      <section
        aria-labelledby="data-quality-heading"
        className="grid gap-6 lg:grid-cols-[1fr_1.5fr]"
      >
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
              <CheckCircle2
                size={20}
                aria-hidden="true"
              />
            </div>

            <div>
              <h2
                id="data-quality-heading"
                className="font-bold text-slate-950"
              >
                Data Availability
              </h2>

              <p className="text-sm text-slate-500">
                Dataset readiness
              </p>
            </div>
          </div>

          <div className="mt-6">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-slate-600">
                Valid records
              </span>

              <span className="text-sm font-bold text-emerald-700">
                {formatPercentage(
                  validationPercentage,
                )}
              </span>
            </div>

            <div
              className="mt-3 h-2 overflow-hidden rounded-full bg-slate-100"
              role="progressbar"
              aria-label="Dataset validation percentage"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={
                validationPercentage
              }
            >
              <div
                className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                style={{
                  width:
                    `${validationPercentage}%`,
                }}
              />
            </div>

            <p className="mt-3 text-xs leading-5 text-slate-500">
              {formatNumber(
                dataset.validRecordCount,
              )}{" "}
              of{" "}
              {formatNumber(
                dataset.recordCount,
              )}{" "}
              total records passed dataset validation.
            </p>

            {dataset.invalidRecordCount >
              0 && (
              <p className="mt-2 text-xs font-medium text-amber-600">
                {formatNumber(
                  dataset.invalidRecordCount,
                )}{" "}
                records did not pass validation.
              </p>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-gradient-to-br from-slate-950 to-slate-900 p-6 text-white shadow-sm">
          <div className="flex items-start gap-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/10 text-emerald-300">
              <BarChart3
                size={20}
                aria-hidden="true"
              />
            </div>

            <div className="min-w-0">
              <h2 className="font-bold">
                How the dataset is used
              </h2>

              <p className="mt-2 text-sm leading-6 text-slate-300">
                {
                  dataset.model
                    .description
                }
              </p>
            </div>
          </div>

          <div className="mt-6 grid gap-3 sm:grid-cols-3">
            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Inputs
              </p>

              <p className="mt-2 text-lg font-bold">
                {formatNumber(
                  dataset.featureCount,
                )}
              </p>

              <p className="mt-1 text-xs text-slate-400">
                Model{" "}
                {dataset.featureCount ===
                1
                  ? "feature"
                  : "features"}
              </p>
            </div>

            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Target
              </p>

              <p className="mt-2 text-lg font-bold">
                {formatNumber(
                  dataset.targetCount,
                )}
              </p>

              <p className="mt-1 text-xs text-slate-400">
                Prediction{" "}
                {dataset.targetCount ===
                1
                  ? "target"
                  : "targets"}
              </p>
            </div>

            <div className="rounded-xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Records
              </p>

              <p className="mt-2 text-lg font-bold">
                {formatNumber(
                  dataset.validRecordCount,
                )}
              </p>

              <p className="mt-1 text-xs text-slate-400">
                Valid samples
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* ======================================================================
          FOOTER
      ====================================================================== */}

      <footer className="border-t border-slate-200 pt-6">
        <div className="flex flex-col gap-3 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between">
          <p>
            Dataset:{" "}
            <span className="font-medium text-slate-500">
              {
                dataset.datasetName
              }
            </span>
          </p>

          {formattedLastUpdated && (
            <p>
              Last updated:{" "}
              <time
                dateTime={
                  dataset.lastUpdated ??
                  undefined
                }
              >
                {
                  formattedLastUpdated
                }
              </time>
            </p>
          )}
        </div>
      </footer>
    </div>
  );
}