/**
 * ============================================================================
 * CO₂ EMISSION PREDICTOR
 * ============================================================================
 *
 * DatasetPage
 *
 * Production-ready dataset information page.
 *
 * Responsibilities
 * ----------------
 * - Load dataset metadata from FastAPI.
 * - Display real dataset statistics.
 * - Display model information.
 * - Display dataset features.
 * - Display data-quality information.
 * - Support loading, error and refresh states.
 * - Validate API responses before rendering.
 * - Support request cancellation and timeout.
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
  useState,
} from "react";


/* ============================================================================
   TYPES
============================================================================ */

type DatasetFeatureRole =
  | "Feature"
  | "Target";

type DatasetStatisticType =
  | "records"
  | "engine-size"
  | "fuel-consumption"
  | "target";


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
  type?: DatasetStatisticType;
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
    Record<string, number>
  >;

  targetStatistics: {
    column: string;
    min: number;
    max: number;
    mean: number;
    median: number;
    std: number;
  };

  statistics: DatasetStatisticResponse[];

  model: DatasetModel;

  lastUpdated?: string | null;
}


interface ApiErrorResponse {
  detail?: unknown;
  message?: unknown;
  error?: unknown;
}


/* ============================================================================
   CONFIGURATION
============================================================================ */

const API_BASE_URL = (
  import.meta.env.VITE_API_URL ??
  "http://127.0.0.1:8000"
).replace(/\/+$/, "");


const DATASET_METADATA_PATH = (
  import.meta.env.VITE_DATASET_METADATA_ENDPOINT ??
  "/dataset/metadata"
).replace(/^\/+/, "");


const DATASET_METADATA_ENDPOINT =
  `${API_BASE_URL}/${DATASET_METADATA_PATH}`;


const REQUEST_TIMEOUT_MS = 15_000;


const DEFAULT_ERROR_MESSAGE =
  "Unable to load dataset information from the AI API.";


const VALID_STATISTIC_TYPES:
  readonly DatasetStatisticType[] = [
    "records",
    "engine-size",
    "fuel-consumption",
    "target",
  ];


/* ============================================================================
   TYPE GUARDS
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


function isStringArray(
  value: unknown,
): value is string[] {
  return (
    Array.isArray(value) &&
    value.every(
      (item) =>
        typeof item === "string",
    )
  );
}


function isDatasetFeature(
  value: unknown,
): value is DatasetFeature {
  if (!isObject(value)) {
    return false;
  }

  return (
    typeof value.column === "string" &&
    value.column.trim().length > 0 &&
    (
      value.role === "Feature" ||
      value.role === "Target"
    ) &&
    typeof value.unit === "string" &&
    typeof value.description === "string"
  );
}


function isDatasetStatisticResponse(
  value: unknown,
): value is DatasetStatisticResponse {
  if (!isObject(value)) {
    return false;
  }

  if (
    typeof value.label !== "string" ||
    typeof value.value !== "string" ||
    typeof value.description !== "string"
  ) {
    return false;
  }

  if (value.type === undefined) {
    return true;
  }

  return (
    typeof value.type === "string" &&
    VALID_STATISTIC_TYPES.includes(
      value.type as DatasetStatisticType,
    )
  );
}


function isDatasetModel(
  value: unknown,
): value is DatasetModel {
  if (!isObject(value)) {
    return false;
  }

  return (
    typeof value.name === "string" &&
    value.name.trim().length > 0 &&
    typeof value.description === "string" &&
    value.description.trim().length > 0 &&
    isStringArray(value.features) &&
    value.features.length > 0 &&
    typeof value.target === "string" &&
    value.target.trim().length > 0 &&
    typeof value.targetUnit === "string" &&
    value.targetUnit.trim().length > 0
  );
}


function isStatisticsObject(
  value: unknown,
): value is Record<
  string,
  Record<string, number>
> {
  if (!isObject(value)) {
    return false;
  }

  return Object.values(value).every(
    (statistics) => {
      if (!isObject(statistics)) {
        return false;
      }

      return Object.values(
        statistics,
      ).every(
        (numberValue) =>
          isFiniteNumber(numberValue),
      );
    },
  );
}


function isDatasetMetadataResponse(
  value: unknown,
): value is DatasetMetadataResponse {
  if (!isObject(value)) {
    return false;
  }

  if (
    typeof value.datasetName !== "string" ||
    typeof value.title !== "string" ||
    typeof value.description !== "string"
  ) {
    return false;
  }

  const numericFields = [
    "recordCount",
    "columnCount",
    "missingValues",
    "validRecordCount",
    "invalidRecordCount",
    "featureCount",
    "targetCount",
    "validationPercentage",
  ];

  for (const field of numericFields) {
    if (!isFiniteNumber(value[field])) {
      return false;
    }
  }

  if (
    !Array.isArray(value.columnNames) ||
    !value.columnNames.every(
      (column) =>
        typeof column === "string",
    )
  ) {
    return false;
  }

  if (
    !Array.isArray(value.features) ||
    !value.features.every(
      isDatasetFeature,
    )
  ) {
    return false;
  }

  if (
    !Array.isArray(value.statistics) ||
    !value.statistics.every(
      isDatasetStatisticResponse,
    )
  ) {
    return false;
  }

  if (
    !isStatisticsObject(
      value.featureStatistics,
    )
  ) {
    return false;
  }

  if (
    !isObject(value.targetStatistics)
  ) {
    return false;
  }

  if (
    typeof value.targetStatistics.column !==
    "string"
  ) {
    return false;
  }

  for (
    const field of [
      "min",
      "max",
      "mean",
      "median",
      "std",
    ]
  ) {
    if (
      !isFiniteNumber(
        value.targetStatistics[field],
      )
    ) {
      return false;
    }
  }

  if (!isDatasetModel(value.model)) {
    return false;
  }

  if (
    value.lastUpdated !== undefined &&
    value.lastUpdated !== null &&
    typeof value.lastUpdated !== "string"
  ) {
    return false;
  }

  return true;
}


/* ============================================================================
   HELPERS
============================================================================ */

function getStatisticIcon(
  type?: DatasetStatisticType,
): LucideIcon {
  switch (type) {
    case "records":
      return Rows3;

    case "engine-size":
      return Gauge;

    case "fuel-consumption":
      return Fuel;

    case "target":
      return Target;

    default:
      return Database;
  }
}


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
    return "0.0%";
  }

  const safeValue = Math.min(
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

  return new Intl.DateTimeFormat(
    undefined,
    {
      dateStyle: "medium",
      timeStyle: "short",
    },
  ).format(date);
}


function getApiErrorMessage(
  body: unknown,
  responseStatus: number,
): string {
  if (isObject(body)) {
    const errorBody =
      body as ApiErrorResponse;

    if (
      typeof errorBody.detail ===
        "string" &&
      errorBody.detail.trim()
        .length > 0
    ) {
      return errorBody.detail;
    }

    if (
      typeof errorBody.message ===
        "string" &&
      errorBody.message.trim()
        .length > 0
    ) {
      return errorBody.message;
    }

    if (
      typeof errorBody.error ===
        "string" &&
      errorBody.error.trim()
        .length > 0
    ) {
      return errorBody.error;
    }
  }

  switch (responseStatus) {
    case 400:
      return (
        "The dataset metadata request was invalid."
      );

    case 401:
      return (
        "Authentication is required to access dataset metadata."
      );

    case 403:
      return (
        "You are not authorized to access dataset metadata."
      );

    case 404:
      return (
        "The dataset metadata endpoint was not found on the AI API."
      );

    case 500:
      return (
        "The AI API encountered an internal server error."
      );

    case 502:
    case 503:
    case 504:
      return (
        "The AI API is currently unavailable."
      );

    default:
      return (
        `Dataset API request failed with HTTP ${responseStatus}.`
      );
  }
}


/* ============================================================================
   API CLIENT
============================================================================ */

async function fetchDatasetMetadata(
  signal?: AbortSignal,
): Promise<DatasetMetadataResponse> {
  const controller =
    new AbortController();

  const timeoutId =
    window.setTimeout(
      () => {
        controller.abort();
      },
      REQUEST_TIMEOUT_MS,
    );

  const abortHandler =
    () => {
      controller.abort();
    };

  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener(
        "abort",
        abortHandler,
        {
          once: true,
        },
      );
    }
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

          credentials: "include",

          cache: "no-store",

          signal:
            controller.signal,
        },
      );

    let responseBody:
      unknown = null;

    const contentType =
      response.headers.get(
        "content-type",
      ) ?? "";

    if (
      contentType
        .toLowerCase()
        .includes(
          "application/json",
        )
    ) {
      try {
        responseBody =
          await response.json();
      } catch {
        responseBody = null;
      }
    }

    if (!response.ok) {
      throw new Error(
        getApiErrorMessage(
          responseBody,
          response.status,
        ),
      );
    }

    if (
      !isDatasetMetadataResponse(
        responseBody,
      )
    ) {
      console.error(
        "Invalid dataset metadata response:",
        responseBody,
      );

      throw new Error(
        "The dataset API returned an invalid metadata structure.",
      );
    }

    return responseBody;

  } catch (error) {
    if (
      error instanceof DOMException &&
      error.name === "AbortError"
    ) {
      throw error;
    }

    if (
      error instanceof TypeError
    ) {
      throw new Error(
        "Unable to connect to the FastAPI machine-learning backend. Check that the API is running and CORS is configured correctly.",
      );
    }

    throw error;

  } finally {
    window.clearTimeout(
      timeoutId,
    );

    if (signal) {
      signal.removeEventListener(
        "abort",
        abortHandler,
      );
    }
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
   MAIN COMPONENT
============================================================================ */

export default function DatasetPage() {
  const [
    dataset,
    setDataset,
  ] = useState<
    DatasetMetadataResponse | null
  >(null);

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
  ] = useState<string | null>(null);


  /* --------------------------------------------------------------------------
     LOAD DATASET
  -------------------------------------------------------------------------- */

  const loadDataset =
    useCallback(
      async (
        options?: {
          refresh?: boolean;
          signal?: AbortSignal;
        },
      ): Promise<void> => {
        const refresh =
          options?.refresh ?? false;

        const signal =
          options?.signal;

        try {
          if (refresh) {
            setRefreshing(true);
          } else {
            setLoading(true);
          }

          setError(null);

          const data =
            await fetchDatasetMetadata(
              signal,
            );

          if (
            !signal?.aborted
          ) {
            setDataset(data);
          }

        } catch (err) {
          if (
            err instanceof DOMException &&
            err.name === "AbortError"
          ) {
            return;
          }

          if (
            signal?.aborted
          ) {
            return;
          }

          console.error(
            "Failed to load dataset metadata:",
            err,
          );

          setError(
            err instanceof Error
              ? err.message
              : DEFAULT_ERROR_MESSAGE,
          );

        } finally {
          if (
            !signal?.aborted
          ) {
            setLoading(false);
            setRefreshing(false);
          }
        }
      },
      [],
    );


  /* --------------------------------------------------------------------------
     INITIAL LOAD
  -------------------------------------------------------------------------- */

  useEffect(() => {
    const controller =
      new AbortController();

    void loadDataset({
      signal:
        controller.signal,
    });

    return () => {
      controller.abort();
    };
  }, [loadDataset]);


  /* --------------------------------------------------------------------------
     STATISTICS
  -------------------------------------------------------------------------- */

  const statistics =
    useMemo<DatasetStatistic[]>(
      () => {
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
      },
      [dataset],
    );


  /* --------------------------------------------------------------------------
     VALIDATION
  -------------------------------------------------------------------------- */

  const validationPercentage =
    useMemo(
      () => {
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
      },
      [dataset],
    );


  /* --------------------------------------------------------------------------
     LAST UPDATED
  -------------------------------------------------------------------------- */

  const formattedLastUpdated =
    useMemo(
      () =>
        formatDate(
          dataset?.lastUpdated,
        ),
      [dataset?.lastUpdated],
    );


  /* --------------------------------------------------------------------------
     REFRESH
  -------------------------------------------------------------------------- */

  const handleRefresh =
    useCallback(
      () => {
        void loadDataset({
          refresh: true,
        });
      },
      [loadDataset],
    );


  /* --------------------------------------------------------------------------
     LOADING
  -------------------------------------------------------------------------- */

  if (loading) {
    return (
      <DatasetPageSkeleton />
    );
  }


  /* --------------------------------------------------------------------------
     ERROR
  -------------------------------------------------------------------------- */

  if (
    error ||
    !dataset
  ) {
    return (
      <DatasetErrorState
        message={
          error ??
          DEFAULT_ERROR_MESSAGE
        }
        refreshing={
          refreshing
        }
        onRetry={
          handleRefresh
        }
      />
    );
  }


  /* --------------------------------------------------------------------------
     SAFE VALUES
  -------------------------------------------------------------------------- */

  const hasFeatures =
    dataset.features.length > 0;

  const hasStatistics =
    statistics.length > 0;


  /* ==========================================================================
     PAGE
  ========================================================================== */

  return (
    <div className="space-y-8 pb-10">

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
              onClick={handleRefresh}
              disabled={refreshing}
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
              {dataset.featureCount === 1
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
              {dataset.targetCount === 1
                ? "target"
                : "targets"}

            </div>

          </div>

        </div>

      </header>


      {/* ======================================================================
          OVERVIEW
      ====================================================================== */}

      <section aria-labelledby="dataset-overview-heading">

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
              (statistic, index) => {

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
                          {statistic.label}
                        </p>

                        <p className="mt-3 break-words text-2xl font-bold tracking-tight text-slate-950">
                          {statistic.value}
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
                      {statistic.description}
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

      <section className="grid gap-6 md:grid-cols-3">

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
          FEATURES
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
                  {dataset.model.name}
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
                {dataset.datasetName} dataset features
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
                          {feature.column}
                        </code>

                      </td>


                      <td className="px-6 py-5">

                        <span
                          className={
                            feature.role === "Target"
                              ? "inline-flex items-center rounded-full bg-blue-50 px-2.5 py-1 text-xs font-bold text-blue-700"
                              : "inline-flex items-center rounded-full bg-emerald-50 px-2.5 py-1 text-xs font-bold text-emerald-700"
                          }
                        >
                          {feature.role}
                        </span>

                      </td>


                      <td className="px-6 py-5 font-medium text-slate-700">
                        {feature.unit || "—"}
                      </td>


                      <td className="max-w-md px-6 py-5 leading-6 text-slate-500">
                        {feature.description ||
                          "No description available."}
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

              {dataset.featureCount === 1
                ? "feature"
                : "features"}{" "}

              and{" "}

              <strong className="font-semibold text-slate-700">
                {formatNumber(
                  dataset.targetCount,
                )}
              </strong>{" "}

              target{" "}

              {dataset.targetCount === 1
                ? "variable."
                : "variables."}

            </p>

          </div>

        </div>

      </section>


      {/* ======================================================================
          MODEL INFORMATION
      ====================================================================== */}

      <section className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">

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

            <h2 className="mt-1 text-xl font-bold text-slate-950">
              {dataset.model.name}
            </h2>

            <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-500">
              {dataset.model.description}
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
              {dataset.model.target}
            </code>

          </div>


          <div className="rounded-xl bg-slate-50 p-4">

            <p className="text-xs font-bold uppercase tracking-wider text-slate-400">
              Target Unit
            </p>

            <p className="mt-3 text-lg font-bold text-slate-950">
              {dataset.model.targetUnit}
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


            {dataset.invalidRecordCount > 0 && (

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
                {dataset.model.description}
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
                {dataset.featureCount === 1
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
                {dataset.targetCount === 1
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
              {dataset.datasetName}
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
                {formattedLastUpdated}
              </time>

            </p>

          )}

        </div>

      </footer>

    </div>
  );
}