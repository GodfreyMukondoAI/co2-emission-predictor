import type {
  ApiWelcomeResponse,
  HealthResponse,
  ModelInfoResponse,
  PredictionRequest,
  PredictionResponse,
} from "../types/prediction";

/* ==========================================================================
   ENVIRONMENT CONFIGURATION
   ========================================================================== */

const configuredApiUrl = import.meta.env.VITE_API_URL?.trim();

if (!configuredApiUrl) {
  throw new Error(
    "VITE_API_URL is not configured. " +
      "Please define VITE_API_URL in the appropriate Vite environment file.",
  );
}

/**
 * Normalized API base URL.
 *
 * Example:
 * VITE_API_URL=https://co2-emission-predictor-ys3q.onrender.com
 *
 * Trailing slashes are removed so endpoints can safely be appended.
 */
const API_BASE_URL = configuredApiUrl.replace(/\/+$/, "");

/* ==========================================================================
   API ERROR
   ========================================================================== */

class ApiError extends Error {
  public readonly status: number;

  constructor(message: string, status: number) {
    super(message);

    this.name = "ApiError";
    this.status = status;

    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/* ==========================================================================
   ERROR RESPONSE TYPE
   ========================================================================== */

interface ApiErrorResponse {
  detail?: unknown;
  message?: unknown;
  error?: unknown;
}

/* ==========================================================================
   HELPERS
   ========================================================================== */

/**
 * Determines whether a value is a non-null object
 * that is not an array.
 */
function isObject(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

/**
 * Extracts a useful error message from common
 * FastAPI / application error response formats.
 */
function extractErrorMessage(
  value: unknown,
): string | null {
  if (!isObject(value)) {
    return null;
  }

  const errorData = value as ApiErrorResponse;

  if (
    typeof errorData.detail === "string" &&
    errorData.detail.trim()
  ) {
    return errorData.detail.trim();
  }

  if (
    typeof errorData.message === "string" &&
    errorData.message.trim()
  ) {
    return errorData.message.trim();
  }

  if (
    typeof errorData.error === "string" &&
    errorData.error.trim()
  ) {
    return errorData.error.trim();
  }

  return null;
}

/**
 * Safely converts a response body into a useful error message.
 */
async function extractResponseError(
  response: Response,
): Promise<string | null> {
  try {
    const contentType =
      response.headers.get("content-type") ?? "";

    if (
      !contentType
        .toLowerCase()
        .includes("application/json")
    ) {
      return null;
    }

    const errorData: unknown = await response.json();

    return extractErrorMessage(errorData);
  } catch {
    return null;
  }
}

/* ==========================================================================
   GENERIC API REQUEST
   ========================================================================== */

/**
 * Generic HTTP request helper used by all prediction API operations.
 *
 * IMPORTANT:
 * This API does not use browser cookies/session credentials.
 *
 * Therefore credentials are intentionally NOT included.
 *
 * This prevents the browser from requiring:
 *
 * Access-Control-Allow-Credentials: true
 *
 * from the FastAPI server during cross-origin requests.
 */
async function request<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const normalizedEndpoint = endpoint.startsWith("/")
    ? endpoint
    : `/${endpoint}`;

  const url =
    `${API_BASE_URL}${normalizedEndpoint}`;

  let response: Response;

  try {
    response = await fetch(url, {
      ...options,

      headers: {
        Accept: "application/json",

        ...(options.body
          ? {
              "Content-Type": "application/json",
            }
          : {}),

        ...(options.headers ?? {}),
      },

      /*
       * Do NOT use:
       *
       * credentials: "include"
       *
       * The CO₂ prediction API is a public API and does not
       * require browser cookies or session credentials.
       *
       * Omitting this property uses the browser's default
       * credentials mode ("same-origin"), which does not
       * send credentials to the cross-origin Render API.
       */
    });
  } catch (error) {
    console.error(
      "Prediction API connection error:",
      error,
    );

    throw new ApiError(
      "Unable to connect to the prediction API. " +
        "Please check that the API is available and try again.",
      0,
    );
  }

  /* ------------------------------------------------------------------------
     HTTP ERROR
     ------------------------------------------------------------------------ */

  if (!response.ok) {
    let message =
      `Request failed with status ${response.status}.`;

    const extractedMessage =
      await extractResponseError(response);

    if (extractedMessage) {
      message = extractedMessage;
    }

    console.error(
      "Prediction API HTTP error:",
      {
        url,
        status: response.status,
        statusText: response.statusText,
        message,
      },
    );

    throw new ApiError(
      message,
      response.status,
    );
  }

  /* ------------------------------------------------------------------------
     NO CONTENT
     ------------------------------------------------------------------------ */

  if (response.status === 204) {
    return undefined as T;
  }

  /* ------------------------------------------------------------------------
     RESPONSE CONTENT TYPE
     ------------------------------------------------------------------------ */

  const contentType =
    response.headers.get("content-type") ?? "";

  if (
    !contentType
      .toLowerCase()
      .includes("application/json")
  ) {
    throw new ApiError(
      "The prediction API returned an unexpected response format.",
      response.status,
    );
  }

  /* ------------------------------------------------------------------------
     JSON RESPONSE
     ------------------------------------------------------------------------ */

  try {
    return (await response.json()) as T;
  } catch (error) {
    console.error(
      "Prediction API JSON parsing error:",
      error,
    );

    throw new ApiError(
      "The prediction API returned invalid JSON.",
      response.status,
    );
  }
}

/* ==========================================================================
   API INFORMATION
   ========================================================================== */

/**
 * Returns basic information about the CO₂ Prediction API.
 */
export async function getApiInfo(): Promise<ApiWelcomeResponse> {
  return request<ApiWelcomeResponse>("/");
}

/* ==========================================================================
   HEALTH CHECK
   ========================================================================== */

/**
 * Checks whether the production prediction API is healthy.
 */
export async function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health");
}

/* ==========================================================================
   MODEL INFORMATION
   ========================================================================== */

/**
 * Returns information about the loaded machine-learning model.
 */
export async function getModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>("/api/model");
}

/* ==========================================================================
   CO₂ PREDICTION
   ========================================================================== */

/**
 * Sends vehicle information to the production ML API
 * and returns the predicted CO₂ emission.
 */
export async function predictCO2(
  payload: PredictionRequest,
): Promise<PredictionResponse> {
  return request<PredictionResponse>(
    "/api/predict",
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

/* ==========================================================================
   PUBLIC API
   ========================================================================== */

export { ApiError };