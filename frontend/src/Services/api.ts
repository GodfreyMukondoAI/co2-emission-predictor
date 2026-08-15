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

const configuredApiUrl =
  import.meta.env.VITE_API_URL?.trim();

if (!configuredApiUrl) {
  throw new Error(
    "VITE_API_URL is not configured. " +
      "Please define VITE_API_URL in the appropriate Vite environment file.",
  );
}

const API_BASE_URL = configuredApiUrl.replace(/\/+$/, "");

/* ==========================================================================
   API ERROR
   ========================================================================== */

class ApiError extends Error {
  public readonly status: number;

  constructor(
    message: string,
    status: number,
  ) {
    super(message);

    this.name = "ApiError";
    this.status = status;

    Object.setPrototypeOf(
      this,
      ApiError.prototype,
    );
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

function isObject(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === "object" &&
    value !== null &&
    !Array.isArray(value)
  );
}

function extractErrorMessage(
  value: unknown,
): string | null {
  if (!isObject(value)) {
    return null;
  }

  const errorData =
    value as ApiErrorResponse;

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

/* ==========================================================================
   GENERIC API REQUEST
   ========================================================================== */

async function request<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const normalizedEndpoint =
    endpoint.startsWith("/")
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
              "Content-Type":
                "application/json",
            }
          : {}),

        ...(options.headers ?? {}),
      },

      credentials: "include",
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

    try {
      const errorData: unknown =
        await response.json();

      const extractedMessage =
        extractErrorMessage(errorData);

      if (extractedMessage) {
        message = extractedMessage;
      }
    } catch {
      // Keep the default HTTP error message.
    }

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
  } catch {
    throw new ApiError(
      "The prediction API returned invalid JSON.",
      response.status,
    );
  }
}

/* ==========================================================================
   API INFORMATION
   ========================================================================== */

export async function getApiInfo(): Promise<ApiWelcomeResponse> {
  return request<ApiWelcomeResponse>("/");
}

/* ==========================================================================
   HEALTH CHECK
   ========================================================================== */

export async function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(
    "/api/health",
  );
}

/* ==========================================================================
   MODEL INFORMATION
   ========================================================================== */

export async function getModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>(
    "/api/model",
  );
}

/* ==========================================================================
   CO₂ PREDICTION
   ========================================================================== */

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