
/**
 * ============================================================
 * CO₂ EMISSION PREDICTOR
 * ============================================================
 *
 * Application Environment Configuration
 *
 * Centralized runtime configuration for the React/Vite
 * frontend.
 *
 * IMPORTANT:
 * Vite exposes only variables prefixed with VITE_ to the
 * browser.
 *
 * Supported environment files:
 *
 *   .env.development
 *   .env.production
 *
 * Example production API:
 *
 *   VITE_API_URL=https://co2-emission-predictor-ys3q.onrender.com
 *
 * ============================================================
 */

/* ============================================================================
   CONSTANTS
============================================================================ */

const DEFAULT_API_TIMEOUT = 30_000;

const DEFAULT_APP_NAME =
  "CO₂ Emission Predictor";

const DEFAULT_APP_VERSION =
  "1.0.0";

const DEFAULT_DATASET_METADATA_ENDPOINT =
  "/dataset/metadata";

/* ============================================================================
   URL NORMALIZATION
============================================================================ */

/**
 * Removes whitespace and trailing slashes from a URL.
 */
function normalizeUrl(
  value: string,
): string {
  return value
    .trim()
    .replace(/\/+$/, "");
}

/**
 * Reads and validates the FastAPI base URL.
 *
 * The URL must:
 *
 * - exist
 * - be valid
 * - use HTTP or HTTPS
 * - contain no query string
 * - contain no hash fragment
 */
function getApiUrl(): string {
  const raw =
    import.meta.env.VITE_API_URL;

  if (
    typeof raw !== "string" ||
    raw.trim().length === 0
  ) {
    throw new Error(
      "VITE_API_URL is not configured. " +
        "Create .env.development or .env.production " +
        "in the frontend project root.",
    );
  }

  const normalized =
    normalizeUrl(raw);

  let parsed: URL;

  try {
    parsed = new URL(normalized);
  } catch {
    throw new Error(
      `VITE_API_URL contains an invalid URL: "${normalized}".`,
    );
  }

  if (
    parsed.protocol !== "http:" &&
    parsed.protocol !== "https:"
  ) {
    throw new Error(
      "VITE_API_URL must use HTTP or HTTPS.",
    );
  }

  if (
    parsed.search ||
    parsed.hash
  ) {
    throw new Error(
      "VITE_API_URL must not contain a query string or hash fragment.",
    );
  }

  return normalized;
}

/* ============================================================================
   API TIMEOUT
============================================================================ */

/**
 * Reads VITE_API_TIMEOUT.
 *
 * Example:
 *
 *   VITE_API_TIMEOUT=30000
 *
 * Invalid values fall back to 30 seconds.
 */
function getApiTimeout(): number {
  const raw =
    import.meta.env.VITE_API_TIMEOUT;

  if (
    typeof raw !== "string" ||
    raw.trim().length === 0
  ) {
    return DEFAULT_API_TIMEOUT;
  }

  const value =
    Number(raw.trim());

  if (
    !Number.isFinite(value) ||
    value <= 0
  ) {
    return DEFAULT_API_TIMEOUT;
  }

  const timeout =
    Math.floor(value);

  return timeout > 0
    ? timeout
    : DEFAULT_API_TIMEOUT;
}

/* ============================================================================
   DATASET ENDPOINT
============================================================================ */

/**
 * Reads the dataset metadata endpoint.
 *
 * Expected:
 *
 *   /dataset/metadata
 *
 * Also accepts:
 *
 *   dataset/metadata
 *
 * which is normalized to:
 *
 *   /dataset/metadata
 */
function getDatasetMetadataEndpoint(): string {
  const raw =
    import.meta.env
      .VITE_DATASET_METADATA_ENDPOINT;

  if (
    typeof raw !== "string" ||
    raw.trim().length === 0
  ) {
    return DEFAULT_DATASET_METADATA_ENDPOINT;
  }

  const value =
    raw.trim();

  /*
   * The endpoint must be relative.
   *
   * The hostname belongs in VITE_API_URL.
   */
  if (
    /^https?:\/\//i.test(value)
  ) {
    throw new Error(
      "VITE_DATASET_METADATA_ENDPOINT must be a relative path. " +
        "Configure the hostname using VITE_API_URL.",
    );
  }

  if (
    value.startsWith("//")
  ) {
    throw new Error(
      "VITE_DATASET_METADATA_ENDPOINT must be a relative API path.",
    );
  }

  if (
    value.includes("?") ||
    value.includes("#")
  ) {
    throw new Error(
      "VITE_DATASET_METADATA_ENDPOINT must not contain a query string or hash fragment.",
    );
  }

  const normalized =
    value.startsWith("/")
      ? value
      : `/${value}`;

  return (
    normalized.replace(/\/+$/, "") ||
    DEFAULT_DATASET_METADATA_ENDPOINT
  );
}

/* ============================================================================
   BOOLEAN ENVIRONMENT VARIABLES
============================================================================ */

/**
 * Converts a Vite environment variable into a boolean.
 *
 * Supported true values:
 *
 *   true
 *   1
 *   yes
 *   on
 *
 * Supported false values:
 *
 *   false
 *   0
 *   no
 *   off
 *
 * Invalid values use the supplied fallback.
 */
function getBoolean(
  value: string | undefined,
  fallback: boolean,
): boolean {
  if (
    typeof value !== "string"
  ) {
    return fallback;
  }

  switch (
    value.trim().toLowerCase()
  ) {
    case "true":
    case "1":
    case "yes":
    case "on":
      return true;

    case "false":
    case "0":
    case "no":
    case "off":
      return false;

    default:
      if (
        import.meta.env.DEV
      ) {
        console.warn(
          `[CO₂ Emission Predictor] Invalid boolean environment value "${value}". ` +
            `Using fallback: ${fallback}.`,
        );
      }

      return fallback;
  }
}

/* ============================================================================
   APPLICATION METADATA
============================================================================ */

/**
 * Application name.
 */
function getAppName(): string {
  const value =
    import.meta.env.VITE_APP_NAME;

  return (
    typeof value === "string" &&
    value.trim()
      ? value.trim()
      : DEFAULT_APP_NAME
  );
}

/**
 * Application version.
 */
function getAppVersion(): string {
  const value =
    import.meta.env.VITE_APP_VERSION;

  return (
    typeof value === "string" &&
    value.trim()
      ? value.trim()
      : DEFAULT_APP_VERSION
  );
}

/**
 * Application environment.
 */
function getAppEnvironment(): string {
  const value =
    import.meta.env.VITE_APP_ENV;

  return (
    typeof value === "string" &&
    value.trim()
      ? value.trim()
      : import.meta.env.MODE
  );
}

/* ============================================================================
   RESOLVE CONFIGURATION
============================================================================ */

const apiUrl =
  getApiUrl();

const apiTimeout =
  getApiTimeout();

const datasetMetadataEndpoint =
  getDatasetMetadataEndpoint();

const appName =
  getAppName();

const appVersion =
  getAppVersion();

const appEnv =
  getAppEnvironment();

const enableDebug =
  getBoolean(
    import.meta.env.VITE_ENABLE_DEBUG,
    import.meta.env.DEV,
  );

const enableAnalytics =
  getBoolean(
    import.meta.env.VITE_ENABLE_ANALYTICS,
    false,
  );

/* ============================================================================
   PUBLIC CONFIGURATION
============================================================================ */

export const env = Object.freeze({
  /**
   * FastAPI base URL.
   *
   * Development:
   *
   *   http://127.0.0.1:8000
   *
   * Production:
   *
   *   https://co2-emission-predictor-ys3q.onrender.com
   */
  apiUrl,

  /**
   * Application name.
   */
  appName,

  /**
   * Application version.
   */
  appVersion,

  /**
   * Application environment.
   */
  appEnv,

  /**
   * API timeout in milliseconds.
   */
  apiTimeout,

  /**
   * Dataset metadata endpoint path.
   */
  datasetMetadataEndpoint,

  /**
   * Enables frontend debugging.
   */
  enableDebug,

  /**
   * Enables analytics.
   */
  enableAnalytics,
} as const);

/* ============================================================================
   DERIVED ENDPOINTS
============================================================================ */

/**
 * Fully resolved dataset metadata URL.
 *
 * Example:
 *
 * https://co2-emission-predictor-ys3q.onrender.com/dataset/metadata
 */
export const datasetMetadataUrl =
  `${env.apiUrl}${env.datasetMetadataEndpoint}`;

/* ============================================================================
   DEVELOPMENT DIAGNOSTICS
============================================================================ */

/**
 * Debug information is intentionally limited to non-secret
 * client configuration.
 *
 * NEVER put private API keys, database passwords, JWT secrets,
 * or other credentials into VITE_ environment variables.
 */
if (
  import.meta.env.DEV &&
  env.enableDebug
) {
  console.debug(
    "[CO₂ Emission Predictor] Environment configuration loaded.",
    {
      apiUrl: env.apiUrl,
      apiTimeout: env.apiTimeout,
      datasetMetadataEndpoint:
        env.datasetMetadataEndpoint,
      appName: env.appName,
      appVersion: env.appVersion,
      appEnv: env.appEnv,
      enableDebug: env.enableDebug,
      enableAnalytics:
        env.enableAnalytics,
    },
  );
}

/* ============================================================================
   DEFAULT EXPORT
============================================================================ */

export default env;

