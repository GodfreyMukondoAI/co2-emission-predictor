/**
 * ============================================================
 * APPLICATION ENVIRONMENT CONFIGURATION
 * ============================================================
 *
 * Centralized, type-safe access to Vite environment variables.
 *
 * Environment variables must be prefixed with VITE_ in order
 * for Vite to expose them to the browser.
 * ============================================================
 */

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  readonly VITE_APP_NAME?: string;
  readonly VITE_APP_VERSION?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/**
 * Removes trailing slashes from URLs.
 */
function normalizeUrl(value: string): string {
  return value.trim().replace(/\/+$/, "");
}

/**
 * Reads and validates the API URL.
 */
function getApiUrl(): string {
  const value = import.meta.env.VITE_API_URL;

  if (!value || !value.trim()) {
    throw new Error(
      "VITE_API_URL is not configured. " +
        "Create a .env.development or .env.production file " +
        "in the Vite project root and define VITE_API_URL.",
    );
  }

  const normalized = normalizeUrl(value);

  try {
    const url = new URL(normalized);

    if (!["http:", "https:"].includes(url.protocol)) {
      throw new Error(
        "VITE_API_URL must use HTTP or HTTPS.",
      );
    }
  } catch {
    throw new Error(
      "VITE_API_URL contains an invalid URL.",
    );
  }

  return normalized;
}

export const env = Object.freeze({
  apiUrl: getApiUrl(),

  appName:
    import.meta.env.VITE_APP_NAME?.trim() ||
    "CO₂ Emission Predictor",

  appVersion:
    import.meta.env.VITE_APP_VERSION?.trim() ||
    "1.0.0",
});