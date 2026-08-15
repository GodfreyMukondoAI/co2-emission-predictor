import {
  useCallback,
  useEffect,
  useState,
  type ReactNode,
} from "react";

import Navbar from "./components/layout/Navbar";
import Sidebar from "./components/layout/Sidebar";
import Footer from "./components/layout/Footer";

import HomePage from "./pages/HomePage";
import PredictionPage from "./pages/PredictionPage";
import ModelPage from "./pages/ModelPage";
import DatasetPage from "./pages/DatasetPage";
import AboutPage from "./pages/AboutPage";

import {
  checkHealth,
  getModelInfo,
  predictCO2,
} from "./Services/api";

import type {
  ModelInfoResponse,
  PredictionResponse,
} from "./types/prediction";

/* ==========================================================================
   PAGE TYPES
   ========================================================================== */

export type Page =
  | "home"
  | "prediction"
  | "model"
  | "dataset"
  | "about";

/* ==========================================================================
   CONSTANTS
   ========================================================================== */

const VALID_PAGES: readonly Page[] = [
  "home",
  "prediction",
  "model",
  "dataset",
  "about",
];

/* ==========================================================================
   TYPE GUARDS
   ========================================================================== */

function isPage(
  value: string,
): value is Page {
  return (
    VALID_PAGES.includes(
      value as Page,
    )
  );
}

/* ==========================================================================
   APPLICATION
   ========================================================================== */

function App() {
  /* ==========================================================================
     NAVIGATION STATE
     ========================================================================== */

  const [
    page,
    setPage,
  ] = useState<Page>("home");

  const [
    mobileSidebarOpen,
    setMobileSidebarOpen,
  ] = useState<boolean>(false);

  /* ==========================================================================
     API STATE
     ========================================================================== */

  const [
    apiOnline,
    setApiOnline,
  ] = useState<boolean>(false);

  /* ==========================================================================
     MODEL STATE
     ========================================================================== */

  const [
    modelInfo,
    setModelInfo,
  ] = useState<ModelInfoResponse | null>(
    null,
  );

  const [
    modelLoading,
    setModelLoading,
  ] = useState<boolean>(true);

  /* ==========================================================================
     PREDICTION STATE
     ========================================================================== */

  const [
    prediction,
    setPrediction,
  ] = useState<PredictionResponse | null>(
    null,
  );

  const [
    predictionError,
    setPredictionError,
  ] = useState<string | null>(
    null,
  );

  const [
    predictionLoading,
    setPredictionLoading,
  ] = useState<boolean>(false);

  /* ==========================================================================
     HEALTH CHECK
     ========================================================================== */

  const loadHealth =
    useCallback(
      async (): Promise<void> => {
        try {
          await checkHealth();

          setApiOnline(true);
        } catch (error) {
          console.error(
            "[App] FastAPI health check failed:",
            error,
          );

          setApiOnline(false);
        }
      },
      [],
    );

  /* ==========================================================================
     LOAD MODEL INFORMATION
     ========================================================================== */

  const loadModelInfo =
    useCallback(
      async (): Promise<void> => {
        setModelLoading(true);

        try {
          const response =
            await getModelInfo();

          setModelInfo(response);
        } catch (error) {
          console.error(
            "[App] Failed to load model information:",
            error,
          );

          setModelInfo(null);
        } finally {
          setModelLoading(false);
        }
      },
      [],
    );

  /* ==========================================================================
     INITIAL APPLICATION LOAD
     ========================================================================== */

  useEffect(() => {
    let mounted = true;

    const initializeApplication =
      async (): Promise<void> => {
        const results =
          await Promise.allSettled([
            loadHealth(),
            loadModelInfo(),
          ]);

        if (!mounted) {
          return;
        }

        /*
         * Promise.allSettled is intentionally used here.
         *
         * The application should still render if one backend
         * request fails. For example, the Home page can still
         * display while model metadata is unavailable.
         */
        const failedRequests =
          results.filter(
            (result) =>
              result.status ===
              "rejected",
          );

        if (
          failedRequests.length > 0
        ) {
          console.warn(
            `[App] ${failedRequests.length} initialization request(s) failed.`,
          );
        }
      };

    void initializeApplication();

    return () => {
      mounted = false;
    };
  }, [
    loadHealth,
    loadModelInfo,
  ]);

  /* ==========================================================================
     CO₂ PREDICTION
     ========================================================================== */

  const handlePrediction =
    useCallback(
      async (
        engineSize: number,
        fuelConsumption: number,
      ): Promise<void> => {
        /*
         * Validate inputs before making the API request.
         * The backend remains the final authority for validation.
         */

        if (
          !Number.isFinite(engineSize) ||
          !Number.isFinite(
            fuelConsumption,
          )
        ) {
          setPrediction(null);

          setPredictionError(
            "Engine size and fuel consumption must be valid numbers.",
          );

          return;
        }

        setPredictionLoading(true);
        setPredictionError(null);

        try {
          const result =
            await predictCO2({
              engine_size:
                engineSize,

              fuel_consumption_mpg:
                fuelConsumption,
            });

          setPrediction(result);
        } catch (error) {
          console.error(
            "[App] CO₂ prediction failed:",
            error,
          );

          setPrediction(null);

          setPredictionError(
            error instanceof Error
              ? error.message
              : "Unable to generate CO₂ prediction.",
          );
        } finally {
          setPredictionLoading(false);
        }
      },
      [],
    );

  /* ==========================================================================
     NAVIGATION
     ========================================================================== */

  const handleNavigate =
    useCallback(
      (nextPage: string): void => {
        if (!isPage(nextPage)) {
          console.warn(
            `[App] Ignoring invalid navigation target: "${nextPage}"`,
          );

          return;
        }

        setPage(nextPage);

        /*
         * Always close the mobile sidebar after navigation.
         */
        setMobileSidebarOpen(false);

        /*
         * Return focus to the main application area
         * where possible for better keyboard accessibility.
         */
        window.setTimeout(() => {
          const mainContent =
            document.getElementById(
              "main-content",
            );

          mainContent?.focus({
            preventScroll: true,
          });
        }, 0);
      },
      [],
    );

  /* ==========================================================================
     MOBILE SIDEBAR
     ========================================================================== */

  const handleOpenMobileSidebar =
    useCallback((): void => {
      setMobileSidebarOpen(true);
    }, []);

  const handleCloseMobileSidebar =
    useCallback((): void => {
      setMobileSidebarOpen(false);
    }, []);

  /* ==========================================================================
     PAGE RENDERING
     ========================================================================== */

  const renderPage =
    useCallback((): ReactNode => {
      switch (page) {
        /* ----------------------------------------------------------------------
           HOME
           ---------------------------------------------------------------------- */

        case "home":
          return (
            <HomePage
              apiOnline={apiOnline}
              modelInfo={modelInfo}
              modelLoading={modelLoading}
              onNavigate={handleNavigate}
            />
          );

        /* ----------------------------------------------------------------------
           PREDICTION
           ---------------------------------------------------------------------- */

        case "prediction":
          return (
            <PredictionPage
              prediction={prediction}
              error={predictionError}
              loading={predictionLoading}
              onPredict={handlePrediction}
            />
          );

        /* ----------------------------------------------------------------------
           MODEL
           ---------------------------------------------------------------------- */

        case "model":
          return (
            <ModelPage
              modelInfo={modelInfo}
              modelLoading={modelLoading}
            />
          );

        /* ----------------------------------------------------------------------
           DATASET
           ---------------------------------------------------------------------- */

        case "dataset":
          return <DatasetPage />;

        /* ----------------------------------------------------------------------
           ABOUT
           ---------------------------------------------------------------------- */

        case "about":
          return <AboutPage />;

        /* ----------------------------------------------------------------------
           FALLBACK
           ---------------------------------------------------------------------- */

        default:
          return (
            <HomePage
              apiOnline={apiOnline}
              modelInfo={modelInfo}
              modelLoading={modelLoading}
              onNavigate={handleNavigate}
            />
          );
      }
    }, [
      page,
      apiOnline,
      modelInfo,
      modelLoading,
      prediction,
      predictionError,
      predictionLoading,
      handleNavigate,
      handlePrediction,
    ]);

  /* ==========================================================================
     APPLICATION LAYOUT
     ========================================================================== */

  return (
    <div className="min-h-screen bg-slate-50 text-slate-950">
      <div className="flex min-h-screen">

        {/* ====================================================================
            SIDEBAR
            ==================================================================== */}

        <Sidebar
          currentPage={page}
          onNavigate={handleNavigate}
          mobileOpen={
            mobileSidebarOpen
          }
          onClose={
            handleCloseMobileSidebar
          }
        />

        {/* ====================================================================
            MAIN APPLICATION AREA
            ==================================================================== */}

        <div className="flex min-w-0 flex-1 flex-col">

          {/* ==================================================================
              NAVBAR
              ================================================================== */}

          <Navbar
            onMenuClick={
              handleOpenMobileSidebar
            }
          />

          {/* ==================================================================
              MAIN CONTENT
              ================================================================== */}

          <main
            id="main-content"
            tabIndex={-1}
            className="flex-1 outline-none"
          >
            <div className="mx-auto w-full max-w-[1500px] px-4 py-8 sm:px-6 lg:px-8">
              {renderPage()}
            </div>
          </main>

          {/* ==================================================================
              FOOTER
              ================================================================== */}

          <Footer />

        </div>
      </div>
    </div>
  );
}

/* ==========================================================================
   EXPORT
   ========================================================================== */

export default App;