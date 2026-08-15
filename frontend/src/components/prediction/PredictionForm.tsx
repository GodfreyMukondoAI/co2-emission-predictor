import {
  AlertCircle,
  Gauge,
  Fuel,
  Sparkles,
} from "lucide-react";
import { useState, type FormEvent } from "react";

import Button from "../common/Button";
import Card from "../common/Card";

interface PredictionFormProps {
  onSubmit: (
    engineSize: number,
    fuelConsumption: number,
  ) => Promise<void>;
  loading: boolean;
}

function PredictionForm({
  onSubmit,
  loading,
}: PredictionFormProps) {
  const [engineSize, setEngineSize] = useState("3.0");
  const [fuelConsumption, setFuelConsumption] =
    useState("25");
  const [validationError, setValidationError] =
    useState("");

  async function handleSubmit(
    event: FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();

    if (loading) {
      return;
    }

    setValidationError("");

    const engine = Number(engineSize);
    const fuel = Number(fuelConsumption);

    if (!engineSize.trim()) {
      setValidationError(
        "Please enter the engine size.",
      );
      return;
    }

    if (!Number.isFinite(engine) || engine <= 0) {
      setValidationError(
        "Engine size must be greater than zero.",
      );
      return;
    }

    if (!fuelConsumption.trim()) {
      setValidationError(
        "Please enter the fuel consumption.",
      );
      return;
    }

    if (!Number.isFinite(fuel) || fuel <= 0) {
      setValidationError(
        "Fuel consumption must be greater than zero.",
      );
      return;
    }

    try {
      await onSubmit(engine, fuel);
    } catch {
      // The parent component is responsible for
      // displaying API errors.
    }
  }

  return (
    <Card className="overflow-hidden">
      <div className="border-b border-slate-200 bg-gradient-to-br from-white to-slate-50 px-6 py-6 sm:px-7">
        <div className="flex items-start gap-4">
          <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-emerald-100 text-emerald-700">
            <Sparkles
              size={21}
              strokeWidth={2}
              aria-hidden="true"
            />
          </div>

          <div>
            <p className="text-xs font-bold uppercase tracking-[0.16em] text-emerald-600">
              Vehicle inputs
            </p>

            <h2 className="mt-1 text-xl font-bold tracking-tight text-slate-950">
              Enter vehicle details
            </h2>

            <p className="mt-2 max-w-xl text-sm leading-6 text-slate-500">
              Provide the engine size and combined fuel
              consumption to generate a CO₂ emissions
              estimate.
            </p>
          </div>
        </div>
      </div>

      <form
        onSubmit={handleSubmit}
        noValidate
        className="space-y-6 p-6 sm:p-7"
      >
        {/* Engine size */}
        <div>
          <label
            htmlFor="engine-size"
            className="mb-2 block text-sm font-semibold text-slate-800"
          >
            Engine Size
          </label>

          <div className="relative">
            <Gauge
              size={19}
              strokeWidth={2}
              className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400"
              aria-hidden="true"
            />

            <input
              id="engine-size"
              name="engine-size"
              type="number"
              inputMode="decimal"
              min="0.1"
              step="0.1"
              value={engineSize}
              onChange={(event) => {
                setEngineSize(event.target.value);

                if (validationError) {
                  setValidationError("");
                }
              }}
              disabled={loading}
              aria-invalid={Boolean(validationError)}
              aria-describedby={
                validationError
                  ? "prediction-validation-error"
                  : undefined
              }
              className="w-full rounded-xl border border-slate-300 bg-white py-3.5 pl-11 pr-14 text-sm font-medium text-slate-950 shadow-sm outline-none transition placeholder:text-slate-400 hover:border-slate-400 focus:border-emerald-500 focus:ring-4 focus:ring-emerald-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:opacity-70"
              placeholder="3.0"
            />

            <span className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-400">
              L
            </span>
          </div>

          <p className="mt-2 text-xs text-slate-400">
            Enter the engine displacement in litres.
          </p>
        </div>

        {/* Fuel consumption */}
        <div>
          <label
            htmlFor="fuel-consumption"
            className="mb-2 block text-sm font-semibold text-slate-800"
          >
            Combined Fuel Consumption
          </label>

          <div className="relative">
            <Fuel
              size={19}
              strokeWidth={2}
              className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400"
              aria-hidden="true"
            />

            <input
              id="fuel-consumption"
              name="fuel-consumption"
              type="number"
              inputMode="decimal"
              min="0.1"
              step="0.1"
              value={fuelConsumption}
              onChange={(event) => {
                setFuelConsumption(event.target.value);

                if (validationError) {
                  setValidationError("");
                }
              }}
              disabled={loading}
              aria-invalid={Boolean(validationError)}
              aria-describedby={
                validationError
                  ? "prediction-validation-error"
                  : undefined
              }
              className="w-full rounded-xl border border-slate-300 bg-white py-3.5 pl-11 pr-16 text-sm font-medium text-slate-950 shadow-sm outline-none transition placeholder:text-slate-400 hover:border-slate-400 focus:border-emerald-500 focus:ring-4 focus:ring-emerald-100 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:opacity-70"
              placeholder="25"
            />

            <span className="pointer-events-none absolute right-3.5 top-1/2 -translate-y-1/2 text-xs font-bold text-slate-400">
              MPG
            </span>
          </div>

          <p className="mt-2 text-xs text-slate-400">
            Enter the vehicle's combined fuel consumption.
          </p>
        </div>

        {/* Validation error */}
        {validationError && (
          <div
            id="prediction-validation-error"
            role="alert"
            className="flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3.5 text-sm text-red-700"
          >
            <AlertCircle
              size={18}
              strokeWidth={2}
              className="mt-0.5 shrink-0"
              aria-hidden="true"
            />

            <div>
              <p className="font-semibold">
                Check your input
              </p>

              <p className="mt-0.5 text-red-600">
                {validationError}
              </p>
            </div>
          </div>
        )}

        {/* Submit */}
        <div className="pt-1">
          <Button
            type="submit"
            loading={loading}
            disabled={loading}
            className="w-full bg-emerald-600 shadow-sm hover:bg-emerald-700 focus:ring-4 focus:ring-emerald-200"
          >
            {loading
              ? "Generating Prediction..."
              : "Generate Prediction"}
          </Button>
        </div>

        {/* Small information note */}
        <div className="flex items-start gap-2 border-t border-slate-100 pt-5">
          <Sparkles
            size={15}
            className="mt-0.5 shrink-0 text-emerald-500"
            aria-hidden="true"
          />

          <p className="text-xs leading-5 text-slate-500">
            The prediction is generated by a trained
            Multiple Linear Regression model using
            engine size and combined fuel consumption.
          </p>
        </div>
      </form>
    </Card>
  );
}

export default PredictionForm;