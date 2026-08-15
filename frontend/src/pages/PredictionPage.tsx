import {
  BrainCircuit,
} from "lucide-react";

import PredictionForm from "../components/prediction/PredictionForm";
import PredictionResult from "../components/prediction/PredictionResult";
import type { PredictionResponse } from "../types/prediction";

interface PredictionPageProps {
  prediction: PredictionResponse | null;
  error: string | null;
  loading: boolean;
  onPredict: (
    engineSize: number,
    fuelConsumption: number,
  ) => Promise<void>;
}

function PredictionPage({
  prediction,
  error,
  loading,
  onPredict,
}: PredictionPageProps) {
  return (
    <div className="space-y-8">
      <section>
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
            <BrainCircuit size={24} />
          </div>

          <div>
            <p className="text-sm font-semibold text-emerald-600">
              PREDICTION ENGINE
            </p>

            <h1 className="mt-1 text-3xl font-bold tracking-tight text-slate-950">
              Vehicle CO₂ Prediction
            </h1>

            <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
              Submit vehicle characteristics to the
              FastAPI backend and receive a prediction
              from the trained regression model.
            </p>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <PredictionForm
          onSubmit={onPredict}
          loading={loading}
        />

        <PredictionResult
          prediction={prediction}
          error={error}
        />
      </section>
    </div>
  );
}

export default PredictionPage;