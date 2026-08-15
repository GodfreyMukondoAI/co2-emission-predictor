import {
  ArrowRight,
  Leaf,
} from "lucide-react";

interface PredictionCardProps {
  onClick: () => void;
}

function PredictionCard({
  onClick,
}: PredictionCardProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group w-full rounded-2xl border border-slate-200 bg-white p-6 text-left shadow-sm transition hover:-translate-y-1 hover:border-emerald-200 hover:shadow-lg"
    >
      <div className="flex items-start justify-between">
        <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-50 text-emerald-600">
          <Leaf size={23} />
        </div>

        <ArrowRight
          size={20}
          className="text-slate-400 transition group-hover:translate-x-1 group-hover:text-emerald-600"
        />
      </div>

      <h3 className="mt-6 text-lg font-bold text-slate-950">
        Predict vehicle emissions
      </h3>

      <p className="mt-2 text-sm leading-6 text-slate-500">
        Use the trained machine-learning model to
        estimate CO₂ emissions from basic vehicle
        specifications.
      </p>

      <p className="mt-5 text-sm font-semibold text-emerald-600">
        Start prediction
      </p>
    </button>
  );
}

export default PredictionCard;