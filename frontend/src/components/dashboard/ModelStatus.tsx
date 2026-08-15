import {
  CheckCircle2,
  Cpu,
  Database,
  ShieldCheck,
} from "lucide-react";

interface ModelStatusProps {
  online: boolean;
}

function ModelStatus({
  online,
}: ModelStatusProps) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-slate-950">
            Model Status
          </p>

          <p className="mt-1 text-xs text-slate-500">
            Current prediction service
          </p>
        </div>

        <span
          className={[
            "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold",
            online
              ? "bg-emerald-50 text-emerald-700"
              : "bg-red-50 text-red-700",
          ].join(" ")}
        >
          <span
            className={[
              "h-2 w-2 rounded-full",
              online
                ? "bg-emerald-500"
                : "bg-red-500",
            ].join(" ")}
          />

          {online ? "Operational" : "Offline"}
        </span>
      </div>

      <div className="mt-6 space-y-4">
        <div className="flex items-center gap-3">
          <Cpu
            size={18}
            className="text-slate-400"
          />

          <div>
            <p className="text-sm font-medium text-slate-800">
              Algorithm
            </p>

            <p className="text-xs text-slate-500">
              Multiple Linear Regression
            </p>
          </div>

          <CheckCircle2
            size={17}
            className="ml-auto text-emerald-500"
          />
        </div>

        <div className="flex items-center gap-3">
          <Database
            size={18}
            className="text-slate-400"
          />

          <div>
            <p className="text-sm font-medium text-slate-800">
              Dataset
            </p>

            <p className="text-xs text-slate-500">
              Fuel Consumption CO₂ dataset
            </p>
          </div>

          <CheckCircle2
            size={17}
            className="ml-auto text-emerald-500"
          />
        </div>

        <div className="flex items-center gap-3">
          <ShieldCheck
            size={18}
            className="text-slate-400"
          />

          <div>
            <p className="text-sm font-medium text-slate-800">
              API
            </p>

            <p className="text-xs text-slate-500">
              FastAPI prediction service
            </p>
          </div>

          <CheckCircle2
            size={17}
            className="ml-auto text-emerald-500"
          />
        </div>
      </div>
    </div>
  );
}

export default ModelStatus;