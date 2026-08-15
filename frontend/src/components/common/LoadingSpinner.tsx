import {
  LoaderCircle,
} from "lucide-react";

interface LoadingSpinnerProps {
  label?: string;
}

function LoadingSpinner({
  label = "Loading...",
}: LoadingSpinnerProps) {
  return (
    <div className="flex items-center justify-center gap-3 py-10 text-sm text-slate-500">
      <LoaderCircle
        size={20}
        className="animate-spin"
      />

      <span>{label}</span>
    </div>
  );
}

export default LoadingSpinner;