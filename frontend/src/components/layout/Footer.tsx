import {
  Leaf,
  ShieldCheck,
} from "lucide-react";

function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white">
      <div className="flex flex-col gap-3 px-6 py-6 text-sm text-slate-500 sm:flex-row sm:items-center sm:justify-between lg:px-8">
        <div className="flex items-center gap-2">
          <Leaf size={16} />

          <span>
            CO₂ Emission Predictor
          </span>
        </div>

        <div className="flex items-center gap-2">
          <ShieldCheck size={16} />

          <span>
            Machine learning estimate • v1.0.0
          </span>
        </div>
      </div>
    </footer>
  );
}

export default Footer;