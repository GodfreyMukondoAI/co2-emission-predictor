import {
  Activity,
  Menu,
  Leaf,
} from "lucide-react";

interface NavbarProps {
  onMenuClick: () => void;
}

function Navbar({
  onMenuClick,
}: NavbarProps) {
  return (
    <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/95 backdrop-blur">
      <div className="flex h-16 items-center justify-between px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onMenuClick}
            className="rounded-lg p-2 text-slate-600 hover:bg-slate-100 lg:hidden"
            aria-label="Open navigation"
          >
            <Menu size={22} />
          </button>

          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-600 text-white">
              <Leaf size={19} />
            </div>

            <div>
              <p className="text-sm font-bold text-slate-950">
                CO₂ Predictor
              </p>

              <p className="hidden text-xs text-slate-500 sm:block">
                Machine Learning Platform
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="hidden items-center gap-2 rounded-full bg-emerald-50 px-3 py-1.5 text-xs font-semibold text-emerald-700 sm:flex">
            <Activity size={14} />
            AI Model Online
          </div>

          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-slate-900 text-xs font-bold text-white">
            AI
          </div>
        </div>
      </div>
    </header>
  );
}

export default Navbar;