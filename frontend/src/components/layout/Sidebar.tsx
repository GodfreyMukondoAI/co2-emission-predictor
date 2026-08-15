import {
  BarChart3,
  Database,
  Info,
  LayoutDashboard,
  Leaf,
  Settings2,
  X,
} from "lucide-react";

interface SidebarProps {
  currentPage: string;
  onNavigate: (page: string) => void;
  mobileOpen: boolean;
  onClose: () => void;
}

const navigation = [
  {
    id: "home",
    label: "Overview",
    icon: LayoutDashboard,
  },
  {
    id: "prediction",
    label: "Prediction",
    icon: Leaf,
  },
  {
    id: "model",
    label: "Model",
    icon: BarChart3,
  },
  {
    id: "dataset",
    label: "Dataset",
    icon: Database,
  },
  {
    id: "about",
    label: "About",
    icon: Info,
  },
];

function Sidebar({
  currentPage,
  onNavigate,
  mobileOpen,
  onClose,
}: SidebarProps) {
  return (
    <>
      {mobileOpen && (
        <button
          type="button"
          className="fixed inset-0 z-40 bg-slate-950/40 lg:hidden"
          onClick={onClose}
          aria-label="Close navigation"
        />
      )}

      <aside
        className={[
          "fixed left-0 top-0 z-50 h-screen w-72",
          "border-r border-slate-200 bg-white",
          "transition-transform duration-300",
          "lg:static lg:z-auto lg:translate-x-0",
          mobileOpen
            ? "translate-x-0"
            : "-translate-x-full",
        ].join(" ")}
      >
        <div className="flex h-16 items-center justify-between border-b border-slate-200 px-5">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-emerald-600 text-white">
              <Leaf size={19} />
            </div>

            <div>
              <p className="font-bold text-slate-950">
                CO₂ Predictor
              </p>

              <p className="text-xs text-slate-500">
                ML Platform
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 lg:hidden"
            aria-label="Close navigation"
          >
            <X size={20} />
          </button>
        </div>

        <nav className="space-y-1 p-4">
          <p className="mb-3 px-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Workspace
          </p>

          {navigation.map((item) => {
            const Icon = item.icon;
            const active =
              currentPage === item.id;

            return (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  onNavigate(item.id);
                  onClose();
                }}
                className={[
                  "flex w-full items-center gap-3 rounded-xl px-3 py-3",
                  "text-left text-sm font-medium transition",
                  active
                    ? "bg-emerald-50 text-emerald-700"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-950",
                ].join(" ")}
              >
                <Icon size={19} />

                <span>{item.label}</span>

                {active && (
                  <span className="ml-auto h-2 w-2 rounded-full bg-emerald-600" />
                )}
              </button>
            );
          })}
        </nav>

        <div className="absolute bottom-0 left-0 right-0 border-t border-slate-200 p-4">
          <div className="rounded-xl bg-slate-50 p-4">
            <div className="flex items-center gap-2">
              <Settings2
                size={17}
                className="text-slate-500"
              />

              <span className="text-sm font-semibold text-slate-800">
                Model Version
              </span>
            </div>

            <p className="mt-2 text-xs text-slate-500">
              Multiple Linear Regression
            </p>

            <p className="mt-1 text-xs font-medium text-emerald-600">
              v1.0.0
            </p>
          </div>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;