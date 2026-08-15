import type {
  ButtonHTMLAttributes,
  ReactNode,
} from "react";

import {
  LoaderCircle,
} from "lucide-react";

interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  loading?: boolean;
}

function Button({
  children,
  loading = false,
  disabled,
  className = "",
  ...props
}: ButtonProps) {
  return (
    <button
      {...props}
      disabled={disabled || loading}
      className={[
        "inline-flex items-center justify-center gap-2",
        "rounded-xl px-5 py-3",
        "text-sm font-semibold",
        "transition-all duration-200",
        "focus:outline-none focus:ring-2",
        "focus:ring-emerald-500 focus:ring-offset-2",
        "disabled:cursor-not-allowed",
        "disabled:opacity-60",
        "bg-slate-950 text-white",
        "hover:bg-slate-800",
        "active:scale-[0.98]",
        className,
      ].join(" ")}
    >
      {loading && (
        <LoaderCircle
          size={18}
          className="animate-spin"
        />
      )}

      {children}
    </button>
  );
}

export default Button;